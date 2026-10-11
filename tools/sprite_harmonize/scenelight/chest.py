"""Chest stage (reference pack only): the reference's light on the breasts, cut by the DRAWN lines, at the reference's strength.

Why: the pack's interior transfer is soft (SIGMA_IN = 1.5 % of figure height), so the chest kept the drawing's
frontal highlight and read as flat (user: "the inside of the chest does not show volume according to the light").
A first version put the light in the right places but was too weak ("the method is right, the light is weaker than
the reference"): it treated the cup print as line art (never lit, and left out of the frame's local mean, so the
delta came out negative over the cups), blurred the narrow highlights and squashed them with the print softening.
What it does, per frame, using only that frame's own drawing for geometry (RESEARCH.md 12-13):
1. find the two drawn cups (bright low-chroma cloth in the upper torso, cut by the line art) -> a zone around them
   (upper-breast skin, cleavage, cups, under-bust); nothing outside it changes;
2. only TRUE line art is "line" (dark pixels, or thin strokes that are themselves dark; tested on a slightly blurred
   L* so a frame's resampling can not flip the decision); the cup print is cloth: it takes the light like the cloth
   around it, so its relative contrast is kept;
3. the cup print is softened first (tex), then the reference's local light minus the frame's local light is added,
   low-passed per material (skin onto skin, cloth onto cloth) by a recursive edge-aware filter whose barriers are the
   drawn lines, at the scale it is seen (0.3 % of figure height on cloth, 0.2 % on skin) -> highlights stop at the
   cleavage line / cup outlines like the reference's;
4. the frame's light is measured with its texture marks closed over, so a mark is not "filled in" by the light;
5. anti-aliased line pixels take only their surface share of the light of the open surface beside them; line art
   never gets lighter; mixed skin / dark-cloth pixels do not borrow skin light;
6. a broad level term (1.5 % of figure height, +-2.5 L*) on the chest region gives the pixels that can take light the
   brightness the drawn lines can not (our lines are much darker than the painted reference's);
7. no added light above the reference's own local level + 3 L*, a soft shoulder above 94 L*; alpha never changes.
"""
import numpy as np
import cv2

from .color import to_lab, from_lab, materials

CHEST = {
    "sig": 0.003, "sig_skin": 0.002,   # light scale on cloth / skin (fraction of figure height)
    "barrier": 40.0,    # step cost of a line pixel for the edge-aware filter
    "tex": 0.4,         # cup print contrast kept (1 = unchanged)
    "tex_sig": 0.008,   # scale the print is softened around
    "line_L": 40.0, "dark_rng": 8.0,   # L* at / below which a pixel is line art (ramp over dark_rng)
    "lsig": 0.8,        # thin-stroke test on L* blurred by this many px at a 1391 px figure (resampling-tolerant)
    "thin_lo": 50.0, "thin_hi": 58.0,  # thin strokes darker than thin_lo are line art (ramp to thin_hi) ...
    "bh_raw": (5.0, 5.0), "bh_blur": (4.0, 4.0),   # ... when this much darker than around them (per pixel / blurred L*)
    "er_reach": 58.0,   # hysteresis: a contrast stroke darker than this also protects its anti-aliased edge
    "guard_L": 54.0,    # thin contrast strokes darker than this never get lighter
    "cup_in": 0.0022, "cup_out": 0.0015,   # cup outline band: dark strokes there are line art by contrast
    "stroke_Lmax": 78.0, "stroke_Lraw": 74.0,   # ... but never the light print (L* 75-85)
    "mf_close": 7, "mf_k": 2.5,        # marks closed over (px) / their darkness put back at mf_k x the light scale
    "c_min": 0.35,      # filtered support below this shrinks the delta
    "edge_fs": 1.5, "own_sup": 0.015, "own_prot": 0.35,   # line-edge fill; narrow cloth keeps its own (positive) light
    "lvl_sig": 0.015, "lvl_max": 2.5, "lvl_soft": 0.004,  # chest level term (0 = off)
    "hot_sig": 0.001, "hot_m": 3.0,    # added light capped at the reference's local level + hot_m
    "knee": 94.0, "Lmax": 97.0,        # highlight shoulder
    "up": 0.045, "down": 0.03, "side": 0.01,  # zone around the cups (fraction of figure height); not up to the neck
    "feather": 0.012,
}


def rf_filter(I, cost, sigma, iters=3):
    """Recursive domain-transform filter (Gastal & Oliveira 2011). I: HxWxC, cost: HxW (1 = free, large = barrier)."""
    J = I.astype(np.float32).copy()
    ch = 0.5 * (cost[:, 1:] + cost[:, :-1])
    cv_ = 0.5 * (cost[1:, :] + cost[:-1, :])
    for i in range(iters):
        sH = sigma * np.sqrt(3) * 2 ** (iters - i - 1) / np.sqrt(4 ** iters - 1)
        a = np.exp(-np.sqrt(2) / max(sH, 1e-3))
        Vh = (a ** ch)[..., None].astype(np.float32)
        Vv = (a ** cv_)[..., None].astype(np.float32)
        for x in range(1, J.shape[1]):
            J[:, x] += Vh[:, x - 1] * (J[:, x - 1] - J[:, x])
        for x in range(J.shape[1] - 2, -1, -1):
            J[:, x] += Vh[:, x] * (J[:, x + 1] - J[:, x])
        for y in range(1, J.shape[0]):
            J[y] += Vv[y - 1] * (J[y - 1] - J[y])
        for y in range(J.shape[0] - 2, -1, -1):
            J[y] += Vv[y] * (J[y + 1] - J[y])
    return J


def line_weight(L, line_L=40.0):
    """Drawn line art and its anti-aliased edge: very dark pixels, or thin structures darker than their surroundings."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bh = cv2.morphologyEx(L.astype(np.float32), cv2.MORPH_CLOSE, k) - L
    return np.maximum(np.clip((line_L + 8 - L) / 16.0, 0, 1), np.clip((bh - 5) / 10.0, 0, 1)).astype(np.float32)


def find_cups(frame):
    """Two drawn cups of the neutral-lit frame, or (None, Hf, reason). Watershed seeded in the cloth, bounded by lines."""
    a = frame[..., 3] > 0.5
    ys = np.nonzero(a.any(1))[0]
    if len(ys) < 2:
        return None, 0, "empty frame"
    y0, Hf = ys.min(), ys.max() - ys.min()
    lab = to_lab(np.clip(frame[..., :3], 0, 1))
    L, C = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    line = cv2.dilate((L < 45).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    band = np.zeros_like(a)
    band[y0 + int(0.18 * Hf): y0 + int(0.40 * Hf)] = True
    cloth = a & (C < 9) & (L > 62) & ~line & band
    k = max(3, int(0.006 * Hf) | 1)
    cloth = cv2.morphologyEx(cloth.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab_, st, cen = cv2.connectedComponentsWithStats(cloth, 8)
    comps = sorted(range(1, n), key=lambda i: -st[i, 4])[:2]
    if len(comps) < 2:
        return None, Hf, "no cups found"
    (xa, ya), (xb, yb) = cen[comps[0]], cen[comps[1]]
    if (min(st[comps[0], 4], st[comps[1], 4]) < 0.0008 * Hf ** 2 or abs(ya - yb) > 0.05 * Hf
            or not 0.03 * Hf < abs(xa - xb) < 0.25 * Hf):
        return None, Hf, "cups not side by side"
    two = sorted(comps, key=lambda i: cen[i][0])
    markers = np.zeros(a.shape, np.int32)
    er = lambda m, s: cv2.erode(m.astype(np.uint8), np.ones((s, s), np.uint8)) > 0
    for j, i in enumerate(two, 1):
        markers[er(lab_ == i, 5)] = j
    wide = np.zeros_like(a)
    wide[y0 + int(0.24 * Hf): y0 + int(0.42 * Hf)] = True
    markers[er(a & (C > 14) & (L > 58), 5) & (markers == 0)] = 3      # skin
    markers[(~er(a, 9)) & (markers == 0)] = 5                          # silhouette edge and outside
    markers[~wide & (markers == 0)] = 4
    grad = cv2.morphologyEx((L / 100 * 255).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ws = cv2.watershed(np.dstack([grad] * 3), markers)
    cups = [ws == 1, ws == 2]
    if min(c.sum() for c in cups) < 0.0008 * Hf ** 2:
        return None, Hf, "cups not side by side"
    return cups, Hf, "ok"


def true_line_weight(L, P, bh_lo, bh_rng):
    """Line art: very dark pixels, or thin strokes darker than their surroundings that are themselves dark.
    Light thin strokes (the cup print) are not line art."""
    L = L.astype(np.float32)
    bh = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) - L
    dark = np.clip((P["line_L"] + P["dark_rng"] - L) / P["dark_rng"], 0, 1)
    thin = np.clip((bh - bh_lo) / bh_rng, 0, 1) * np.clip((P["thin_hi"] - L) / (P["thin_hi"] - P["thin_lo"]), 0, 1)
    return np.maximum(dark, thin).astype(np.float32)


def ink_share(L, core):
    """Ink coverage around line art: (surface - L) / (surface - ink); 1 on the line, partial on its anti-aliased edge."""
    L = L.astype(np.float32)
    surf = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ink = cv2.erode(L, np.ones((5, 5), np.uint8))
    cov = np.clip((surf - L) / np.maximum(surf - ink, 4.0), 0, 1)
    near = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return np.where(near, cov, 0).astype(np.float32)


def _cups_to_outline(Lf, both, a, Hf, skin, line_L, reach=0.006):
    """The cups grown through cloth to their drawn outline (find_cups' watershed can stop a few px inside it at a
    print edge), plus the dark outline met on the way. Growth stops at the outline and at skin."""
    dark = ((Lf < 50) | ((line_weight(Lf, line_L) > 0.5) & (Lf < 62))) & a
    free = (~dark & a & (skin < 0.5)).astype(np.uint8)
    g = both.astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(max(1, int(round(reach * Hf)))):
        g = cv2.dilate(g, k3) & free | g
    h, w = g.shape                              # outside of the grown cups (enclosed print marks are not outline)
    inv = (1 - g).astype(np.uint8)
    ffm = np.zeros((h + 2, w + 2), np.uint8)
    seeds = ([(x, y) for y in (0, h - 1) for x in range(0, w, 7)] + [(x, y) for x in (0, w - 1) for y in range(0, h, 7)])
    for sx, sy in seeds:
        if inv[sy, sx] == 1:
            cv2.floodFill(inv, ffm, (sx, sy), 2)
    outside = inv == 2
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    ring = (cv2.dilate(g, k3) > 0) & dark & (cv2.dilate(outside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    keep[np.unique(lab[ring])] = True
    keep &= st[:, 4] >= max(3, 0.01 * Hf)
    keep[0] = False
    return (g > 0) | (keep[lab] & (cv2.dilate(g, k3) > 0))


def chest_light(frame, out, ref, conf=None, params=None):
    """frame: neutral-lit RGBA; out: relit RGBA (pack result); ref: reference RGBA warped into this frame;
    conf: HxW 0..1 trust in the warp (None = 1). Returns (rgba, info). Nothing outside the chest zone changes."""
    P = dict(CHEST, **(params or {}))
    cups, Hf, why = find_cups(frame)
    if cups is None:
        return out, {"chest": why}
    a_full = frame[..., 3] > 0.5
    both_f = cups[0] | cups[1]
    ys, xs = np.nonzero(both_f)
    H, W = a_full.shape
    zy0, zy1 = max(0, int(ys.min() - P["up"] * Hf)), min(H, int(ys.max() + P["down"] * Hf))
    zx0, zx1 = max(0, int(xs.min() - P["side"] * Hf)), min(W, int(xs.max() + P["side"] * Hf))
    m = int(0.03 * Hf) + 2
    y0, y1, x0, x1 = max(zy0 - m, 0), min(zy1 + m, H), max(zx0 - m, 0), min(zx1 + m, W)
    sl = (slice(y0, y1), slice(x0, x1))
    f, o, r = frame[sl], out[sl], np.clip(ref[sl], 0, 1)
    a, both = a_full[sl], both_f[sl]
    labf, labo, labr = to_lab(np.clip(f[..., :3], 0, 1)), to_lab(np.clip(o[..., :3], 0, 1)), to_lab(r[..., :3])
    Lf = labf[..., 0]
    mats_f, mats_r = materials(f), materials(r)
    s = Hf / 1391.0                             # pixel sizes below are tuned at a 1391 px figure

    # --- what counts as line art (decisions robust to how a frame's resampling rendered a thin stroke)
    wl_raw = true_line_weight(Lf, P, *P["bh_raw"])   # per pixel
    wl_tl = wl_raw
    wh = np.clip((line_weight(Lf, P["line_L"]) - 0.5) / 0.3, 0, 1) * np.clip((P["er_reach"] + 4 - Lf) / 8, 0, 1)
    wl_raw = np.maximum(wl_raw, wh).astype(np.float32)          # hysteresis: contrast strokes near the threshold
    Ll = cv2.GaussianBlur(Lf.astype(np.float32), (0, 0), P["lsig"] * s)
    wl_px = np.maximum(true_line_weight(Ll, P, *P["bh_blur"]), np.clip((P["line_L"] + P["dark_rng"] - Lf) / P["dark_rng"], 0, 1))
    olext = _cups_to_outline(Lf, both, a, Hf, mats_f["skin"], P["line_L"])
    bu = both.astype(np.uint8)                  # cup outline band: dark strokes there are line art by contrast
    sd = cv2.distanceTransform(bu, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - bu, cv2.DIST_L2, 3)
    wband = np.clip(P["cup_in"] * Hf - sd + 0.5, 0, 1) * np.clip(sd + P["cup_out"] * Hf + 0.5, 0, 1)
    wst = (line_weight(Ll, P["line_L"]) * wband * np.clip((P["stroke_Lmax"] + 4 - Ll) / 8, 0, 1)
           * np.clip((P["stroke_Lraw"] + 4 - Lf) / 8, 0, 1))
    wl_px = np.maximum(wl_px, wst).astype(np.float32)
    line = wl_px > 0.5
    lmask = line.astype(np.float32)
    cost = 1 + P["barrier"] * np.clip(cv2.GaussianBlur(lmask, (0, 0), 0.8) * 2, 0, 1) + 200 * (~a)

    # --- the cup print softened first, so the light added afterwards is not squashed
    base = labo.copy()
    if P["tex"] < 1:
        cupm = (both & ~line).astype(np.float32)
        wl0 = cv2.dilate(np.maximum(wl_px, wl_raw), np.ones((5, 5), np.uint8))
        cupw = cv2.GaussianBlur(both.astype(np.float32), (0, 0), 1.5) * (1 - wl0) * mats_f["light"]
        sb = P["tex_sig"] * Hf
        lpb = cv2.GaussianBlur(base[..., 0] * cupm, (0, 0), sb) / np.maximum(cv2.GaussianBlur(cupm, (0, 0), sb), 1e-4)
        base[..., 0] += (1 - P["tex"]) * cupw * (lpb - base[..., 0])

    # --- reference light minus frame light, per material, cut by the drawn lines
    D = np.zeros_like(labo)
    Wc = np.zeros(a.shape, np.float32)
    cupsoft = cv2.GaussianBlur(both.astype(np.float32), (0, 0), 1.5)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["mf_close"], P["mf_close"]))
    wt = line_weight(Lf, P["line_L"]) * (1 - lmask)              # texture marks (not line art), closed over
    Lmf = base[..., 0] + wt * (cv2.morphologyEx(base[..., 0].astype(np.float32), cv2.MORPH_CLOSE, kc) - base[..., 0])
    for k in ("skin", "light"):
        mo = (a & (mats_f[k] > 0.5)).astype(np.float32) * (1 - lmask)
        mr = mo * (r[..., 3] > 0.5) * (mats_r["dark"] < 0.5)    # never pull in the reference's dark cloth
        sk = (P["sig_skin"] if k == "skin" else P["sig"]) * Hf
        so5 = rf_filter(np.dstack([base * mo[..., None], (Lmf * mo)[..., None], mo]), cost, sk)
        so = np.dstack([so5[..., :3], so5[..., 4:]])
        lo = so[..., :3] / np.maximum(so[..., 3:], 1e-4)
        lmf = so5[..., 3] / np.maximum(so5[..., 4], 1e-4)
        mark = lmf - lo[..., 0]                 # the marks' darkness, put back at a mid scale
        sm = rf_filter(np.dstack([mark * mo, mo]), cost, sk * P["mf_k"])
        lo[..., 0] = lmf - sm[..., 0] / np.maximum(sm[..., 1], 1e-4)
        sr = rf_filter(np.dstack([labr * mr[..., None], mr]), cost, sk)
        lr = sr[..., :3] / np.maximum(sr[..., 3:], 1e-4)
        c = np.clip(np.minimum(so[..., 3], sr[..., 3]) / P["c_min"], 0, 1) * mats_f[k]
        dk = (lr - lo) * c[..., None]
        if k == "light":                        # reference colour on the cups only (straps / knot / leaves keep theirs)
            dk[..., 1:] *= cupsoft[..., None]
        D += dk
        Wc += c
    gw = cv2.GaussianBlur(Wc, (0, 0), 1.5)      # line / partial pixels: the delta of the surface right beside them
    Dn = (np.dstack([cv2.GaussianBlur(D[..., i], (0, 0), 1.5) for i in range(3)])
          / np.maximum(gw, 1e-3)[..., None] * (gw > 0.05)[..., None])
    D = D + np.clip(1 - Wc, 0, 1)[..., None] * Dn
    wl_e = np.maximum(wl_px, wl_raw)            # ink share of a pixel takes no light
    prot = np.maximum(wl_e, ink_share(Lf, wl_e > 0.5))
    new = base.copy()
    new[..., 0] += D[..., 0] * (1 - prot)
    new[..., 1:] += D[..., 1:] * (1 - prot)[..., None]

    # --- level term: the brightness the drawn lines can not take, on the chest region (not the arm / under-bust)
    zone = np.zeros_like(a)
    zone[zy0 - y0: zy1 - y0, zx0 - x0: zx1 - x0] = True
    if P["lvl_sig"]:
        G = lambda x: cv2.GaussianBlur(x, (0, 0), P["lvl_sig"] * Hf)
        ae = cv2.erode(a.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        sc = cv2.morphologyEx(((mats_f["skin"] > 0.5) | olext).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        cy, cx = np.nonzero(olext)
        pts = np.concatenate([np.stack([cx, cy], 1), np.stack([cx, cy - int(P["up"] * Hf)], 1)]).astype(np.int32)
        hull = np.zeros(a.shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
        app = a & sc & zone & (hull > 0)
        af = (ae & app).astype(np.float32)
        # the reference's dark cloth never enters, but lines / dark cloth drawn in BOTH images count on both sides
        # (a reference that kept the line art must not read as "our lines are darker" and lift the chest)
        ar = af * (r[..., 3] > 0.5) * ((mats_r["dark"] < 0.5) | (wl_e > 0.5) | (mats_f["dark"] > 0.5))
        gar, gaf = G(ar), G(af)
        resid = G(labr[..., 0] * ar) / np.maximum(gar, 1e-3) - G(new[..., 0] * af) / np.maximum(gaf, 1e-3)
        resid *= np.clip(gar / np.maximum(gaf, 1e-3) / 0.7, 0, 1)
        comp = np.clip(resid * gaf / np.maximum(G(af * (1 - prot)), 1e-3), -P["lvl_max"], P["lvl_max"])
        comp *= (np.clip(cv2.GaussianBlur(app.astype(np.float32), (0, 0), P["lvl_soft"] * Hf) * 2, 0, 1)
                 * np.clip(gaf / np.maximum(G(a.astype(np.float32)), 1e-3) / 0.3, 0, 1))
        new[..., 0] += comp * (1 - prot)

    # --- guards
    dl = new[..., 0] - labo[..., 0]             # mixed skin / dark-cloth pixels: no borrowed skin light
    new[..., 0] = labo[..., 0] + np.minimum(dl, 0) + np.maximum(dl, 0) * np.clip(1 - mats_f["dark"], 0, 1)
    new[..., 0] = labo[..., 0] + _edge_fill((new[..., 0] - labo[..., 0]).astype(np.float32), wl_e, prot, a, P)
    arf = (r[..., 3] > 0.5).astype(np.float32)  # added light never above the reference's local level + hot_m
    hs = P["hot_sig"] * Hf
    lpr = cv2.GaussianBlur(labr[..., 0] * arf, (0, 0), hs) / np.maximum(cv2.GaussianBlur(arf, (0, 0), hs), 1e-3)
    lim = np.maximum(np.maximum(base[..., 0], labo[..., 0]), lpr + P["hot_m"])
    new[..., 0] = np.where(new[..., 0] > base[..., 0], np.minimum(new[..., 0], lim), new[..., 0])
    kn, lm = P["knee"], P["Lmax"]               # highlight shoulder: nothing is pushed into white
    new[..., 0] = np.where(new[..., 0] > kn, kn + (lm - kn) * np.tanh((new[..., 0] - kn) / (lm - kn)), new[..., 0])
    dl = new[..., 0] - labo[..., 0]             # line art never lighter (print softening included)
    guard = np.maximum(np.maximum(wl_px, (Lf <= P["line_L"])), wl_tl)
    guard = np.maximum(guard, np.clip((line_weight(Lf, P["line_L"]) - 0.35) / 0.1, 0, 1) * np.clip((P["guard_L"] - Lf) / 4, 0, 1))
    new[..., 0] = labo[..., 0] + np.minimum(dl, 0) + np.maximum(dl, 0) * (1 - guard)

    zw = cv2.GaussianBlur(zone.astype(np.float32), (0, 0), P["feather"] * Hf) * a
    if conf is not None:
        zw = zw * np.clip(conf[sl], 0, 1)
    new = labo + (new - labo) * zw[..., None]
    new[..., 0] = np.clip(new[..., 0], 0, 100)
    res = out.copy()
    res[sl][..., :3] = np.where(a[..., None], from_lab(new), o[..., :3])
    return res.astype(np.float32), {"chest": "ok", "chest_box": [int(zy0), int(zy1), int(zx0), int(zx1)]}


def _edge_fill(dl, wl_e, prot, a, P):
    """Pixels within 1 px of line art (anti-aliased edges, thin bands between double lines) take only the light of the
    open surface beside them, times their surface share; narrow cloth with no open surface in reach keeps its own
    positive light (never darkened, so drawn light strips between double lines stay)."""
    sq = lambda r_: np.ones((2 * r_ + 1, 2 * r_ + 1), np.uint8)
    q = np.clip(2 * cv2.dilate(wl_e, sq(1)), 0, 1) * a
    opn = (1 - np.clip(2 * cv2.dilate(wl_e, sq(2)), 0, 1)) * a
    gs = lambda x: cv2.GaussianBlur(x, (0, 0), P["edge_fs"])
    go = gs(opn)
    fillv = gs(dl * opn) / np.maximum(go, 1e-3) * np.clip(go / 0.1, 0, 1)
    ownw = np.clip(1 - prot / P["own_prot"], 0, 1)
    fillv = fillv + (1 - np.clip(go / P["own_sup"], 0, 1)) * np.maximum(dl, 0) * ownw
    return (1 - q) * dl + q * fillv * (1 - prot)
