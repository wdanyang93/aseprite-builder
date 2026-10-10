"""Chest stage (reference pack only): the reference's light on the breasts at a fine scale, cut by the DRAWN lines.

Why: the pack's interior transfer is soft (SIGMA_IN = 1.5 % of figure height), so the chest kept the drawing's
frontal highlight and read as flat (user: "the inside of the chest does not show volume according to the light").
What it does, per frame, using only that frame's own drawing for geometry:
1. find the two drawn bra cups (bright low-chroma cloth in the upper torso, cut by the line art);
2. a zone around them (upper-breast skin, cleavage, cups, under-bust) - nothing outside it changes;
3. "region selection + brightness": the reference's local light minus the frame's local light, low-passed with a
   recursive edge-aware filter whose barriers are the drawn lines, per material (skin onto skin, cloth onto cloth),
   so a highlight stops exactly at the cleavage line / cup outline instead of spreading as a round glow;
4. the cup print's contrast is lowered around a larger-scale base (tex) so the form light reads, as in the reference;
5. line art and its anti-aliased edge never get lighter; alpha is never touched.
"""
import numpy as np
import cv2

from .color import to_lab, from_lab, materials

CHEST = {
    "sig": 0.004,       # light scale (fraction of figure height); ~4 px on the 1400 px neutral
    "barrier": 40.0,    # step cost of a drawn line pixel for the edge-aware filter
    "tex": 0.4,         # cup print contrast kept (1 = unchanged)
    "tex_sig": 0.008,   # scale the print is softened around
    "line_L": 40.0,     # L* below which a pixel is line art
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


def chest_light(frame, out, ref, conf=None, params=None):
    """frame: neutral-lit RGBA; out: relit RGBA (pack result); ref: reference RGBA warped into this frame;
    conf: HxW 0..1 trust in the warp (None = 1). Returns (rgba, info). Nothing outside the chest zone changes."""
    P = dict(CHEST, **(params or {}))
    cups, Hf, why = find_cups(frame)
    if cups is None:
        return out, {"chest": why}
    a_full = frame[..., 3] > 0.5
    both = cups[0] | cups[1]
    ys, xs = np.nonzero(both)
    H, W = a_full.shape
    zy0, zy1 = max(0, int(ys.min() - P["up"] * Hf)), min(H, int(ys.max() + P["down"] * Hf))
    zx0, zx1 = max(0, int(xs.min() - P["side"] * Hf)), min(W, int(xs.max() + P["side"] * Hf))
    m = int(0.03 * Hf) + 2
    y0, y1, x0, x1 = max(zy0 - m, 0), min(zy1 + m, H), max(zx0 - m, 0), min(zx1 + m, W)
    sl = (slice(y0, y1), slice(x0, x1))
    f, o, r = frame[sl], out[sl], np.clip(ref[sl], 0, 1)
    a = a_full[sl]
    labf, labo, labr = to_lab(np.clip(f[..., :3], 0, 1)), to_lab(np.clip(o[..., :3], 0, 1)), to_lab(r[..., :3])
    wl_px = line_weight(labf[..., 0], P["line_L"])
    line = wl_px > 0.5
    cost = 1 + P["barrier"] * np.clip(cv2.GaussianBlur(line.astype(np.float32), (0, 0), 0.8) * 2, 0, 1) + 200 * (~a)
    mats_f, mats_r = materials(f), materials(r)
    sig = P["sig"] * Hf
    d = np.zeros_like(labo)
    for k in ("skin", "light"):                 # reference skin light onto skin, cloth light onto cloth
        mo = (a & ~line & (mats_f[k] > 0.5)).astype(np.float32)
        mr = mo * ((mats_r["dark"] < 0.5) & (r[..., 3] > 0.5))      # never pull in the reference's dark cloth
        so = rf_filter(np.dstack([labo * mo[..., None], mo]), cost, sig)
        sr = rf_filter(np.dstack([labr * mr[..., None], mr]), cost, sig)
        lo, lr = so[..., :3] / np.maximum(so[..., 3:], 1e-4), sr[..., :3] / np.maximum(sr[..., 3:], 1e-4)
        c = np.clip(np.minimum(so[..., 3], sr[..., 3]) / 0.35, 0, 1) * mats_f[k]
        d += (lr - lo) * c[..., None]

    new = labo.copy()
    new[..., 0] += d[..., 0]
    if P["tex"] < 1:                            # soften the cup print so the light on the form reads
        cupm = (both[sl] & ~line).astype(np.float32)
        wl0 = cv2.dilate(wl_px, np.ones((5, 5), np.uint8))
        cupw = cv2.GaussianBlur(both[sl].astype(np.float32), (0, 0), 1.5) * (1 - wl0) * mats_f["light"]
        sb = P["tex_sig"] * Hf
        lpb = cv2.GaussianBlur(new[..., 0] * cupm, (0, 0), sb) / np.maximum(cv2.GaussianBlur(cupm, (0, 0), sb), 1e-4)
        new[..., 0] += (1 - P["tex"]) * cupw * (lpb - new[..., 0])
    wl = cv2.dilate(wl_px, np.ones((3, 3), np.uint8))
    dl = new[..., 0] - labo[..., 0]
    new[..., 0] = labo[..., 0] + np.minimum(dl, 0) + np.maximum(dl, 0) * (1 - wl)   # lines never lighter
    new[..., 1:] += d[..., 1:]

    zone = np.zeros_like(a)
    zone[zy0 - y0: zy1 - y0, zx0 - x0: zx1 - x0] = True
    zw = cv2.GaussianBlur(zone.astype(np.float32), (0, 0), P["feather"] * Hf) * a
    if conf is not None:
        zw = zw * np.clip(conf[sl], 0, 1)
    new = labo + (new - labo) * zw[..., None]
    new[..., 0] = np.clip(new[..., 0], 0, 100)
    res = out.copy()
    res[sl][..., :3] = np.where(a[..., None], from_lab(new), o[..., :3])
    return res.astype(np.float32), {"chest": "ok", "chest_box": [int(zy0), int(zy1), int(zx0), int(zx1)]}
