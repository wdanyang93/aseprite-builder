"""Scene light for 2D character frames.

Default model ("wash", per frame, no reference needed):
  - colour base per material (colour.py)
  - a gentle linear brightness slope across the figure (direction and strength fitted)
  - thin rim lights on the silhouette edge from a key side and a back side (notch-aware edge band)
  - the drawing's own painted shading is kept as is (never subtracted) -> the drawn form never changes
  - orange translucent glow on thin fur / hair edges

Legacy model ("normals", cfg["model"] = "normals"), kept for comparison:
  1. Height field = sqrt of the Poisson inflation of the WHOLE-BODY silhouette (never per part, so garment and
     part boundaries can't create dents or dark seams).
  2. (optional, off by default: geo["bh"] > 0) breast ellipsoids detected from the bright, low-chroma cloth
     blobs in the upper-torso band. Off because they changed the drawn chest shape; the drawing's own painted
     shading is kept instead and the scene light is added on top.
  3. Normals from the height field -> shading S = key light (wrapped Lambert) + back/rim light + linear terms
     + a small share of the drawing's own painted shading. S is a smooth L* offset added on top of the colour base.
  4. Thin fur / hair edges get an orange translucent glow (back light through fur).

The light parameters are fitted ONCE from a (neutral, hand-lit reference) pair of the same pose and stored
as ~20 numbers in the scene profile.
"""
import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from .color import apply_color, from_lab, lowpass, materials, to_lab

LOW_H = 320  # figure height (px) of the low-res geometry grid
# bh = breast-ellipsoid height. Default 0 (off): ellipsoids estimated from the top garment never sit exactly on
# the drawn breasts, and the light they add (a bright spot above / a dark patch on a breast) reads as a CHANGED
# chest shape. With bh = 0 the drawn chest form is kept (chest-centre form correlation 0.986 vs 0.866) and the
# reference match drops only a little (chest skin 0.93 -> 0.90 on the Lyn pair).
GEO_DEFAULT = {"zscale": 1.0, "bh": 0.0, "up": 0.7, "kx": 1.5, "ky": 1.8, "bp": 3.0, "union": "smooth"}
# keep_drawn: the fit may add to, but never subtract, the drawing's own painted shading.
# model "wash": linear brightness slope across the figure + thin rim lights on the silhouette edge; the interior
# form is the drawing's own shading. The older "normals" model (Lambert on an inflated silhouette) put a
# light/dark line through the chest where the torso+arm silhouette turns, which read as a bulge.
CFG_DEFAULT = {"model": "wash", "p2": 2.0, "lin": True, "bumplin": True, "deshade": True, "ds_split": True,
               "glow": True, "pg": 3.0, "chest_w": 3.0, "mean_w": 0.5, "keep_drawn": True}
VIEWS = ("front", "front_left", "front_right", "left", "right", "back", "back_left", "back_right", "walk34")


# ------------------------------------------------------------------ geometry
def _poisson(mask):
    """Lap(u) = -2 inside mask, u = 0 outside (a strip of width w gives x(w-x))."""
    H, W = mask.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        j = np.full(n, -1)
        j[ok] = idx[yy[ok], xx[ok]]
        good = j >= 0
        rows.append(np.arange(n)[good]); cols.append(j[good]); vals.append(np.full(good.sum(), -1.0))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    u = spl.spsolve(A.tocsc(), np.full(n, 2.0))
    out = np.zeros((H, W), np.float64)
    out[ys, xs] = u
    return out


def _n_breasts(view):
    if view is None:
        return 2
    if view.startswith("back"):
        return 0
    if view in ("left", "right"):
        return 1
    return 2


def detect_breasts(rgb_l, m, yrange, view, geo):
    """Bright low-chroma cloth blobs in the upper-torso band -> breast ellipses (cx, cy, rx, ry) in low-res px."""
    nb = _n_breasts(view)
    if nb == 0:
        return []
    y0, y1 = yrange
    Hf = y1 - y0
    lab = to_lab(rgb_l)
    C = np.hypot(lab[..., 1], lab[..., 2])
    cl = (C < 7.5) & (lab[..., 0] > 75) & m
    yy = np.arange(cl.shape[0])[:, None]
    cl &= ((yy - y0) / Hf > 0.2) & ((yy - y0) / Hf < 0.42)
    k = max(3, int(round(0.015 * Hf)) | 1)
    cl = cv2.morphologyEx(cl.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cl = cv2.morphologyEx(cl, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_cc, stats, cents = cv2.connectedComponentsWithStats(cl, connectivity=8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = int(np.argmax(areas)) + 1
    if areas[big - 1] < (0.02 * Hf) ** 2:
        return []
    keep = [i for i in range(1, n) if areas[i - 1] > 0.3 * areas[big - 1] and abs(cents[i][1] - cents[big][1]) < 0.05 * Hf]
    sel = np.isin(lab_cc, keep)
    ys, xs = np.nonzero(sel)
    if nb == 2:
        c = np.array([np.percentile(xs, 25), np.percentile(xs, 75)], np.float64)
        a = np.zeros(len(xs), int)
        for _ in range(20):
            a = np.abs(xs[:, None] - c[None]).argmin(1)
            c = np.array([xs[a == j].mean() if (a == j).any() else c[j] for j in range(2)])
        groups = [a == 0, a == 1]
    else:
        groups = [np.ones(len(xs), bool)]
    out = []
    for gsel in groups:
        if gsel.sum() < 20:
            continue
        gx, gy = xs[gsel], ys[gsel]
        sx, sy = gx.std() * 2.0, gy.std() * 2.0
        cx, cy = gx.mean(), gy.mean() - geo["up"] * sy
        out.append((float(cx), float(cy), float(geo["kx"] * sx), float(geo["ky"] * sy)))
    return out


def geometry(rgba, view=None, geo=None):
    """Low-res normals of the frame. All arrays at LOW_H figure height."""
    geo = geo or GEO_DEFAULT
    alpha = rgba[..., 3]
    ys, xs = np.nonzero(alpha > 0.5)
    y0, y1 = ys.min(), ys.max()
    Hf = float(y1 - y0 + 1)
    s = LOW_H / Hf
    H, W = alpha.shape
    Wl, Hl = int(round(W * s)) + 1, int(round(H * s)) + 1
    al = cv2.resize(alpha, (Wl, Hl), interpolation=cv2.INTER_AREA)
    m = cv2.morphologyEx((al > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    h = np.sqrt(np.maximum(_poisson(m), 0)) * geo["zscale"]
    rgb_l = cv2.resize(rgba[..., :3], (Wl, Hl), interpolation=cv2.INTER_AREA)
    breasts = detect_breasts(rgb_l, m, (y0 * s, y1 * s), view, geo)
    bump = np.zeros_like(h)
    yy, xx = np.mgrid[0:Hl, 0:Wl].astype(np.float32)
    domes = []
    for (cx, cy, rx, ry) in breasts:
        r2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        domes.append(geo["bh"] * 0.5 * (rx + ry) * np.clip(1 - r2, 0, 1) ** geo.get("bp", 1.5))
    if len(domes) == 1:
        bump = domes[0]
    elif len(domes) == 2:
        if geo.get("union", "max") == "smooth":
            # soft maximum: no sharp V-crease where the two domes meet (it read as a changed cleavage)
            k = max(1e-3, geo.get("union_k", 0.25) * float(max(d.max() for d in domes)))
            hi = np.maximum(domes[0], domes[1])
            bump = hi + k * np.log(np.exp((domes[0] - hi) / k) + np.exp((domes[1] - hi) / k)) - k * np.log(2.0) * (
                np.minimum(domes[0], domes[1]) > 0)
            bump = np.maximum(bump, 0)
        else:
            bump = np.maximum(domes[0], domes[1])

    def normals(hh):
        hh = cv2.GaussianBlur(hh.astype(np.float32), (0, 0), geo.get("hsig", 1.0))
        gy, gx = np.gradient(hh)
        nz = 1.0 / np.sqrt(1 + gx ** 2 + gy ** 2)
        return -gx * nz, -gy * nz, nz  # image y down: ny > 0 faces down

    nx0, ny0, nz0 = normals(h)
    h = h + bump * m
    nx, ny, nz = normals(h)
    # wash + rim features: position inside the figure's own box, and an edge band with the outward
    # silhouette direction. Nothing here can put a shape INSIDE the body.
    # edge band from a blurred silhouette: 1 on a straight edge, fading inward over ~2*rim_w, and automatically
    # weaker in concave notches (neck/shoulder, armpit) where the blur stays high - a distance-based band leaked
    # from the neck notch into the upper chest and looked like a bulge
    bl = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, geo.get("rim_w", 0.012) * LOW_H))
    edge = np.clip((1 - bl) / 0.5, 0, 1) ** 1.5 * m
    ab = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.012 * LOW_H))
    gyy, gxx = np.gradient(ab)
    gn = np.hypot(gxx, gyy) + 1e-6
    ys_, xs_ = np.nonzero(m)
    u = (xx - (xs_.min() + xs_.max()) / 2) / LOW_H
    v = (yy - (ys_.min() + ys_.max()) / 2) / LOW_H
    return {"nx": nx, "ny": ny, "nz": nz, "nx0": nx0, "ny0": ny0, "nz0": nz0, "h": h, "m": m,
            "u": u, "v": v, "edge": edge, "nxs": -gxx / gn, "nys": -gyy / gn,
            "breasts": breasts, "bump": bump, "s": s, "Hf": Hf, "shape": (H, W)}


def upsample(x, g):
    """Low-res field -> full frame; outside the silhouette filled by nearest inside value (no dark halo)."""
    H, W = g["shape"]
    m = g["m"]
    xf = x.astype(np.float32).copy()
    if (~m).any():
        _, lab = cv2.distanceTransformWithLabels((~m).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(m)
        lut = np.zeros(lab.max() + 1, np.float32)
        lut[lab[ys, xs]] = xf[ys, xs]
        xf = lut[lab]
    return cv2.resize(xf, (W, H), interpolation=cv2.INTER_CUBIC)


def own_shading(frame, base, mats, Hf, cfg):
    """The drawing's own mid-frequency painted shading (L* units), split into achromatic white cloth and
    everything chromatic (skin, hair, fur). Dark cloth and line art are excluded."""
    L = to_lab(base[..., :3])[..., 0]
    lab_in = to_lab(frame[..., :3])
    Cin = np.hypot(lab_in[..., 1], lab_in[..., 2])
    a = (frame[..., 3] > 0.5) & (lab_in[..., 0] > 50) & ~mats["line"]
    t = np.clip((Cin - 6.0) / 3.0, 0, 1)
    w_white = (1 - t) * (lab_in[..., 0] > 70)
    s1, s2 = cfg.get("ds1", 0.005) * Hf, cfg.get("ds2", 0.04) * Hf
    outs = []
    for w in (w_white, 1 - w_white):
        reg = a & (w > 0.5)
        o = np.zeros_like(L)
        if reg.sum() >= 50:
            o = w * (lowpass(L, reg, s1) - lowpass(L, reg, s2)) * a
        outs.append(o)
    return outs if cfg.get("ds_split") else [outs[0] + outs[1]]


def fur_weight(frame):
    """Low-frequency weight of bright, weakly saturated pixels (fur / hair), excluding skin and white cloth."""
    lab = to_lab(frame[..., :3])
    C = np.hypot(lab[..., 1], lab[..., 2])
    t = np.clip((14.0 - C) / 3.0, 0, 1) * np.clip((C - 4.5) / 2.0, 0, 1) * np.clip((lab[..., 0] - 55.0) / 10.0, 0, 1)
    t = t * t * (3 - 2 * t)
    a = (frame[..., 3] > 0.5).astype(np.float32)
    Hf = float(np.ptp(np.nonzero(a)[0]) + 1)
    return lowpass(t.astype(np.float32), a > 0, 0.006 * Hf) * a


def glow_field(g, p):
    return upsample((1 - g["nz"]) ** p, g)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(np.linalg.norm(v), 1e-9)


def light_model(th, N, N0, cfg, Ns=None):
    """th = [k1, l1x, l1y, l1z, wrap1, k2, l2x, l2y, l2z, cz, lx, ly, bx, by, bz, ds_white, ds_chroma]
    key light (wrapped Lambert) + back/rim light (squared) + flat-normal term + linear normal terms
    + breast-bump terms + share of own painted shading."""
    nx, ny, nz = N
    k1, a1, b1, c1, w1, k2, a2, b2, c2, cz = th[:10]
    l1, l2 = _unit([a1, b1, c1]), _unit([a2, b2, c2])
    w1 = abs(w1)
    tau = cfg.get("soft", 0.0)
    if tau > 0:
        # smooth clamp: a hard max(.,0) leaves a sharp light/dark line (terminator) across the body that
        # reads as a bump or dent in the drawing
        clamp = lambda x: tau * np.logaddexp(0.0, x / tau)
    else:
        clamp = lambda x: np.maximum(x, 0)
    S = k1 * clamp((nx * l1[0] + ny * l1[1] + nz * l1[2] + w1) / (1 + w1))
    S = S + k2 * clamp(nx * l2[0] + ny * l2[1] + nz * l2[2]) ** cfg.get("p2", 2.0)
    S = S + cz * N0[2]
    if cfg.get("lin"):
        S = S + th[10] * nx + th[11] * ny
    if cfg.get("bumplin"):
        S = S + th[12] * (nx - N0[0]) + th[13] * (ny - N0[1]) + th[14] * (nz - N0[2])
    if cfg.get("deshade") and Ns is not None:
        for i, x in enumerate(Ns):
            S = S + th[15 + i] * x
    return S


def _softclamp(x, tau=0.15):
    return tau * np.logaddexp(0.0, x / tau)


WASH_KEYS = ("u", "v", "edge", "nxs", "nys")


def wash_model(th, F, cfg, Ns=None):
    """th = [gx, gy, c, k1, a1, k2, a2, ds_white, ds_chroma]
    wash: a gentle linear brightness slope across the figure (gx, gy in L* per figure height)
    rims: k * edge-band * softclamp(silhouette-normal . light-direction)^2 for a key-side and a back-side light.
    The interior form comes only from the drawing's own painted shading (never subtracted)."""
    gx, gy, c, k1, a1, k2, a2 = th[:7]
    S = gx * F["u"] + gy * F["v"] + c
    for k, a in ((k1, a1), (k2, a2)):
        S = S + k * F["edge"] * _softclamp(np.cos(a) * F["nxs"] + np.sin(a) * F["nys"]) ** 2
    if cfg.get("deshade") and Ns is not None:
        for i, x in enumerate(Ns):
            S = S + th[7 + i] * x
    return S


# ------------------------------------------------------------------ fit
def fit_light(neutral, pairs, color_prof, view="walk34", geo=None, cfg=None):
    """Fit the light parameters so the relit neutral's low-pass L* follows the reference inside every
    fitting region (shape) and across regions (weighted), using label-free regions from `pairs`."""
    from scipy.optimize import least_squares
    geo, cfg = dict(geo or GEO_DEFAULT), dict(cfg or CFG_DEFAULT)
    prof = {"color": color_prof, "geo": geo, "cfg": cfg}
    base = apply_color(neutral, color_prof)
    base_w = pairs.warp(base)
    g = geometry(neutral, view, geo)
    wash = cfg.get("model") == "wash"
    keys = WASH_KEYS if wash else ("nx", "ny", "nz", "nx0", "ny0", "nz0")
    maps = {k: pairs.warp(upsample(g[k], g)) for k in keys}
    maps["Ns"] = [pairs.warp(x) for x in own_shading(neutral, base, materials(neutral), g["Hf"], cfg)]
    if g["breasts"]:
        bump_w = pairs.warp(upsample((g["bump"] > 1e-3).astype(np.float32), g)) > 0.5
    else:  # no ellipsoids: emphasise the upper-torso band instead (where the chest is)
        yy = np.arange(pairs.shape[0])[:, None]
        ys_ = np.nonzero(pairs.ref[..., 3].max(1) > 0.5)[0]
        t = (yy - ys_.min()) / max(ys_.max() - ys_.min(), 1)
        bump_w = np.broadcast_to((t > 0.17) & (t < 0.36), pairs.shape)
    Lc = to_lab(base_w[..., :3])[..., 0]
    Lr = to_lab(pairs.ref[..., :3])[..., 0]
    sig = max(2.0, 0.006 * pairs.Hf)
    rng = np.random.default_rng(0)
    ids, tg, wt, ysl, xsl = [], [], [], [], []
    for ri, R in enumerate(pairs.regions):
        d = lowpass(Lr, R["region"], sig) - lowpass(Lc, R["region"], sig)
        ys, xs = np.nonzero(R["mask"])
        sel = rng.choice(len(ys), min(len(ys), 1500), replace=False)
        ys, xs = ys[sel], xs[sel]
        w = R["w"] * (cfg["chest_w"] if (bump_w[ys, xs]).mean() > 0.3 else 1.0)
        ids.append(np.full(len(ys), ri)); tg.append(d[ys, xs]); wt.append(np.full(len(ys), w / np.sqrt(len(ys))))
        ysl.append(ys); xsl.append(xs)
    ids, tg, wt = np.concatenate(ids), np.concatenate(tg), np.concatenate(wt)
    ys, xs = np.concatenate(ysl), np.concatenate(xsl)
    Ns = [x[ys, xs] for x in maps["Ns"]]
    if wash:
        F = {k: maps[k][ys, xs] for k in WASH_KEYS}
        model = lambda th: wash_model(th, F, cfg, Ns)
    else:
        N = tuple(maps[k][ys, xs] for k in ("nx", "ny", "nz"))
        N0 = tuple(maps[k][ys, xs] for k in ("nx0", "ny0", "nz0"))
        model = lambda th: light_model(th, N, N0, cfg, Ns)
    cnt = np.bincount(ids)
    demean = lambda v: v - (np.bincount(ids, v) / cnt)[ids]
    tgd = demean(tg)
    rw = np.bincount(ids, wt) / cnt * np.sqrt(cnt)
    tgm = np.bincount(ids, tg) / cnt

    def res(th):
        Sm = model(th)
        r1 = (demean(Sm) - tgd) * wt
        e = np.bincount(ids, Sm) / cnt - tgm
        e = e - (e * rw).sum() / rw.sum()
        return np.concatenate([r1, cfg["mean_w"] * e * rw])

    if wash:
        best = None
        # right key rim + left back rim, and the mirrored start
        for x0 in ([6, -3, 0, 20, 0.0, 20, np.pi, 0.01, 0.01], [-6, -3, 0, 20, np.pi, 20, 0.0, 0.01, 0.01]):
            r = least_squares(res, x0, bounds=([-60, -60, -60, 0, -10, 0, -10, 0, 0], [60, 60, 60, 80, 10, 80, 10, 3, 3]),
                              loss="soft_l1", f_scale=3.0 * wt.mean())
            if best is None or r.cost < best.cost:
                best = r
        prof["light"] = [float(v) for v in best.x]
        return _finish_fit(prof, neutral, pairs, g, cfg, view, Lr)
    nb = 17
    lo = [0, -1, -1, -1, 0, 0, -1, -1, -1] + [-np.inf] * (nb - 9)
    hi = [80, 1, 1, 1, 1.0, 80, 1, 1, 1] + [np.inf] * (nb - 9)
    if cfg.get("keep_drawn"):
        # never subtract the artist's own painted shading (cleavage, under-breast shadow): it defines the drawn form
        lo[15] = lo[16] = 0.0
    lo[4] = cfg.get("min_wrap", 0.0)  # wrapped key light: fades gradually instead of cutting off
    best = None
    for x0 in ([20, 0.6, -0.6, 0.5, 0.3, 10, -0.7, 0.3, -0.6, -5],   # front-right key, left-back rim
               [20, 0.6, -0.6, -0.5, 0.2, 10, -0.7, 0.5, -0.5, 0],
               [15, 0.8, -0.3, 0.2, 0.5, 10, -0.5, -0.5, -0.7, -8],
               [25, 0.5, -0.5, -0.7, 0.1, 5, 0.3, 0.7, -0.6, -4]):
        x0 = list(x0) + [0.0] * (nb - 10)
        x0[4] = max(x0[4], lo[4] + 0.05) if lo[4] < 1 else 1.0
        if cfg.get("keep_drawn"):
            x0[15] = x0[16] = 0.01
        r = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0 * wt.mean())
        if best is None or r.cost < best.cost:
            best = r
    prof["light"] = [float(v) for v in best.x]
    return _finish_fit(prof, neutral, pairs, g, cfg, view, Lr)


def _finish_fit(prof, neutral, pairs, g, cfg, view, Lr):
    """Fur/hair edge glow and the global brightness offset (shared by both light models)."""
    prof["glow"], prof["dL"] = [0.0, 0.0, 0.0], 0.0
    if cfg.get("glow"):
        ow = pairs.warp(relight(neutral, prof, view))
        lo_, lr_ = to_lab(ow[..., :3]), to_lab(pairs.ref[..., :3])
        Gw = pairs.warp(glow_field(g, cfg["pg"]) * fur_weight(neutral))
        cols, ds = [], [[], [], []]
        regs = [R for R in pairs.regions]
        for i, R in enumerate(regs):
            reg = R["region"] & (ow[..., 3] > 0.5) & (pairs.mat["dark"] < 0.3)
            if reg.sum() < 50:
                continue
            cols.append((i, reg))
        X = np.zeros((sum(r.sum() for _, r in cols), 1 + len(cols)))
        row = 0
        for j, (i, reg) in enumerate(cols):
            n = reg.sum()
            X[row:row + n, 0] = Gw[reg]
            X[row:row + n, 1 + j] = 1.0
            for c in range(3):
                ds[c].append(lowpass(lr_[..., c] - lo_[..., c], reg, 2.0)[reg])
            row += n
        prof["glow"] = [float(np.linalg.lstsq(X, np.concatenate(ds[c]), rcond=None)[0][0]) for c in range(3)]

    ow = pairs.warp(relight(neutral, prof, view))
    A = (ow[..., 3] > 0.9) & pairs.reliable
    prof["dL"] = float(Lr[A].mean() - to_lab(ow[..., :3])[..., 0][A].mean())
    return prof


# ------------------------------------------------------------------ apply
def relight(frame, prof, view=None):
    """Relight one frame (float RGBA). Uses only the frame and the profile."""
    mats = materials(frame)
    base = apply_color(frame, prof["color"], mats)
    cfg, th = prof["cfg"], prof["light"]
    g = geometry(frame, view, prof["geo"])
    wash = cfg.get("model") == "wash"
    if wash:
        S = wash_model(np.array(th), g, cfg)
    else:
        S = light_model(np.array(th), (g["nx"], g["ny"], g["nz"]), (g["nx0"], g["ny0"], g["nz0"]), cfg)
    S = upsample(S - S[g["m"]].mean() + prof.get("dL", 0.0), g)
    if cfg.get("deshade"):
        i0 = 7 if wash else 15
        for i, x in enumerate(own_shading(frame, base, mats, g["Hf"], cfg)):
            S = S + th[i0 + i] * x
    lab = to_lab(base[..., :3])
    lab[..., 0] = lab[..., 0] + S
    gl = prof.get("glow", [0, 0, 0])
    if any(gl):
        G = glow_field(g, cfg["pg"]) * fur_weight(frame)
        for c in range(3):
            lab[..., c] += gl[c] * G
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    return np.dstack([from_lab(lab), frame[..., 3]]).astype(np.float32)
