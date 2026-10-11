"""Colour regions and texture regions of a drawing (design 3): materials WITHOUT fixed class names.

Light hardly depends on the material (material explains 9.5 % / 8.7 % of the light ratio), so regions are only used as
filter walls, as the scope a light ratio may be carried over in, and for per-region checks. They come from the
character's ORIGINAL drawing, never from a reference:
1. palette: Lab k-means (L* weighted down, so a painted shading band does not become its own material) on the figure's
   pixels that are neither line art nor high-gradient (anti-aliased edges, painted transitions); k = 4-12, chosen by
   the BIC of the k-means mixture (spherical, shared variance) on a fixed-size sample (on painted art the BIC keeps
   rising, so it is an upper bound); centres closer than merge_dE are then merged, never below k_min colours;
2. connected regions: watershed of the gradient magnitude with the line map as walls, seeded by the connected pieces
   of each palette colour -> regions meet at the drawn lines;
3. texture regions (fur / hair): high-frequency L* energy AND structure-tensor coherence (oriented strands), measured
   with the line art filled in by its surface; not a colour band (the old fur colour band was Lyn-only);
4. frames: every pixel assigned to the nearest card palette colour (-1 when farther than max_dE: a colour the card
   has never seen -> warn), and the same line-walled watershed for the frame's own connected regions.
All lengths are fractions of the figure height H. Deterministic (fixed seeds).
"""
import numpy as np
import cv2
from scipy import ndimage as ndi

from .color import to_lab
from .geom import figure_height, crop_box

REGIONS = {
    "w_L": 0.4,           # L* weight in the clustering space (a*, b* weight 1)
    "k_min": 4, "k_max": 12,
    "n_sample": 3000,     # BIC sample size (fixed, so the penalty means the same for every drawing)
    "n_fit": 40000,       # pixels used to fit the centres
    "iters": 30,
    "seed": 0,
    "grad_q": 0.6,        # keep the pixels below this quantile of the Lab gradient for the palette
    "grad_sig": 0.001,    # gradient measured on Lab blurred by this (fraction of H, at least 0.5 px)
    "line_w": 0.3,        # line weight above this = line art (left out of the palette, walls for the watershed)
    "merge_dE": 8.0,      # centres closer than this (clustering space) are one colour
    "min_seed": 0.0004,   # seed pieces smaller than this share of the figure area are flooded, not seeds
    "max_dE": 20.0,       # frame pixel farther than this from every card colour -> -1 (new colour)
    "tex_hp": 0.004,      # texture: high-pass scale ...
    "tex_win": 0.012,     # ... energy / structure tensor window
    "tex_E": (2.5, 4.0),  # high-pass L* std ramp
    "tex_coh": (0.15, 0.3),   # coherence ramp (of the high-passed L*)
    "tex_smooth": 0.01,
    "tex_line_w": 0.05,   # line pixels (and 1 px around them) above this weight are filled in and left out
    "tex_edge": 0.006,    # band along the silhouette left out of the texture measures
    "tex_min": 0.002,     # texture pieces smaller than this share of the figure area are dropped
}


def _feat(lab, p):
    f = lab.reshape(-1, 3).astype(np.float64).copy()
    f[:, 0] *= p["w_L"]
    return f


def _unfeat(c, p):
    c = np.array(c, np.float64).copy()
    c[:, 0] /= p["w_L"]
    return c


# ------------------------------------------------------------------ k-means + BIC
def kmeans(X, k, seed=0, iters=30):
    """Deterministic k-means (k-means++ init with a fixed seed). Returns centres (k, d), labels, inertia."""
    rng = np.random.default_rng(seed)
    n = len(X)
    C = [X[rng.integers(n)]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        pr = d2 / max(d2.sum(), 1e-12)
        C.append(X[rng.choice(n, p=pr)])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.array(C)
    lab = np.zeros(n, np.int64)
    for _ in range(iters):
        D = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = D.argmin(1)
        if _ > 0 and (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    D = ((X[:, None, :] - C[None]) ** 2).sum(-1)
    lab = D.argmin(1)
    return C, lab, float(D[np.arange(n), lab].sum())


def kmeans_bic(X, C):
    """BIC of the hard k-means mixture with one shared spherical variance (Pelleg & Moore 2000); higher is better."""
    n, d = X.shape
    k = len(C)
    D = ((X[:, None, :] - C[None]) ** 2).sum(-1)
    lab = D.argmin(1)
    nk = np.bincount(lab, minlength=k).astype(np.float64)
    var = D[np.arange(n), lab].sum() / max(d * (n - k), 1)
    var = max(var, 1e-6)
    ok = nk > 0
    ll = (nk[ok] * np.log(nk[ok] / n)).sum() - n * d / 2 * np.log(2 * np.pi * var) - d * (n - k) / 2
    return float(ll - (k * (d + 1)) / 2 * np.log(n))


def _merge(C, counts, dE, k_min=1):
    C, counts = [c for c in C], list(counts)
    while len(C) > max(1, k_min):
        A = np.array(C)
        D = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1)) + np.eye(len(C)) * 1e9
        i, j = np.unravel_index(D.argmin(), D.shape)
        if D[i, j] >= dE:
            break
        w = counts[i] + counts[j]
        C[i] = (C[i] * counts[i] + C[j] * counts[j]) / max(w, 1)
        counts[i] = w
        del C[j], counts[j]
    return np.array(C), np.array(counts)


# ------------------------------------------------------------------ per-drawing measurements
def lab_gradient(lab, H, p=None):
    """Lab gradient magnitude (Delta E per px) on slightly blurred Lab."""
    p = dict(REGIONS, **(p or {}))
    s = max(0.5, p["grad_sig"] * H)
    lb = cv2.GaussianBlur(lab.astype(np.float32), (0, 0), s)
    g = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(lb[..., c], cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(lb[..., c], cv2.CV_32F, 0, 1, ksize=3) / 8
        g += gx * gx + gy * gy
    return np.sqrt(g)


def palette_pixels(rgba, line_w, H=None, p=None):
    """Lab values of a drawing's flat-colour pixels: in the figure (eroded 1 px), not line art, low gradient."""
    p = dict(REGIONS, **(p or {}))
    H = H or figure_height(rgba[..., 3])
    lab = to_lab(np.clip(rgba[..., :3], 0, 1))
    fig = cv2.erode((rgba[..., 3] > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    nl = line_w <= p["line_w"]
    nl &= cv2.erode(nl.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    g = lab_gradient(lab, H, p)
    cand = fig & nl
    if cand.sum() < 50:
        cand = fig
    thr = np.quantile(g[cand], p["grad_q"]) if cand.any() else 0
    keep = cand & (g <= thr)
    return lab[keep]


def fit_palette(pixels, p=None):
    """Palette (Lab centres, k x 3) of a set of Lab pixels; k chosen by BIC over k_min..k_max, then close centres
    merged. Returns dict(centres, counts, k_bic, bic)."""
    p = dict(REGIONS, **(p or {}))
    X = _feat(np.asarray(pixels, np.float32).reshape(-1, 1, 3), p)
    rng = np.random.default_rng(p["seed"])
    Xs = X[rng.choice(len(X), min(len(X), p["n_sample"]), replace=False)] if len(X) else X
    Xf = X[rng.choice(len(X), min(len(X), p["n_fit"]), replace=False)] if len(X) else X
    if len(Xs) < p["k_min"] * 5:
        c = Xf.mean(0, keepdims=True) if len(Xf) else np.zeros((1, 3))
        return {"centres": _unfeat(c, p), "counts": np.array([len(Xf)]), "k_bic": 1, "bic": {}}
    bic = {}
    best = None
    for k in range(p["k_min"], min(p["k_max"], len(Xs) // 5) + 1):
        C, _, _ = kmeans(Xs, k, p["seed"], p["iters"])
        bic[k] = kmeans_bic(Xs, C)
        if best is None or bic[k] > bic[best]:
            best = k
    C, lab, _ = kmeans(Xf, best, p["seed"], p["iters"])
    counts = np.bincount(lab, minlength=best)
    C, counts = C[counts > 0], counts[counts > 0]
    C, counts = _merge(C, counts, p["merge_dE"], min(p["k_min"], len(C)))
    order = np.argsort(C[:, 0])                     # dark to light: stable, readable palette order
    return {"centres": _unfeat(C[order], p), "counts": counts[order], "k_bic": int(best),
            "bic": {int(k): round(v, 1) for k, v in bic.items()}}


def assign(rgba, centres, p=None):
    """Nearest palette colour per pixel (clustering space). Returns (index HxW int32, -1 outside the figure or farther
    than max_dE from every colour; distance HxW, inf outside the figure)."""
    p = dict(REGIONS, **(p or {}))
    fig = rgba[..., 3] > 0.5
    idx = -np.ones(fig.shape, np.int32)
    dist = np.full(fig.shape, np.inf, np.float32)
    if not fig.any():
        return idx, dist
    X = _feat(to_lab(np.clip(rgba[..., :3], 0, 1))[fig][:, None, :], p).astype(np.float32)
    Cf = _feat(np.asarray(centres, np.float32).reshape(-1, 1, 3), p).astype(np.float32)
    D = np.sqrt(((X[:, None, :] - Cf[None]) ** 2).sum(-1))
    j = D.argmin(1)
    d = D[np.arange(len(j)), j]
    idx[fig] = np.where(d <= p["max_dE"], j, -1)
    dist[fig] = d
    return idx, dist


def segment(rgba, centres, line_w, H=None, p=None):
    """Connected colour regions (see _segment), computed on the figure's bounding box."""
    H = H or figure_height(rgba[..., 3])
    box = crop_box(rgba[..., 3], 4)
    if box is None:
        z = np.zeros(rgba.shape[:2], np.int32)
        return {"labels": z, "colour": np.array([-1], np.int32), "index": z - 1, "n": 0}
    r = _segment(rgba[box], centres, line_w[box], H, p)
    lbl = np.zeros(rgba.shape[:2], np.int32)
    idx = -np.ones(rgba.shape[:2], np.int32)
    lbl[box], idx[box] = r["labels"], r["index"]
    r.update(labels=lbl, index=idx)
    return r


def _segment(rgba, centres, line_w, H=None, p=None):
    """Connected colour regions of a drawing. Seeds = connected pieces of one palette colour among the flat, non-line
    pixels (pieces smaller than min_seed are not seeds); the watershed of the Lab gradient, with line art as walls,
    floods every figure pixel from them. Returns dict(labels HxW int32 (0 outside the figure, regions 1..n),
    colour (palette index per region, index 0 unused = -1), index (nearest palette colour per pixel), n)."""
    p = dict(REGIONS, **(p or {}))
    a = rgba[..., 3] > 0.5
    H = H or figure_height(rgba[..., 3])
    idx, _ = assign(rgba, centres, dict(p, max_dE=1e9))
    lab = to_lab(np.clip(rgba[..., :3], 0, 1))
    g = lab_gradient(lab, H, p)
    flat = a & (line_w <= p["line_w"])
    if flat.any():
        flat &= g <= np.quantile(g[flat], p["grad_q"])
    min_px = max(4, int(p["min_seed"] * a.sum()))
    markers = np.zeros(a.shape, np.int32)
    colour = [-1]
    nxt = 1
    for j in range(len(centres)):
        n, cc, st, _ = cv2.connectedComponentsWithStats((flat & (idx == j)).astype(np.uint8), connectivity=4)
        ok = np.nonzero(st[:, 4] >= min_px)[0]
        ok = ok[ok > 0]
        if not len(ok):
            continue
        lut = np.zeros(n, np.int32)
        lut[ok] = nxt + np.arange(len(ok))
        markers = np.where(lut[cc] > 0, lut[cc], markers)
        colour += [j] * len(ok)
        nxt += len(ok)
    if nxt == 1:                                     # nothing flat enough: one region per palette colour
        markers = np.where(a, idx + 1, 0).astype(np.int32)
        colour = [-1] + list(range(len(centres)))
        nxt = len(centres) + 1
    elev = g / max(float(np.quantile(g[a], 0.99)) if a.any() else 1.0, 1e-3)
    elev = np.clip(elev + 4.0 * np.clip(line_w / max(p["line_w"], 1e-3), 0, 1), 0, 6)
    E = (elev / 6 * 60000).astype(np.uint16)
    E[~a] = 65535
    lbl = ndi.watershed_ift(E, markers)
    lbl = np.where(a, lbl, 0).astype(np.int32)
    lbl[lbl < 0] = 0
    return {"labels": lbl, "colour": np.array(colour, np.int32), "index": np.where(a, idx, -1).astype(np.int32),
            "n": int(nxt - 1)}


def texture(rgba, H=None, line=None, p=None):
    """Fur / hair texture weight and mask (see _texture), computed on the figure's bounding box."""
    p = dict(REGIONS, **(p or {}))
    H = H or figure_height(rgba[..., 3])
    box = crop_box(rgba[..., 3], int(3 * p["tex_win"] * H) + 4)
    t, m = np.zeros(rgba.shape[:2], np.float32), np.zeros(rgba.shape[:2], bool)
    if box is None:
        return t, m
    ln = None if line is None else {"weight": line["weight"][box], "surface": line["surface"][box]}
    t[box], m[box] = _texture(rgba[box], H, ln, p)
    return t, m


def _texture(rgba, H=None, line=None, p=None):
    """Fur / hair texture weight (HxW 0..1): oriented fine detail. High-pass L* energy times structure-tensor coherence
    of the high-passed L*, both measured only on interior surface pixels (line art and its anti-aliased edge, when a
    line map dict is given, and a band along the silhouette are left out, so outlines are not "texture"), smoothed,
    small pieces dropped. Returns (soft weight, mask)."""
    p = dict(REGIONS, **(p or {}))
    a = rgba[..., 3] > 0.5
    H = H or figure_height(rgba[..., 3])
    L = to_lab(np.clip(rgba[..., :3], 0, 1))[..., 0].astype(np.float32)
    valid = a.copy()
    if line is not None:
        ln = cv2.dilate((line["weight"] > p["tex_line_w"]).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        L = np.where(ln, line["surface"], L).astype(np.float32)
        valid &= ~ln
    e = max(1, int(round(p["tex_edge"] * H)))
    valid &= cv2.erode(a.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))) > 0
    af, vf = a.astype(np.float32), valid.astype(np.float32)
    G = lambda x, s: cv2.GaussianBlur(x, (0, 0), max(0.5, s))
    sw = p["tex_win"] * H
    den = np.maximum(G(vf, sw), 1e-3)
    hp = (L - G(L * af, p["tex_hp"] * H) / np.maximum(G(af, p["tex_hp"] * H), 1e-3)) * vf
    E = np.sqrt(np.maximum(G(hp * hp, sw) / den, 0))
    Ls = G(hp, 0.6)                                  # orientation of the fine detail (a smooth shading ramp has none)
    gx = cv2.Sobel(Ls, cv2.CV_32F, 1, 0, ksize=3) / 8 * vf
    gy = cv2.Sobel(Ls, cv2.CV_32F, 0, 1, ksize=3) / 8 * vf
    jxx, jyy, jxy = G(gx * gx, sw), G(gy * gy, sw), G(gx * gy, sw)
    tr = jxx + jyy
    dif = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = (dif / np.maximum(tr, 1e-6)) ** 2
    ramp = lambda x, lo, hi: np.clip((x - lo) / (hi - lo), 0, 1)
    support = np.clip(G(vf, sw) / np.maximum(G(af, sw), 1e-3) / 0.3, 0, 1)    # enough interior surface measured
    t = ramp(E, *p["tex_E"]) * ramp(coh, *p["tex_coh"]) * support * af
    t = G(t, p["tex_smooth"] * H) / np.maximum(G(af, p["tex_smooth"] * H), 1e-3) * af
    m = t > 0.5
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = st[:, 4] >= max(4, p["tex_min"] * a.sum())
    keep[0] = False
    m = keep[cc]
    return np.clip(t, 0, 1).astype(np.float32), m
