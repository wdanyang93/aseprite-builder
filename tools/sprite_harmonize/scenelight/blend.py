"""Scene blend: make a game-scale sprite sit in a scene like a painter's scene-integrated version of the same character,
by colour, tone and outline only (no shadow, nothing redrawn, alpha untouched).

The painter study (v10, ~1600 case-specific lines) distilled to four steps:
1. per Adobe part a smooth monotone L* tone curve (the sprite part's L* quantiles -> the target part's, measured at
   game scale; the painter lifts the darks; the outer 2 game px get their own curve) plus the mean a*/b* shift,
   applied at full resolution; Adobe line art never gets lighter;
2. hair: small dark flecks that are not Adobe line art are pulled toward the local hair median (clean hair);
3. premultiplied-alpha downscale to the game height (area filter: no halo, no colour from transparent pixels);
4. an anti-aliased outer line ~1.3 game px inside the alpha boundary in each part's target outer-line colour
   (painter: nearly black on boots / gloves); never lighter than the input pixel, a darker one keeps its own tone.
Parts come from Adobe (image_select_subject bodyParts / image_select_by_prompt masks), lines are the thin dark paths of
Adobe image_vectorize; part names of sprite and target must match ('hair' gets step 2). Parts should not overlap
(overlaps are averaged; e.g. legs minus shoes), 'subject' (the whole figure) is ignored, figure pixels no part covers
take a whole-figure curve. Fit once on one frame and its target, apply to every frame of the scene:
    look = fit_blend(rgba, parts, painter_rgb, painter_parts, height, lines)     # JSON-able: json.dump(look, f)
    out = apply_blend(other_rgba, other_parts, look, height, other_lines)        # float RGBA at `height` rows
"""
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from .color import figure_height, from_lab, lowpass, to_lab

BLEND = {
    "q": (2, 5, 10, 25, 50, 75, 90, 98),  # L* quantiles matched per part (the knots of its tone curve)
    "min_px": 150,        # a part needs this many game-res pixels in both images for its own curve (else: whole figure)
    "edge": 2.0,          # the outer `edge` game px of a part get their own curve (rim: outline, anti-aliasing and,
                          # in the target, a little background), the rest the interior curve
    "feather": 0.003,     # part weights are blurred by this (fraction of H): neighbouring curves meet without a seam
    "fleck_win": 0.012,   # local hair median window (fraction of H)
    "fleck_dl": 20.0,     # a hair pixel this much darker (L*) than the local median starts to count as a fleck ...
    "fleck_full": 35.0,   # ... and is pulled all the way to the median from this much darker
    "fleck_edge": 0.004,  # not within this distance of the alpha edge (fraction of H): the silhouette outline stays
    "fleck_area": 3e-5,   # a fleck is a dark spot of at most this area (fraction of H^2) not touching Adobe line art;
                          # longer strokes and continuations of Adobe lines are drawing and stay
    "ring_w": 1.3,        # outer line width (game px)
    "ring_probe": 1.0,    # the target's outer line = median Lab of its outermost pixels (depth <= this, game px)
    "fade": 1.5,          # rim and interior curves cross-fade over this many game px around `edge`
    "filter": "area",     # downscale filter: 'area' (no ringing) or 'bicubic' (PIL, antialiased, sharper)
}


def _f32(x):
    x = np.asarray(x)
    return x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)


def _mask(m):
    m = np.asarray(m, np.float32)
    return m / 255.0 if m.max() > 1.0 else m


def _resize(ch, size, filt):
    if filt == "bicubic":
        from PIL import Image
        return np.asarray(Image.fromarray(ch.astype(np.float32), "F").resize(size, Image.BICUBIC), np.float32)
    return cv2.resize(ch.astype(np.float32), size, interpolation=cv2.INTER_AREA)


def downscale(rgba, height, filt="area"):
    """Premultiplied-alpha downscale to `height` rows. The alpha is the plain downscale of the input alpha."""
    rgba = _f32(rgba)
    size = (max(1, round(rgba.shape[1] * height / rgba.shape[0])), int(height))
    a = np.clip(_resize(rgba[..., 3], size, filt), 0, 1)
    pm = np.dstack([_resize(rgba[..., k] * rgba[..., 3], size, filt) for k in range(3)])
    rgb = np.clip(pm / np.maximum(a[..., None], 1e-4), 0, 1) * (a[..., None] > 1e-4)
    return np.dstack([rgb, a]).astype(np.float32)


def _depth(sil):
    """Distance (px) of each pixel centre inside `sil` from the silhouette contour (outside: <= 0)."""
    return cv2.distanceTransform(sil.astype(np.uint8), cv2.DIST_L2, 5) - 0.5


def _stats(lab, sil, m, edge, q, rim=False):
    """L* quantiles and mean a*, b* of a part's interior (depth > edge) or rim (depth <= edge); (quantiles, a, b, n)."""
    d = _depth(sil)
    k = sil & (m >= 0.5) & ((d <= edge) if rim else (d > edge))
    if k.sum() == 0:
        return None, 0.0, 0.0, 0
    v = lab[k]
    return np.percentile(v[:, 0], q), float(v[:, 1].mean()), float(v[:, 2].mean()), int(k.sum())


def _ring(rgb, sil, m, probe, cover):
    """Lab of the part's outer line in the target: its outermost pixels (depth <= probe) with the background they
    are mixed with (median just outside) taken out at the sprite's own edge coverage `cover` (median alpha there),
    so the line composites back to what the target shows, on any background."""
    band = sil & (m >= 0.5) & (_depth(sil) <= probe)
    out = (cv2.dilate(band.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & \
        ~(cv2.dilate(sil.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)     # 3-5 px outside: clear of the edge
    if band.sum() < 20 or out.sum() < 20:
        return None
    c = max(cover, 0.5)
    line = (np.median(rgb[band], 0) - (1 - c) * np.median(rgb[out], 0)) / c
    return np.round(to_lab(np.clip(line, 0, 1)[None, None])[0, 0], 2).tolist()


def _lut(x, y, gap=0.5, slope=3.0):
    """Smooth monotone L* curve through (0,0), (x_i, y_i), (100,100), tabulated on 0..100 step 0.1. Knots closer than
    `gap` L* are merged (flat fills: no step) and the slope is capped at `slope` (no noise amplification)."""
    x = np.maximum.accumulate(np.clip(np.asarray(x, np.float64), 0.5, 99.5))
    y = np.maximum.accumulate(np.clip(np.asarray(y, np.float64), 0.0, 100.0))
    g = np.r_[0, np.cumsum(np.diff(x) >= gap)]
    n = np.bincount(g)
    x, y = np.r_[0.0, np.bincount(g, x) / n, 100.0], np.r_[0.0, np.bincount(g, y) / n, 100.0]
    y = np.clip(y, 100 - slope * (100 - x), slope * x)    # reachable from (0,0) and to (100,100) within the cap
    for i in range(1, len(x)):
        y[i] = min(y[i], y[i - 1] + slope * (x[i] - x[i - 1]))
    return PchipInterpolator(x, y)(np.linspace(0, 100, 1001)).astype(np.float32)


def _weights(parts, names, alpha, H, feather):
    """Feathered part weights, normalised to sum 1; the whole-figure fallback '_all' only gets the figure pixels no
    part covers (so a part's own edge is not mixed with the whole-figure curve or line)."""
    sig = max(0.5, feather * H)
    ms = {k: _mask(parts[k]) for k in names if k in parts}
    union = np.clip(sum(ms.values()), 0, 1) if ms else np.zeros_like(alpha)
    ms["_all"] = (alpha >= 0.5) * (1 - union)
    w = {k: cv2.GaussianBlur(v, (0, 0), sig) for k, v in ms.items()}
    s = sum(w.values())
    w = {k: v / np.maximum(s, 1e-6) for k, v in w.items()}
    w["_all"] = np.where(s > 1e-6, w["_all"], 1.0)
    return w


def _clean_hair(lab, hair, line, alpha, H, P):
    """Pull small hair flecks (much darker than the local hair median, not Adobe line art) toward that median."""
    hm = (hair >= 0.5) & (alpha >= 0.5)
    if hm.sum() < 50:
        return lab
    base = hm & (line < 0.5)
    k = max(3, int(P["fleck_win"] * H) | 1)
    med = []
    for c, off in ((0, 0.0), (1, 128.0), (2, 128.0)):
        ch = np.where(base, lab[..., c], lowpass(lab[..., c], base, k))   # fill outside the hair: no bias from it
        u8 = np.clip((ch + off) * (2.55 if c == 0 else 1.0), 0, 255).astype(np.uint8)
        med.append(cv2.medianBlur(u8, k).astype(np.float32) / (2.55 if c == 0 else 1.0) - off)
    med = np.dstack(med)
    t = np.clip((med[..., 0] - lab[..., 0] - P["fleck_dl"]) / (P["fleck_full"] - P["fleck_dl"]), 0, 1)
    edge = np.clip(_depth(alpha >= 0.5) / (P["fleck_edge"] * H) - 1.0, 0, 1)
    t = t * t * (3 - 2 * t) * hm * (1 - line) * edge
    n, cc, st, _ = cv2.connectedComponentsWithStats((t > 0).astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] <= P["fleck_area"] * H * H
    near = cv2.dilate((line >= 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    keep[np.unique(cc[near])] = False
    keep[0] = False
    t = t * keep[cc]
    return lab + t[..., None] * (med - lab)


def _prep(sprite, parts, lines, P):
    rgba = _f32(sprite)
    alpha = rgba[..., 3]
    H = figure_height(alpha)
    line = np.zeros_like(alpha) if lines is None else np.clip(_mask(lines), 0, 1)
    lab = to_lab(rgba[..., :3])
    if "hair" in parts:
        lab = _clean_hair(lab, _mask(parts["hair"]), line, alpha, H, P)
    return rgba, alpha, H, line, lab


def fit_blend(sprite, parts, target_rgb, target_parts, height, lines=None, target_sil=None, params=None):
    """Fit the look: per part tone curve, a*/b* shift and outer-line colour, measured at the game height.
    sprite: full-res RGBA (straight alpha); parts / target_parts: name -> mask; target_rgb: the painter's frame at
    game scale (figure about as tall as the sprite at `height`); target_sil: its silhouette (default: all parts)."""
    P = dict(BLEND, **(params or {}))
    rgba, alpha, H, line, lab = _prep(sprite, parts, lines, P)
    small = downscale(np.dstack([from_lab(lab), alpha]), height, P["filter"])
    size = small.shape[1::-1]
    sl, ssil = to_lab(small[..., :3]), small[..., 3] >= 0.5
    trgb = _f32(target_rgb)[..., :3]
    tl = to_lab(trgb)
    sdep = _depth(ssil)
    tm = {k: _mask(v) >= 0.5 for k, v in target_parts.items()}
    tsil = np.any(list(tm.values()), 0) if target_sil is None else _mask(target_sil) >= 0.5
    look = {"params": {k: list(v) if isinstance(v, tuple) else v for k, v in P.items()}, "parts": {}}
    pairs = [(k, _resize(_mask(parts[k]), size, P["filter"]), tm[k]) for k in parts if k in tm and k != "subject"]
    pairs.append(("_all", ssil.astype(np.float32), tsil))
    for k, sm, tmk in pairs:
        xs, sa, sb, ns = _stats(sl, ssil, sm, P["edge"], P["q"])
        ys, ta, tb, nt = _stats(tl, tsil, tmk, P["edge"], P["q"])
        if min(ns, nt) < P["min_px"]:
            continue
        edge = ssil & (sm >= 0.5) & (sdep <= P["ring_probe"])
        cov = float(np.median(small[..., 3][edge])) if edge.any() else 1.0
        xr, _, _, nr = _stats(sl, ssil, sm, P["edge"], P["q"], rim=True)
        yr, _, _, mr = _stats(tl, tsil, tmk, P["edge"], P["q"], rim=True)
        if min(nr, mr) < P["min_px"] // 3:                     # too thin a rim to measure: interior curve there too
            xr, yr = xs, ys
        r2 = lambda v: np.round(v, 2).tolist()
        look["parts"][k] = {"x": r2(xs), "y": r2(ys), "xr": r2(xr), "yr": r2(yr), "da": round(ta - sa, 2),
                            "db": round(tb - sb, 2), "n": [ns, nt],
                            "line": _ring(trgb, tsil, tmk, P["ring_probe"], cov)}
    return look


def _hold(out, ref, m):
    """Where `m` and `out` is lighter (L*) than `ref` (gamut clipping), take the channel-wise minimum."""
    up = m & (to_lab(out[..., :3])[..., 0] > to_lab(ref[..., :3])[..., 0])
    out[..., :3] = np.where(up[..., None], np.minimum(out[..., :3], ref[..., :3]), out[..., :3])
    return out


def _outer_line(res, ref, w, lp, width):
    """(4) Anti-aliased band `width` px inside the alpha edge in each part's target line colour (Lab), never lighter
    than `ref` (the input at this resolution): a pixel that is already darker keeps its own tone."""
    col = np.zeros(res.shape[:2] + (3,), np.float32)
    ws = np.zeros(res.shape[:2], np.float32)
    for k, wk in w.items():
        c = lp[k].get("line") or lp["_all"].get("line")
        if c is not None:
            col += wk[..., None] * np.float32(c)
            ws += wk
    a = res[..., 3]
    cov = np.clip(width - _depth(a >= 0.5) + 0.5, 0, 1) * (a > 0) * (ws > 1e-3)
    if not cov.any():
        return res
    col /= np.maximum(ws, 1e-3)[..., None]
    lab, L0 = to_lab(res[..., :3]), to_lab(ref[..., :3])[..., 0]
    rec = cov * np.clip((L0 - col[..., 0]) / 4.0, 0, 1)    # recolour only where the line is darker than the input
    lab[..., 0] += cov * (np.minimum(L0, col[..., 0]) - lab[..., 0])
    lab[..., 1:] += rec[..., None] * (col[..., 1:] - lab[..., 1:])
    out = res.copy()
    out[..., :3] = np.where(cov[..., None] > 0, from_lab(lab), res[..., :3])
    return _hold(out, ref, cov > 0)


def apply_blend(sprite, parts, look, height=None, lines=None):
    """Apply a fitted look to any frame of the scene; returns float RGBA at `height` rows. parts / lines at the
    sprite's resolution. The alpha is the plain downscale of the input alpha. height=None (full resolution) is for
    checks only: the game-px widths (edge, fade, outer line) then count full-res px."""
    P = dict(BLEND, **look["params"])
    rgba, alpha, H, line, lab = _prep(sprite, parts, lines, P)
    lp = dict(look["parts"])
    lp.setdefault("_all", {"x": [50.0], "y": [50.0], "da": 0.0, "db": 0.0, "line": None})  # leftover keeps its tone
    w = _weights(parts, [k for k in parts if k in lp], alpha, H, P["feather"])
    # (1) per part tone curve + a*/b* shift at full resolution
    scale = 1.0 if height is None else alpha.shape[0] / float(height)       # full-res px per game px
    fade = np.clip((_depth(alpha >= 0.5) / scale - P["edge"]) / P["fade"] + 0.5, 0, 1)   # 0 rim .. 1 interior
    L, out, Lt = lab[..., 0], lab.copy(), np.zeros_like(alpha)
    grid = np.linspace(0, 100, 1001)
    for k, wk in w.items():
        p = lp[k]
        f_in = np.interp(L, grid, _lut(p["x"], p["y"]))
        f_rim = np.interp(L, grid, _lut(p.get("xr", p["x"]), p.get("yr", p["y"])))
        Lt += wk * (f_rim + fade * (f_in - f_rim))
        out[..., 1] += wk * p["da"]
        out[..., 2] += wk * p["db"]
    out[..., 0] = np.where(Lt > L, Lt - line * (Lt - L), Lt)   # Adobe line art never gets lighter
    res = _hold(np.dstack([from_lab(out), alpha]).astype(np.float32), rgba, line > 0)
    fig = alpha > 0
    res[~fig, :3] = rgba[~fig, :3]
    ref = np.dstack([from_lab(lab), alpha]).astype(np.float32)
    ref[~fig, :3] = rgba[~fig, :3]
    # (3) premultiplied downscale (alpha = plain downscale of the input alpha), then (4) the outer line at game scale
    if height is not None:
        res, ref = downscale(res, height, P["filter"]), downscale(ref, height, P["filter"])
        w = {k: _resize(v, res.shape[1::-1], P["filter"]) for k, v in w.items()}
    return _outer_line(res, ref, w, lp, P["ring_w"])


def blend(sprite, parts, target_rgb, target_parts, height, lines=None, target_sil=None, params=None):
    """Fit on this frame and its target, apply to it: (rgba at game height, look)."""
    look = fit_blend(sprite, parts, target_rgb, target_parts, height, lines, target_sil, params)
    return apply_blend(sprite, parts, look, height, lines), look
