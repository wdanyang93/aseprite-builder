"""Line map (design 3): drawn lines are THIN strokes DARKER than the surface around them - not "dark pixels".

Why: an absolute L* threshold called 68 % of a navy outfit and 73-88 % of a black outfit "line", 53 % of dark skin
"dark cloth", 0 % of brown lines (L* ~45) and 0 % of the lines of a 192 px frame (R_audit 1 / 10). Here:
1. L* of the figure's own (straight) colours on a light background (the silhouette outline is a stroke against it),
   blurred a little (fraction of H) so that a resampled frame gets the same decision as its original;
2. black-hat: morphological closing of L* by a disc of radius `scale` x H (0.3-0.5 % H, calibrated per character in
   the character card) minus L*. The closing is the local surface level; only dark structures narrower than the disc
   respond, so dark areas wider than ~0.6 % H (navy / black clothes, dark skin, dark hair) are never lines;
3. ink share = (surface - L*) / (surface - ink L*): how much of a full-strength stroke the pixel is. The ink L* is the
   character's line colour (card), so the same rule finds black and brown lines and does not take a light print or a
   faint fur strand (share ~0.2-0.3) for line art; on a surface about as dark as the ink or darker (navy, black, dark
   skin) the full contrast is taken as 60 % of min(surface, ink) L* instead, so a faint fold mark there stays a mark;
4. hysteresis: seeds need a clear contrast (dl_seed, and share_hi or a third of the surface L* - a drawing may use a
   dark outline and lighter brown interior lines), the stroke grows through connected pixels that pass
   the weaker test (dl_min, share_lo) -> anti-aliased stroke edges follow their stroke, isolated noise does not;
5. soft weight = ink share on the stroke and the pixels touching it (anti-aliased edges keep their partial share),
   measured against the stroke's own darkest pixel nearby, so a lighter second ink is a full stroke at its core.
In fur / hair texture regions (optional strict mask from regions.texture) a stroke must be nearly ink-dark: the drawn
fur pattern (Lyn's brown tail band) and strand shading are texture that takes the light; dark lock separations stay.
Small frames: a stroke narrower than ~1.5 px is blended into its pixels, so the contrast tests scale with H (from the
card's stroke width) and the soft weight falls with the stroke's coverage -> the soft line share stays about the same
from 1400 px down to 48 px instead of dropping to 0.
Reuses the ideas of chest.true_line_weight (thin AND dark enough) and chest.ink_share (coverage of the stroke).
"""
import numpy as np
import cv2

from .color import to_lab
from .geom import figure_height

LINES = {
    "scale": 0.003,       # black-hat disc radius (fraction of H); the character card stores the calibrated value
    "blur": 0.0006,       # L* blur before deciding (fraction of H), skipped below 0.3 px
    "dl_min": 6.0,        # a stroke pixel is at least this much darker than its surface (L*)
    "dl_seed": 12.0,      # hysteresis seeds at least this much
    "share_lo": 0.35,     # ink share needed to grow a stroke
    "share_hi": 0.6,      # ink share needed for a seed
    "rel_seed": 0.33,     # ... or at least this share of the surface L* below it (lighter second ink, e.g. brown
                          #     interior lines in a drawing with a dark outline)
    "rel_floor": 0.6,     # the full stroke contrast is never taken below this share of min(surface L*, ink L*)
    "rel_dark": 0.2,      # on a surface about as dark as the ink (where that floor applies): a seed needs this
                          #     contrast relative to the surface L* (Weber) ...
    "w_dark": 0.3,        # ... and a stroke of this Weber contrast has full weight
    "strict_seed": 0.85,  # inside fur / hair texture (strict mask): ink share needed for a seed ...
    "strict_grow": 0.7,   # ... and to grow; not within strict_edge x H of the silhouette (the outline stays a line)
    "strict_edge": 0.01,
    "ink_L": None,        # ink L* (character card); None = estimate from the frame itself
    "ink_default": 30.0,  # used when neither the card nor the frame gives one
    "ink_max": 60.0,      # an estimated ink lighter than this is not trusted (small frames: strokes are blended)
    "min_alpha": 0.5,     # line map only on pixels of the figure
    "max_width": 0.006,   # stroke parts wider than this (fraction of H; or 1.5 x the card's stroke width) are dark
                          # areas, not lines, even where the black-hat disc is larger
    "width": 0.002,       # typical stroke width (fraction of H; character card). Below res_px px a stroke is
    "res_px": 1.5,        # blended into its pixels, so the contrast tests scale down with H ...
    "res_floor": 0.45,    # ... but not below this share of their value
}


def _hysteresis(grow, seed):
    n, lab = cv2.connectedComponents(grow.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[seed & grow])] = True
    keep[0] = False
    return keep[lab]


def estimate_ink(L, bh, fig, p=None):
    """Ink L* of a frame: the dark end of its strong strokes (None when there are too few)."""
    p = dict(LINES, **(p or {}))
    s = fig & (bh >= 2 * p["dl_seed"])
    if s.sum() < 30:
        return None
    v = float(np.percentile(L[s], 20))
    return v if v <= p["ink_max"] else None


def _masked_blur(L, fig, s):
    """Gaussian blur of L over the figure's pixels only (normalized convolution): transparent pixels never leak in."""
    if s < 0.3:
        return L
    w = fig.astype(np.float32)
    num = cv2.GaussianBlur(L * w, (0, 0), s)
    den = cv2.GaussianBlur(w, (0, 0), s)
    return np.where(fig, num / np.maximum(den, 1e-3), L).astype(np.float32)


def _surface(Lb, fig, r):
    """Local surface L*: morphological closing by a disc of radius r over the FIGURE's pixels only (a disc's level is
    the lightest figure pixel in it; discs with no figure pixel do not count). So the background is no stroke
    contrast: a thin mid-tone limb of a small frame is its own surface, not a dark stroke against a white void, while
    the silhouette outline is still darker than the figure surface next to it."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    pad = r + 1
    lo = cv2.copyMakeBorder(np.where(fig, Lb, -1.0).astype(np.float32), pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                            value=-1.0)
    dil = cv2.dilate(lo, k)
    dil = np.where(dil < 0, 1000.0, dil).astype(np.float32)
    surf = cv2.erode(dil, k, borderType=cv2.BORDER_CONSTANT, borderValue=1000.0)[pad:-pad, pad:-pad]
    return np.where(fig, np.maximum(surf, Lb), Lb).astype(np.float32), k


def line_map(rgba, H=None, p=None, strict=None):
    """Line map of an RGBA frame (float 0..1, straight alpha). Returns a dict:
    weight (HxW soft 0..1: ink share on strokes and their anti-aliased edges), mask (weight >= 0.5: mostly ink),
    stroke (every pixel of a detected stroke, also the faint ones of a small frame), bh (black-hat,
    L* below the local surface), surface (local surface L*), L (L* used), ink_L, scale_px (black-hat disc radius).
    strict: optional HxW bool (fur / hair texture regions, regions.texture): there a stroke away from the silhouette
    must be nearly as dark as the ink (strict_seed / strict_grow) - drawn fur pattern and strand shading are texture;
    the silhouette outline (within strict_edge x H of the edge) and ink-dark strokes stay line art."""
    p = dict(LINES, **(p or {}))
    rgba = np.asarray(rgba, np.float32)
    a = rgba[..., 3]
    H = H or figure_height(a)
    fig = a > p["min_alpha"]
    L = to_lab(np.clip(rgba[..., :3], 0, 1))[..., 0].astype(np.float32)
    L = np.where(fig, L, 0).astype(np.float32)               # transparent pixels: never read
    Lb = _masked_blur(L, fig, p["blur"] * H)
    r = max(1, int(round(p["scale"] * H)))
    surf, k = _surface(Lb, fig, r)
    bh = np.where(fig, surf - Lb, 0).astype(np.float32)
    bh_raw = np.where(fig, np.maximum(surf - L, 0), 0)       # seeds also from the unblurred L*: 1 px strokes
    q = float(np.clip(p["width"] * H / p["res_px"], p["res_floor"], 1.0))   # 1 = strokes resolved
    est = estimate_ink(Lb, bh, fig, p) if q >= 0.75 else None   # blended strokes of a small frame look lighter
    cands = [v for v in (p["ink_L"], est) if v is not None]
    ink = max(cands) if cands else p["ink_default"]           # a frame drawn with lighter lines than the card's
    floor = p["rel_floor"] * np.minimum(surf, ink)
    den = np.maximum(np.maximum(surf - ink, p["dl_seed"]), floor)
    share = bh / den
    # a surface about as dark as the ink or darker (dark skin, navy, black cloth): the full stroke contrast is not
    # surface - ink there, so a stroke counts by its contrast relative to the surface (Weber) instead
    dark = (surf - ink) < floor
    surf1 = np.maximum(surf, 1.0)
    bs = np.maximum(bh, bh_raw)
    strong = ((bs / den >= q * p["share_hi"]) | (bs >= q * p["rel_seed"] * surf1)
              | (dark & (bs >= q * p["rel_dark"] * surf1)))
    seed = fig & strong & (bs >= q * np.where(dark, np.maximum(p["dl_min"], p["rel_dark"] * surf1), p["dl_seed"]))
    grow = fig & (share >= q * p["share_lo"]) & (bh >= q * p["dl_min"])
    if strict is not None:
        dist = cv2.distanceTransform(fig.astype(np.uint8), cv2.DIST_L2, 3)
        st = strict & (dist > p["strict_edge"] * H)            # the silhouette outline is never "texture"
        seed &= ~st | (bs / den >= q * p["strict_seed"])
        grow &= ~st | (share >= q * p["strict_grow"])
    core = _hysteresis(grow, seed)
    dw = max(3.0, p["max_width"] * H, 1.5 * p["width"] * H)    # dark parts wider than this are areas, not strokes
    kw = int(np.ceil(dw)) | 1
    if kw < 2 * r + 1:
        core &= ~(cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kw))) > 0)
    near = (cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & fig & (bh > 0.5 * q * p["dl_min"])
    # soft weight: ink share against the stroke's OWN darkest pixel nearby (a lighter second ink is still a full
    # stroke at its core); on small frames, where a dark pixel is itself a blended stroke, against the character ink;
    # on a dark surface a stroke of Weber contrast w_dark is full strength
    ink_loc = np.maximum(np.where(fig, cv2.erode(np.where(fig, Lb, 1000.0).astype(np.float32), k), Lb), ink)
    t = float(np.clip((q - 0.5) / 0.25, 0, 1))
    ink_ref = ink + t * (ink_loc - ink)
    fl = p["rel_floor"] * np.minimum(surf, ink_ref)
    fl = np.where(dark, np.minimum(fl, p["w_dark"] * surf1), fl)
    wsh = bh / np.maximum(surf - ink_ref, np.maximum(q * p["dl_min"], fl))
    w = np.where(core | near, np.clip(wsh, 0, 1), 0).astype(np.float32)
    return {"weight": w, "mask": w >= 0.5, "stroke": core, "bh": bh.astype(np.float32),
            "surface": surf.astype(np.float32), "L": Lb, "ink_L": float(ink), "scale_px": r, "res_q": q}


def stroke_width(weight, mask, H):
    """Typical stroke width as a fraction of H: soft stroke area / stroke length, the length counted as the ridge
    pixels of the stroke mask (local maxima of the distance to its edge). None when there are no strokes."""
    m = mask.astype(np.uint8)
    if m.sum() < 20 or H <= 0:
        return None
    d = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ridge = (d > 0) & (d >= cv2.dilate(d, np.ones((3, 3), np.uint8)) - 1e-6)
    if ridge.sum() < 10:
        return None
    return float(weight.sum() / ridge.sum() / H)
