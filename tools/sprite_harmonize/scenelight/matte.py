"""Input preparation (F0): give a flat-background reference an alpha, and optional edge defringe.

A hand-lit reference painted on a flat colour (grey #808080 or any flat colour, RGB without alpha) is matted:
  1. the background is estimated from the canvas border (robust plane per channel: a flat colour or a gentle gradient);
     a border that is not flat is refused with a clear error
  2. colour key: pixels that differ from the background; small specks dropped; enclosed holes filled unless they are
     flat background (real gaps between limbs stay open)
  3. the ORIGINAL's alpha is moved onto the reference: IoU similarity (scenelight.align) then an ECC affine refinement.
     ECC returns the map reference -> original coordinates, so the warp uses WARP_INVERSE_MAP
  4. the anti-aliased ring is unmixed from the background (C = a F + (1 - a) B); where the moved alpha covers pixels the
     reference left as background, the nearest painted colour is used (no grey fringe, outline ink kept)
Modes: 'orig' = the original's alpha moved (the reference keeps the original outline: checked by IoU against the key),
'key' = the reference's own silhouette (soft edge from the two-colour coverage estimate), 'auto' = 'key' when the key's
edge has enough contrast to the background (then every edge colour is the reference's real paint; the moved alpha
overhangs a repainted edge by a few px and those colours would be filled in), else 'orig'. In 'auto' / 'orig' a reference
whose silhouette does not keep the original outline (IoU < MIN_IOU) is refused. A reference that already has a real alpha
is passed through unchanged. info reports both IoUs, leak / cut shares and the share of filled-in ('invented') colours.

Defringe (optional, off by default): scenelight.color.defringe replaces semi-transparent edge colours with a blurred
interior average, which lightens drawn outlines. defringe(mode="nearest") takes the nearest opaque pixel's colour
instead (the outline's own ink), so a light halo is removed without lightening the line.

All images are float32 RGBA / RGB in [0, 1] (uint8 input is accepted). Length constants are fractions of the figure height.
"""
import json
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi

from .align import _similarity
from .color import defringe as _defringe_interior, figure_height, load_rgba, save_rgba, to_lab

MIN_IOU = 0.90       # below: the reference does not keep the original outline -> refused (auto / orig)
ORIG_IOU = 0.95      # 'orig' below this IoU (moved alpha vs key) is reported as a warning
KEY_EDGE_OK = 0.9    # auto takes 'key' when this share of the key's edge has enough contrast to the background
BORDER = 0.02        # border strip used for the background estimate (fraction of the canvas' shorter side)
OPEN = 0.002         # speck removal (morphological opening), fraction of H
MIN_PART = 0.002     # components smaller than this share of the largest one are dropped
HOLE_MIN = 0.01      # holes smaller than (HOLE_MIN H)^2 are always filled; larger flat-background holes stay open
BAND = 0.002         # boundary band for unmixing / coverage estimate (fraction of H, at least 2 px)
ECC_H = (256, 640)   # ECC working figure heights (coarse, fine)
A_MIN = 0.25         # unmixing divides by at least this alpha


class MatteError(ValueError):
    """The reference cannot be matted under the stated assumptions (message says why and what to do)."""


def _f32(img):
    x = np.asarray(img)
    if x.dtype == np.uint8 or (x.size and float(x.max()) > 1.5):
        return x.astype(np.float32) / 255.0
    return x.astype(np.float32)


def has_real_alpha(img):
    """True when an RGBA image carries a real cut-out alpha (clearly transparent canvas border + opaque figure)."""
    x = _f32(img)
    if x.ndim != 3 or x.shape[2] < 4:
        return False
    a = x[..., 3]
    b = max(2, int(BORDER * min(a.shape)))
    border = np.concatenate([a[:b].ravel(), a[-b:].ravel(), a[:, :b].ravel(), a[:, -b:].ravel()])
    return bool((border < 0.5).mean() > 0.9 and (a > 0.5).mean() > 0.005 and (a < 0.5).mean() > 0.05)


# ------------------------------------------------------------------ background
def background(rgb, border=BORDER):
    """Robust per-channel plane fit B(x, y) = c0 + c1 u + c2 v on the canvas border (u, v in [-1, 1]).
    Returns dict: coef (3x3), sigma (noise, max over channels), inliers (share of border pixels on the plane),
    range (max-min of the plane over the canvas), colour (plane at the canvas centre, 0..1)."""
    H, W = rgb.shape[:2]
    b = max(4, int(border * min(H, W)))
    m = np.zeros((H, W), bool)
    m[:b], m[-b:], m[:, :b], m[:, -b:] = True, True, True, True
    ys, xs = np.nonzero(m)
    if len(ys) > 40000:                                   # deterministic subsample
        k = np.linspace(0, len(ys) - 1, 40000).astype(int)
        ys, xs = ys[k], xs[k]
    u, v = xs / max(W - 1, 1) * 2 - 1, ys / max(H - 1, 1) * 2 - 1
    X = np.stack([np.ones_like(u), u, v], 1)
    Y = rgb[ys, xs, :3].astype(np.float64)
    keep = np.ones(len(ys), bool)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(X[keep], Y[keep], rcond=None)
        res = np.abs(Y - X @ coef).max(1)
        mad = np.median(res[keep]) + 1e-6
        keep = res < max(6 * 1.4826 * mad, 3 / 255)
    res = Y[keep] - X[keep] @ coef
    sigma = float(np.max(1.4826 * np.median(np.abs(res - np.median(res, 0)), 0)))
    corners = np.array([[1, -1, -1], [1, 1, -1], [1, -1, 1], [1, 1, 1]], np.float64) @ coef
    return {"coef": coef.astype(np.float32), "sigma": max(sigma, 0.5 / 255), "inliers": float(keep.mean()),
            "range": float((corners.max(0) - corners.min(0)).max()), "colour": coef[0].astype(np.float32)}


def bg_image(bgm, shape):
    H, W = shape[:2]
    u = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    v = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    c = bgm["coef"]
    return np.dstack([c[0, k] + c[1, k] * u + c[2, k] * v for k in range(3)]).astype(np.float32)


def _check_flat(bgm):
    if bgm["inliers"] < 0.8 or bgm["range"] > 0.15 or bgm["sigma"] > 0.03:
        raise MatteError(
            "reference background is not flat (border: {:.0%} on a plane, gradient range {:.3f}, noise {:.3f}): "
            "paint the reference on one flat colour, or export it with a real alpha".format(
                bgm["inliers"], bgm["range"], bgm["sigma"]))


# ------------------------------------------------------------------ colour key
def key_silhouette(rgb, B, sigma, H=None):
    """Reference silhouette from a colour key against the background image B.
    Returns (mask, info): specks removed, small components dropped, holes filled unless flat background."""
    d = np.abs(rgb[..., :3] - B).max(-1)
    thr = max(12 / 255, 6 * sigma)
    F = d > thr
    if H is None:
        H = figure_height(F.astype(np.float32))
    k = max(3, int(round(OPEN * H)) | 1)
    F = cv2.morphologyEx(F.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(F.astype(np.uint8), connectivity=8)
    if n <= 1:
        raise MatteError("no figure found on the reference (nothing differs from the background colour)")
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep_ids = 1 + np.nonzero(st[1:, cv2.CC_STAT_AREA] >= MIN_PART * big)[0]
    K = np.isin(lab, keep_ids)
    H = figure_height(K.astype(np.float32))
    holes = ndi.binary_fill_holes(K) & ~K
    hl, nh = ndi.label(holes)
    kept = filled = 0
    if nh:
        idx = np.arange(1, nh + 1)
        area = ndi.sum(np.ones_like(d), hl, idx)
        core = np.where(cv2.erode(holes.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, hl, 0)   # away from the rim
        ncore = ndi.sum(np.ones_like(d), core, idx)
        with np.errstate(invalid="ignore", divide="ignore"):     # holes without a core give NaN -> filled
            dmean = ndi.mean(d, core, idx)
            dstd = ndi.standard_deviation(rgb[..., :3].mean(-1), core, idx)
        flat_bg = ((area >= (HOLE_MIN * H) ** 2) & (ncore >= 0.2 * area) & (dmean <= 2.5 * sigma + 2 / 255)
                   & (dstd <= 2.5 * sigma + 1 / 255))
        fill_ids = idx[~flat_bg]
        K = K | np.isin(hl, fill_ids)
        kept, filled = int(flat_bg.sum()), int(len(fill_ids))
    return K, {"thr": float(thr), "holes_kept": kept, "holes_filled": filled, "H": float(H)}


# ------------------------------------------------------------------ moving the original alpha
def _iou(a, b):
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum() / max((a | b).sum(), 1))


def _h3(T):
    return np.vstack([T, [0, 0, 1]]).astype(np.float64)


def align_alpha(alpha, key):
    """Affine map T (2x3, reference coords -> original coords) moving the original alpha onto the key mask.
    Similarity by silhouette IoU (alpha only), then ECC (MOTION_AFFINE) coarse-to-fine on blurred masks.
    Returns (alpha moved into the key's canvas, T, info)."""
    h, w = key.shape
    Kf = key.astype(np.float32)
    M, _ = _similarity(alpha, Kf)                       # original -> reference (forward)
    A0 = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    iou_init = _iou(A0, Kf)
    T1 = np.eye(2, 3, dtype=np.float32)                 # reference -> A0 coordinates
    Hk = figure_height(Kf)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    ok = True
    for th in ECC_H:
        s = min(1.0, th / Hk)
        size = (max(8, int(round(w * s))), max(8, int(round(h * s))))
        a_s = cv2.GaussianBlur(cv2.resize(A0, size, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
        k_s = cv2.GaussianBlur(cv2.resize(Kf, size, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
        Ts = T1.copy()
        Ts[:, 2] *= s
        try:
            _, Ts = cv2.findTransformECC(k_s, a_s, Ts, cv2.MOTION_AFFINE, crit, None, 5)
        except cv2.error:
            ok = False
            break
        T1 = Ts.copy()
        T1[:, 2] /= s
    T = (_h3(cv2.invertAffineTransform(M)) @ _h3(T1))[:2].astype(np.float32)
    Aw = cv2.warpAffine(alpha, T, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    iou = _iou(Aw, Kf)
    if not ok or iou < iou_init:                        # ECC diverged: keep the IoU similarity
        T, Aw, iou = cv2.invertAffineTransform(M).astype(np.float32), A0, iou_init
    return np.clip(Aw, 0, 1), T, {"iou_init": round(iou_init, 4), "iou_aligned": round(iou, 4), "ecc_ok": ok}


# ------------------------------------------------------------------ unmixing
def _nearest(values, sure):
    """Each pixel takes the value of the nearest 'sure' pixel."""
    if not sure.any():
        return values
    _, (iy, ix) = ndi.distance_transform_edt(~sure, return_indices=True)
    return values[iy, ix]


def _coverage(C, B, Fn, sigma):
    """Two-colour model C = B + a (F - B) with F = the nearest painted colour: coverage a, enough contrast between
    F and B, and whether the pixel lies on that line (small residual)."""
    dFB = Fn - B
    n2 = (dFB ** 2).sum(-1)
    a = np.clip(((C - B) * dFB).sum(-1) / np.maximum(n2, 1e-8), 0, 1)
    contrast = np.sqrt(n2) > max(0.08, 10 * sigma)
    resid = np.abs(C - (B + a[..., None] * dFB)).max(-1)
    return a, contrast, contrast & (resid < max(0.06, 6 * sigma))


def _unmix(C, B, a):
    return np.clip(B + (C - B) / np.maximum(a, A_MIN)[..., None], 0, 1)


def _colours(C, B, Fn, a, band, a_obs, contrast, model, painted):
    """Straight colour of the figure on the boundary band (interior: the reference's own colour).
    low contrast to the background: plain unmixing with a; background-coloured pixels under the alpha (the reference did
    not paint there): nearest painted colour; on the two-colour line: unmix with the observed coverage; else keep C."""
    F = C.copy()
    ba = band & (a > 0)
    low = ba & ~contrast
    F[low] = _unmix(C, B, a)[low]
    mk = ba & contrast
    leak = mk & (~painted | (model & (a_obs < A_MIN)))
    F[leak] = Fn[leak]
    mix = mk & ~leak & model
    F[mix] = (B + (C - B) / np.maximum(a_obs, A_MIN)[..., None])[mix]
    return np.clip(F, 0, 1)


def matte(orig, ref, mode="auto", min_iou=MIN_IOU, orig_iou=ORIG_IOU, return_masks=False):
    """Give `ref` (flat-background RGB/RGBA) an alpha. orig: the original RGBA (needed for 'orig' / 'auto').
    Returns (rgba float32 in the reference's canvas, info). Raises MatteError when the assumptions do not hold."""
    if mode not in ("auto", "orig", "key"):
        raise ValueError(f"mode must be auto, orig or key (got {mode!r})")
    R = _f32(ref)
    if has_real_alpha(R):
        return R.copy(), {"mode": "alpha", "note": "reference already has a real alpha: passed through unchanged"}
    C = np.ascontiguousarray(R[..., :3])
    if orig is None and mode != "key":
        if mode == "orig":
            raise MatteError("mode 'orig' needs the original RGBA")
        mode = "key"
    bgm = background(C)
    _check_flat(bgm)
    B = bg_image(bgm, C.shape)
    sigma = bgm["sigma"]
    K, kinfo = key_silhouette(C, B, sigma)
    H = kinfo["H"]
    info = {"bg": [round(float(v) * 255, 1) for v in bgm["colour"]], "bg_sigma": round(sigma * 255, 2),
            "bg_range": round(bgm["range"] * 255, 1), "key_thr": round(kinfo["thr"] * 255, 1),
            "holes_kept": kinfo["holes_kept"], "holes_filled": kinfo["holes_filled"], "H": H}
    painted = np.abs(C - B).max(-1) > kinfo["thr"]
    r = max(2, int(round(BAND * H)))
    ker = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    sure = cv2.erode((K & painted).astype(np.uint8), k3) > 0      # painted, with painted neighbours (inner ink too)
    Fn = _nearest(C, sure)
    a_obs, contrast, model = _coverage(C, B, Fn, sigma)
    # the key threshold catches dark anti-aliased pixels below half coverage: put the key's edge at coverage 0.5
    edge_out = K & ~(cv2.erode(K.astype(np.uint8), k3) > 0)
    edge_in = (cv2.dilate(K.astype(np.uint8), k3) > 0) & ~K
    K = (K & ~(edge_out & model & (a_obs < 0.5))) | (edge_in & model & (a_obs >= 0.5))

    Aw = None
    if orig is not None:
        O = _f32(orig)
        if O.ndim != 3 or O.shape[2] < 4:
            raise MatteError("the original must be RGBA (its alpha is moved onto the reference)")
        Aw, T, ainfo = align_alpha(np.ascontiguousarray(O[..., 3]), K)
        info.update(ainfo, affine=np.round(T, 4).tolist())
        fig = max(int((Aw > 0.5).sum()), 1)
        info["leak"] = round(float(((Aw > 0.5) & ~K).sum() / fig), 4)    # moved alpha over unpainted pixels
        info["cut"] = round(float((K & (Aw < 0.5)).sum() / fig), 4)      # reference paint outside the moved alpha
        iou = ainfo["iou_aligned"]
        if mode in ("auto", "orig") and iou < min_iou:
            raise MatteError(
                f"the reference does not keep the original outline: silhouette IoU {iou:.3f} < {min_iou} after an "
                "affine fit (redrawn pose, cropped or different framing?). Repaint on the original canvas/pose, or use "
                "mode='key' to take the reference's own silhouette")
    edge_k = K & ~(cv2.erode(K.astype(np.uint8), k3) > 0)
    info["key_edge_contrast"] = round(float(contrast[edge_k].mean()), 3) if edge_k.any() else 0.0
    if mode == "auto":
        # the reference's own outline carries its real paint (rim light lives there); the moved original alpha would
        # invent colours where it overhangs the repainted edge. Use it only when the key's edge is unreliable.
        if info["key_edge_contrast"] >= KEY_EDGE_OK:
            mode = "key"
            info["auto"] = f"key edge contrast {info['key_edge_contrast']:.3f} >= {KEY_EDGE_OK}: key"
        else:
            mode = "orig" if Aw is not None else "key"
            info["auto"] = f"key edge contrast {info['key_edge_contrast']:.3f} < {KEY_EDGE_OK}: {mode}"
            if Aw is not None and iou < orig_iou:
                info["warning"] = f"low-contrast key and moved-alpha IoU {iou:.3f} < {orig_iou}: check the matte"
    info["mode"] = mode

    if mode == "orig":
        a = Aw
        band = cv2.dilate(((a > 0.02) & (a < 0.98)).astype(np.uint8), ker) > 0
        band |= cv2.dilate((K ^ (a > 0.5)).astype(np.uint8), k3) > 0
    else:
        edge = (cv2.dilate(K.astype(np.uint8), k3) > 0) & ~(cv2.erode(K.astype(np.uint8), k3) > 0)
        est = np.where(model, a_obs, cv2.GaussianBlur(K.astype(np.float32), (0, 0), 0.7))
        a = K.astype(np.float32)
        a = np.where(edge & K, np.clip(est, 0.5, 1), a)
        a = np.where(edge & ~K, np.clip(est, 0, 0.49), a)
        a = np.where(a < 0.03, 0, a).astype(np.float32)
        band = edge
        if Aw is not None:
            info["iou_key_vs_moved"] = round(_iou(a, Aw), 4)
    F = _colours(C, B, Fn, a, band, a_obs, contrast, model, painted)
    out = np.dstack([F, np.clip(a, 0, 1)]).astype(np.float32)
    # observed = the colour comes from the reference's own paint (not filled in where the alpha overhangs its edge)
    observed = ~(band & (a > 0) & contrast & (~painted | (model & (a_obs < A_MIN))))
    info["invented"] = round(float((~observed & (a > 0.5)).sum() / max(int((a > 0.5).sum()), 1)), 4)
    if return_masks:
        info["masks"] = {"key": K, "painted": painted, "moved_alpha": Aw, "observed": observed}
    return out, info


def matte_file(orig_path, ref_path, mode="auto", **kw):
    """Convenience: paths in, (rgba, info) out. orig_path may be None (key mode)."""
    from PIL import Image
    im = Image.open(ref_path)
    ref = np.asarray(im.convert("RGBA" if im.mode in ("RGBA", "LA", "PA", "P") else "RGB"))
    orig = load_rgba(orig_path) if orig_path else None
    return matte(orig, ref, mode, **kw)


# ------------------------------------------------------------------ defringe (optional)
def defringe(rgba, mode=None, radius=4):
    """Optional edge colour clean-up of a straight-alpha RGBA (alpha is never changed).
    mode None / 'off': unchanged.  'interior': scenelight.color.defringe (blurred interior colour - lightens outlines).
    'nearest': semi-transparent pixels take the nearest opaque pixel's colour (alpha < 0.5 fully, 0.5..0.95 blended),
    i.e. the outline's own ink: a light halo goes away and the drawn line is not lightened."""
    x = _f32(rgba).copy()
    if mode in (None, "off", False):
        return x
    if mode == "interior":
        x[..., :3] = _defringe_interior(x[..., :3], x[..., 3], radius)
        return x
    if mode != "nearest":
        raise ValueError(f"defringe mode must be None, 'nearest' or 'interior' (got {mode!r})")
    a = x[..., 3]
    solid = a > 0.95
    if not solid.any():
        return x
    fill = _nearest(x[..., :3], solid)
    w = np.clip((a - 0.5) / 0.45, 0, 1)[..., None]
    x[..., :3] = x[..., :3] * w + fill * (1 - w)
    return x


def edge_report(rgba, out=None):
    """Edge-ring statistics (alpha 0.05..0.95). halo = ring L* minus the nearest opaque pixel's L* (light fringe > 0).
    With `out` (same image after defringe): dL = L* change on the ring (alpha-weighted mean, p95, share > +5 L*)."""
    x = _f32(rgba)
    a = x[..., 3]
    ring = (a > 0.05) & (a < 0.95)
    L = to_lab(np.clip(x[..., :3], 0, 1))[..., 0]
    Ln = _nearest(L, a > 0.95)
    rep = {"ring_px": int(ring.sum()), "halo_mean": round(float((L - Ln)[ring].mean()), 2) if ring.any() else 0.0}
    if out is not None and ring.any():
        dL = to_lab(np.clip(_f32(out)[..., :3], 0, 1))[..., 0] - L
        w = a[ring]
        rep.update({"dL_mean_aw": round(float((dL[ring] * w).sum() / w.sum()), 2),
                    "dL_p95": round(float(np.percentile(dL[ring], 95)), 2),
                    "lighter5_px": int((dL[ring] > 5).sum()), "lighter10_px": int((dL[ring] > 10).sum()),
                    "darker5_px": int((dL[ring] < -5).sum())})
    return rep


def prepare(rgba, defringe_mode=None):
    """F0 for a sprite that already has alpha: float RGBA, optional defringe (default off)."""
    return defringe(_f32(rgba), defringe_mode)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="matte a flat-background reference with the original's alpha")
    p.add_argument("reference"); p.add_argument("out")
    p.add_argument("--orig", help="original RGBA (needed for mode orig / auto)")
    p.add_argument("--mode", default="auto", choices=("auto", "orig", "key"))
    p.add_argument("--min-iou", type=float, default=MIN_IOU)
    a = p.parse_args(argv)
    try:
        out, info = matte_file(a.orig, a.reference, a.mode, min_iou=a.min_iou)
    except MatteError as e:
        print("ERROR:", e)
        return 2
    save_rgba(a.out, out)
    print(a.out, json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
