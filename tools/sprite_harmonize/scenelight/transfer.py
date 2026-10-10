"""Reference transfer: give frames the hand-lit reference's own soft light and colour.

fit (once per scene, needs the neutral + reference of the same pose) -> a "pack":
  - the reference warped into the NEUTRAL's geometry (similarity + optical flow), and two confidence maps:
    fine = local agreement of the two drawings' edge structure, cover = both silhouettes overlap
apply (per frame):
  1. the frame is scaled to the neutral's figure height, the neutral is aligned onto it and the reference follows
  2. gates (scale-independent): no folded deformation, drawing-structure match, pre-flow silhouette overlap
     -> a different direction / outfit / drawing gets the plain wash+rim look, never partial reference patches
  3. the delta is measured against the FRAME's own colours (so the neutral's details can't be stamped on):
       inside the body: soft colour / brightness only (sigma 1.5% of figure height; drawn form kept 0.953)
       near the silhouette (exp falloff ~2%): the reference's rim light, low-passed ALONG the contour
     line art may darken with the scene but is never lightened
  4. blended with the wash+rim relight by a smooth confidence weight

The pack holds pixel data of your art (neutral + warped reference): keep it with the project, not in a public repo.
"""
import os

import cv2
import numpy as np

from .align import Pairs, _gray
from .color import apply_color, from_lab, load_rgba, lowpass, materials, save_rgba, to_lab
from . import shading

PACK_VERSION = 2
SIGMA_FINE = 0.008       # rim layer low-pass (fraction of figure height), applied per distance band
SIGMA_IN = 0.015         # interior low-pass: soft light/colour; swept 0.015-0.06, smallest keeping drawn form >= 0.95
EDGE_FALLOFF = 0.02      # how far the rim layer reaches inward
BANDS = (0.0, 0.0015, 0.003, 0.005, 0.008, 0.0115, 1e9)   # distance-to-silhouette bands (fraction of height)
# calibrated on 40+ cases: same drawing / redraw / garment change pass with match >= 0.84, fold <= 0.032, pre-flow IoU >= 0.75;
# other directions / outfits (any scale), mirrored, cropped fall back with match <= 0.52, fold >= 0.084, IoU <= 0.67
GATES = {"max_fold": 0.055, "min_match": 0.68, "min_iou_affine": 0.71}
WORK_H = 1024            # working figure height (px): the pack is scaled down to this once at load


def _fig_h(alpha):
    ys = np.nonzero(alpha.max(1) > 0.5)[0]
    return float(ys.max() - ys.min() + 1) if len(ys) else 0.0


def edge_confidence(a_img, b_img, Hf, lo=0.25, hi=0.6, flat_ok=True):
    """Local correlation of edge strength between two aligned drawings -> 0..1, smoothed inside the figure.
    flat_ok: windows that are flat in both drawings count as agreeing (used for local weights, not for gating)."""
    g = lambda x: cv2.GaussianBlur(_gray(x).astype(np.float32), (0, 0), 1.5)
    e = lambda G: np.hypot(cv2.Sobel(G, cv2.CV_32F, 1, 0), cv2.Sobel(G, cv2.CV_32F, 0, 1))
    ea, eb = e(g(a_img)), e(g(b_img))
    w = int(0.04 * Hf) | 1
    mu = lambda x: cv2.blur(x, (w, w))
    va, vb = np.maximum(mu(ea * ea) - mu(ea) ** 2, 0), np.maximum(mu(eb * eb) - mu(eb) ** 2, 0)
    ncc = (mu(ea * eb) - mu(ea) * mu(eb)) / np.sqrt(np.maximum(va * vb, 1e-6))
    body = (a_img[..., 3] > 0.5) & (b_img[..., 3] > 0.5)
    if flat_ok and body.any():
        t = 0.02 * float(np.median(np.concatenate([va[body], vb[body]])))
        ncc = np.where((va < t) & (vb < t), 1.0, ncc)
    c = np.clip((ncc - lo) / (hi - lo), 0, 1).astype(np.float32)
    return lowpass(c, body, max(1.0, 0.02 * Hf)) * body


def build_pack(neutral, ref, profile=None):
    P = Pairs(ref, neutral, regions=False)          # reference -> neutral geometry
    rn = P.warp(ref)
    Hf = _fig_h(neutral[..., 3])
    a = neutral[..., 3] > 0.5
    conf_fine = edge_confidence(neutral, rn, Hf)
    both = ((neutral[..., 3] > 0.9) & (rn[..., 3] > 0.9)).astype(np.float32)
    conf_cover = lowpass(both, a, max(1.0, 0.02 * Hf)) * a
    return {"neutral": neutral, "ref": rn, "conf_fine": conf_fine.astype(np.float16),
            "conf_cover": conf_cover.astype(np.float16), "align_iou": P.iou}


def save_pack(path, pack):
    os.makedirs(path, exist_ok=True)
    old = os.path.join(path, "delta.npz")   # v1 leftover
    if os.path.exists(old):
        os.remove(old)
    save_rgba(os.path.join(path, "neutral.png"), pack["neutral"])
    save_rgba(os.path.join(path, "reference_in_neutral.png"), pack["ref"])
    np.savez_compressed(os.path.join(path, "conf.npz"), conf_fine=pack["conf_fine"], conf_cover=pack["conf_cover"],
                        align_iou=np.float32(pack["align_iou"]), version=np.int32(PACK_VERSION))


def load_pack(path, work_h=WORK_H):
    need = [os.path.join(path, f) for f in ("neutral.png", "reference_in_neutral.png", "conf.npz")]
    if not all(os.path.exists(p) for p in need):
        if os.path.exists(os.path.join(path, "delta.npz")):
            raise ValueError(f"{path}: reference pack from an older version - re-run `fit --pack`")
        raise FileNotFoundError(f"{path}: not a reference pack (expected neutral.png, reference_in_neutral.png, conf.npz)")
    try:
        with np.load(need[2]) as z:
            ver = int(z["version"])
            cf, cc, iou = z["conf_fine"].astype(np.float32), z["conf_cover"].astype(np.float32), float(z["align_iou"])
        n, r = load_rgba(need[0]), load_rgba(need[1])
    except Exception as e:  # truncated / foreign files
        raise ValueError(f"{path}: damaged reference pack ({e.__class__.__name__}: {e}) - re-run `fit --pack`") from e
    if ver != PACK_VERSION:
        raise ValueError(f"{path}: pack version {ver} != {PACK_VERSION} - re-run `fit --pack`")
    if not (n.shape == r.shape and cf.shape == cc.shape == n.shape[:2]):
        raise ValueError(f"{path}: pack images have different sizes - re-run `fit --pack`")
    k = min(1.0, work_h / max(_fig_h(n[..., 3]), 1.0))
    if k < 0.999:  # all per-frame work happens at this figure height
        size = (max(1, round(n.shape[1] * k)), max(1, round(n.shape[0] * k)))
        rs = lambda x: cv2.resize(x, size, interpolation=cv2.INTER_AREA)
        n, r, cf, cc = np.clip(rs(n), 0, 1), np.clip(rs(r), 0, 1), rs(cf), rs(cc)
    return {"neutral": n, "ref": r, "conf_fine": cf, "conf_cover": cc, "align_iou": iou, "scale": k}


def _resize_rgba(x, s):
    if abs(s - 1) < 1e-3:
        return x
    H, W = x.shape[:2]
    out = cv2.resize(x, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return np.clip(out, 0, 1)


def _fold_fraction(P, a):
    """Share of body pixels where the frame->neutral map folds over (Jacobian determinant <= 0)."""
    dxx, dxy = np.gradient(P.mapx, axis=1), np.gradient(P.mapx, axis=0)
    dyx, dyy = np.gradient(P.mapy, axis=1), np.gradient(P.mapy, axis=0)
    det = dxx * dyy - dxy * dyx
    return float((det[a] <= 0).mean()) if a.any() else 1.0


def _band_delta(Lr, Lb, regs, dist, Hf, sig, bands=None):
    """Per-material masked low-pass difference; with bands, each distance band is low-passed on its own
    (so the rim spreads along the contour, not across it). One fused 7-channel blur per (material, band)."""
    d = np.zeros_like(Lb)
    edges = [0.0, 1e9] if bands is None else [b * Hf for b in bands]
    for wmat, reg in regs:
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            bm = (dist >= lo_) & (dist < hi_)
            r = (reg & bm).astype(np.float32)
            if r.sum() < 30:
                continue
            st = cv2.GaussianBlur(np.dstack([Lr * r[..., None], Lb * r[..., None], r]), (0, 0), sig)
            diff = (st[..., 0:3] - st[..., 3:6]) / np.maximum(st[..., 6:7], 1e-4)
            d += np.where(bm[..., None], wmat[..., None] * diff, 0)
    return d


def _delta_at_scale(q, Lr, Lb, regs, a, Hf, sig, bands=None):
    """Run _band_delta on a q-times smaller copy (the result is smooth) and bring it back."""
    H, W = a.shape
    if q >= 0.99:
        dist = cv2.distanceTransform(a.astype(np.uint8), cv2.DIST_L2, 5)
        return _band_delta(Lr, Lb, regs, dist, Hf, sig, bands)
    size = (max(1, round(W * q)), max(1, round(H * q)))
    rs = lambda x: cv2.resize(x, size, interpolation=cv2.INTER_AREA)
    rsm = lambda m: cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    a_q = rsm(a) > 0.5
    regs_q = [(rs(wm.astype(np.float32)), rsm(rg) > 0.5) for wm, rg in regs]
    dist = cv2.distanceTransform(a_q.astype(np.uint8), cv2.DIST_L2, 5)
    d = _band_delta(rs(Lr), rs(Lb), regs_q, dist, Hf * q, sig * q, bands)
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)


def _crop_box(alpha, margin):
    ys, xs = np.nonzero(alpha > 0.5)
    H, W = alpha.shape
    m = int(round(margin))
    return max(ys.min() - m, 0), min(ys.max() + m + 1, H), max(xs.min() - m, 0), min(xs.max() + m + 1, W)


def _align(frame_c, pack):
    """Scale the (cropped) frame to the pack's figure height and align the neutral onto it."""
    n = pack["neutral"]
    Hn, H0 = _fig_h(n[..., 3]), _fig_h(frame_c[..., 3])
    Fs = _resize_rgba(frame_c, Hn / H0)
    P = Pairs(n, Fs, regions=False)
    a = Fs[..., 3] > 0.5
    match_raw = edge_confidence(P.neutral_w, Fs, Hn, flat_ok=False)
    info = {"iou": P.iou, "iou_affine": P.iou_affine, "fold": _fold_fraction(P, a), "match": float(match_raw[a].mean()),
            "coverage": 0.0}
    return Fs, P, Hn, info


def _transfer(frame, profile, pack, view=None, gates=None):
    """Returns (transfer_rgba or None, fallback_rgba, info). None when the frame fails the gates."""
    fallback = shading.relight(frame, profile, view)
    info = {"iou": 0.0, "iou_affine": 0.0, "fold": 1.0, "match": 0.0, "coverage": 0.0}
    if _fig_h(frame[..., 3]) < 16:
        return None, fallback, info
    y0, y1, x0, x1 = _crop_box(frame[..., 3], 0.05 * _fig_h(frame[..., 3]))   # work on the figure, not the canvas
    fc = frame[y0:y1, x0:x1]
    Fs, P, Hn, info = _align(fc, pack)
    if not passes(info, gates):                      # gate early: nothing below is needed for a fallback frame
        return None, fallback, info

    a = Fs[..., 3] > 0.5
    R = P.warp(pack["ref"])
    cfine, ccover = P.warp(pack["conf_fine"]), P.warp(pack["conf_cover"])
    valid = (Fs[..., 3] > 0.9) & (P.neutral_w[..., 3] > 0.9) & (R[..., 3] > 0.5)
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    k = int(0.004 * Hn) * 2 + 1
    near = (cv2.dilate(valid.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & a   # fur / hair fringes, not torn areas
    lm = edge_confidence(P.neutral_w, Fs, Hn, flat_ok=True)
    w = (lowpass(ccover * lowpass(lm, a, 0.02 * Hn), valid, 0.01 * Hn) * near).astype(np.float32)
    cf = lowpass(cfine * lm, valid, 0.01 * Hn) * near
    cf_rel = np.clip(cf / np.maximum(w, 1e-4), 0, 1)

    base_s = apply_color(Fs, profile["color"])
    mats = materials(Fs)
    line = cv2.dilate(mats["line"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    Lb, Lr = to_lab(base_s[..., :3]), to_lab(R[..., :3])
    ok = a & valid
    regs = [(mats["skin"], ok & (mats["skin"] > 0.5)), (mats["light"], ok & (mats["light"] > 0.5)),
            (mats["dark"], ok & (mats["dark"] > 0.5) & ~line)]
    dist = cv2.distanceTransform(a.astype(np.uint8), cv2.DIST_L2, 5)
    q_in, q_rim = min(1.0, 180.0 / Hn), 1.0   # interior: ~180 px figure height is plenty; rim: full resolution (thin bands)
    d_in = _delta_at_scale(q_in, Lr, Lb, regs, a, Hn, SIGMA_IN * Hn)
    d_rim = cv2.GaussianBlur(_delta_at_scale(q_rim, Lr, Lb, regs, a, Hn, SIGMA_FINE * Hn, BANDS), (0, 0), 1.0)
    e = np.exp(-dist / (EDGE_FALLOFF * Hn)) * cf_rel
    Dc = d_in + e[..., None] * (d_rim - d_in)

    # back to the frame (D and w are smooth); line art may darken with the scene but is never lightened
    Hh, Ww = frame.shape[:2]
    D = np.zeros((Hh, Ww, 3), np.float32)
    W_ = np.zeros((Hh, Ww), np.float32)
    D[y0:y1, x0:x1] = cv2.resize(Dc, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    W_[y0:y1, x0:x1] = np.clip(cv2.resize(w, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), 0, 1)
    base = apply_color(frame, profile["color"])
    lw = cv2.GaussianBlur(materials(frame)["line"].astype(np.float32), (0, 0), 0.7)
    D[..., 0] = D[..., 0] * (1 - lw) + np.minimum(D[..., 0], 0) * lw
    lab = to_lab(base[..., :3]) + D
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    rgb = from_lab(lab) * W_[..., None] + fallback[..., :3] * (1 - W_[..., None])
    info["coverage"] = float(W_[frame[..., 3] > 0.5].mean())
    return np.dstack([rgb, frame[..., 3]]).astype(np.float32), fallback, info


def gate(frame, pack, gates=None):
    """Alignment + gates only (no rendering). Returns info with 'passed'."""
    info = {"iou": 0.0, "iou_affine": 0.0, "fold": 1.0, "match": 0.0, "coverage": 0.0}
    if _fig_h(frame[..., 3]) >= 16:
        y0, y1, x0, x1 = _crop_box(frame[..., 3], 0.05 * _fig_h(frame[..., 3]))
        info = _align(frame[y0:y1, x0:x1], pack)[3]
    info["passed"] = passes(info, gates)
    return info


def passes(info, gates=None):
    g = dict(GATES, **(gates or {}))
    return (info["fold"] <= g["max_fold"] and info["match"] >= g["min_match"]
            and info["iou_affine"] >= g["min_iou_affine"])


def relight_transfer(frame, profile, pack, view=None, gates=None):
    """One frame: reference look if the frame passes the gates, else wash+rim. Returns (rgba, info)."""
    out, fallback, info = _transfer(frame, profile, pack, view, gates)
    info["passed"] = out is not None and passes(info, gates)
    if not info["passed"]:
        info["coverage"] = 0.0
        return fallback, info
    return out, info
