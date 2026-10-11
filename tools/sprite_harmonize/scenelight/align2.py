"""Brightness-robust alignment of a neutral drawing onto a lit reference (same pose, possibly a repaint).

Same result interface as scenelight.align.Pairs (mapx / mapy, warp(), neutral_w, iou, iou_affine, reliable, mat,
regions, Hf), so it can replace it. What changes:
  1. the optical flow (DIS) runs on contrast-normalised STRUCTURE images instead of the plain grey image. The plain grey
     image of a night / darkness reference has almost no contrast, so the flow dragged the neutral's silhouette onto
     interior edges (flow IoU 0.875 night / 0.509 darkness vs 0.971 sunset on the same drawing). Structure images:
       rank  = L* rank-normalised over the figure (invariant to any monotone tone curve)
       grad  = gradient magnitude of log luminance (invariant to smooth multiplicative light), self-normalised (default:
               best interior structure agreement, fold and known-warp error of the candidates on the real pairs)
       mix   = mean of rank and grad (rank carries the large shapes for the coarse flow levels, grad the detail)
       clahe = CLAHE on the figure's L*;  gray = the old input (kept for comparison)
     The silhouette is always a step from 0 to at least FLOOR in both images, so it lines up whatever the light.
  2. flow from both sides: from the reference grid (template = output grid, no inversion) and from the neutral side
     (template = the moving image, flow inverted). Both are needed for the forward-backward check; the map uses the one
     with the higher silhouette IoU ('auto'), or the side asked for (flow_from='ref' / 'neutral'; the names are the
     constructor's arguments, so in transfer.build_pack's swapped call Pairs(ref, neutral) the real neutral side is
     flow_from='ref'). The neutral side mattered with the plain grey input (darkness: IoU 0.95 with the neutral as
     template vs 0.50 with the dark reference); on structure images both sides work and the output-grid flow is
     usually the better one (no inversion; lower fold).
  3. a flow that lowers the silhouette IoU below the affine-only result is rejected (affine only).
  4. per-pixel confidence `conf` (0..1, reference grid): forward-backward consistency x local structure agreement
     (log-luminance edge NCC, same window / thresholds as transfer.edge_confidence). The forward-backward penalty is
     scaled by local texture (flat skin: both flows undetermined, misplacement harmless). `reliable` (fitting pixels)
     also requires conf >= min_conf (see MIN_CONF for the measured trade-off).
All length constants are fractions of the figure height H.
"""
import time

import cv2
import numpy as np

from . import align as _align
from .align import _gray, _similarity
from .color import figure_height, lowpass, materials, srgb_to_linear, to_lab

STRUCTS = ("grad", "mix", "rank", "clahe", "gray")
STRUCT = "grad"
FLOOR = 64            # structure value of the figure's darkest tone (background = 0)
FLOW_SIGMA = 0.0036   # flow smoothing (fraction of H; align.Pairs used 5 px at H ~1400)
FB_TAU = FLOW_SIGMA   # forward-backward error scale of the confidence (fraction of H): the flow's own smoothing
WIN = 0.04            # structure-agreement window (fraction of H), as transfer.edge_confidence
MIN_CONF = 0.25       # fitting pixels (`reliable`) need at least this confidence. Drop-in scene fit (old aligner ->
                      # min_conf 0 -> 0.25), colour error: Lyn R sunset 6.75 -> 6.49 -> 6.13, Lyn L 7.88 -> 8.07 -> 8.07,
                      # Lyn R night 2.56 -> 2.45 -> 2.28, P01 night 6.13 -> 5.77 -> 5.77, P01 lantern 12.61 -> 13.28 ->
                      # 12.40. Caveat: on a heavily redrawn pair the dropped pixels are the relatively brighter
                      # (rim-lit, repainted) ones (ref-neutral -10.6 vs kept -14.3 L*; fitted mean L* 66.0 -> 65.2, ref
                      # 66.1): rim / edge-light fits should weight by conf_fb rather than hard-filter on conf
POSE_IOU = 0.85       # below: "neutral and reference line up poorly" (scene light rig needs >= 0.85)
PACK_IOU = 0.95       # a direction pack needs >= 0.95 (+ the confidence map)
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ------------------------------------------------------------------ structure images
def _loglum(rgba, m):
    """log luminance with an epsilon relative to the figure's own median: a global gain is an exact offset."""
    Y = (srgb_to_linear(np.clip(rgba[..., :3], 0, 1)) * LUM).sum(-1)
    eps = 0.05 * float(np.median(Y[m])) + 1e-5 if m.any() else 1e-3
    return np.log(Y + eps).astype(np.float32)


def _edges(rgba, H):
    """Gradient magnitude of log luminance (light-gain free), lightly smoothed. Colour left in transparent pixels is
    ignored: outside the figure the log luminance is a smooth extension of the figure's edge values."""
    m = rgba[..., 3] > 0.5
    lg = _loglum(rgba, m)
    if m.any():
        sig = max(2.0, 0.003 * H)
        mf = m.astype(np.float32)
        den = cv2.GaussianBlur(mf, (0, 0), sig)
        num = cv2.GaussianBlur(lg * mf, (0, 0), sig)
        fill = (num + 0.02 * float(np.median(lg[m]))) / (den + 0.02)      # fades smoothly to the median, no step
        lg = np.where(m, lg, fill).astype(np.float32)
    lg = cv2.GaussianBlur(lg, (0, 0), max(0.7, 0.0007 * H))
    return np.hypot(cv2.Sobel(lg, cv2.CV_32F, 1, 0), cv2.Sobel(lg, cv2.CV_32F, 0, 1))


def structure(rgba, kind=STRUCT, H=None):
    """uint8 structure image for the optical flow: background 0, figure FLOOR..255 (times alpha on the edge ring)."""
    if kind == "gray":
        return _gray(rgba)
    if kind not in STRUCTS:
        raise ValueError(f"structure kind must be one of {STRUCTS} (got {kind!r})")
    a = rgba[..., 3]
    m = a > 0.5
    if not m.any():
        return np.zeros(a.shape, np.uint8)
    H = H or figure_height(a)
    if kind == "mix":
        r, g = structure(rgba, "rank", H), structure(rgba, "grad", H)
        return ((r.astype(np.float32) + g) * 0.5 + 0.5).astype(np.uint8)
    if kind == "grad":
        g = _edges(rgba, H)
        u = g / (g + 2 * float(np.median(g[m])) + 1e-6)
    else:
        L = to_lab(np.clip(rgba[..., :3], 0, 1))[..., 0]
        if kind == "rank":
            qs = np.quantile(L[m], np.linspace(0, 1, 257))
            qs = qs + np.arange(257) * 1e-6 * max(float(qs[-1] - qs[0]), 1e-3)   # strictly increasing
            u = np.interp(L, qs, np.linspace(0, 1, 257))
        else:   # clahe
            lo, hi = np.percentile(L[m], [1, 99])
            u8 = (np.clip((L - lo) / max(hi - lo, 1e-3), 0, 1) * 255 * m).astype(np.uint8)
            nt = int(np.clip(round(max(a.shape) / (0.12 * H)), 2, 64))
            u = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(nt, nt)).apply(u8).astype(np.float32) / 255
    s = a * (FLOOR + (255 - FLOOR) * u)
    return np.clip(s + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ flow helpers
def _dis(I0, I1, H):
    """DIS flow on I0's grid (I0(x) ~ I1(x + f(x))), smoothed."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return cv2.GaussianBlur(dis.calc(I0, I1, None), (0, 0), max(1.0, FLOW_SIGMA * H))


def _grid(shape):
    H, W = shape[:2]
    return np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def invert_flow(f, iters=6):
    """Displacement g with y = x + g(x) solving y + f(y) = x (fixed point; fine for smooth flows)."""
    gx, gy = _grid(f.shape)
    yx, yy = gx.copy(), gy.copy()
    for _ in range(iters):
        fy = cv2.remap(f, yx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        yx, yy = gx - fy[..., 0], gy - fy[..., 1]
    return np.dstack([yx - gx, yy - gy]).astype(np.float32)


def _maps(M, flow):
    gx, gy = _grid(flow.shape)
    qx, qy = gx + flow[..., 0], gy + flow[..., 1]
    Mi = cv2.invertAffineTransform(M)
    return ((Mi[0, 0] * qx + Mi[0, 1] * qy + Mi[0, 2]).astype(np.float32),
            (Mi[1, 0] * qx + Mi[1, 1] * qy + Mi[1, 2]).astype(np.float32))


def _iou(a, b):
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum() / max((a | b).sum(), 1))


def fold_fraction(mapx, mapy, mask):
    """Share of mask pixels where the map folds over (Jacobian determinant <= 0)."""
    det = np.gradient(mapx, axis=1) * np.gradient(mapy, axis=0) - np.gradient(mapx, axis=0) * np.gradient(mapy, axis=1)
    return float((det[mask] <= 0).mean()) if mask.any() else 1.0


def fb_error(f_ref, f_neu):
    """Forward-backward error on the reference grid: f_ref (ref grid -> moving) followed by f_neu (moving -> ref)."""
    gx, gy = _grid(f_ref.shape)
    back = cv2.remap(f_neu, gx + f_ref[..., 0], gy + f_ref[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.hypot(f_ref[..., 0] + back[..., 0], f_ref[..., 1] + back[..., 1])


def edge_agreement(a_rgba, b_rgba, H, win=WIN, flat_ok=True):
    """Local NCC (window win*H) of the log-luminance edge maps of two aligned drawings, -1..1 (brightness-robust
    counterpart of transfer.edge_confidence). flat_ok: windows flat in both drawings count as agreeing (1)."""
    ea, eb = _edges(a_rgba, H), _edges(b_rgba, H)
    w = int(win * H) | 1
    mu = lambda x: cv2.blur(x, (w, w))
    va, vb = np.maximum(mu(ea * ea) - mu(ea) ** 2, 0), np.maximum(mu(eb * eb) - mu(eb) ** 2, 0)
    ncc = (mu(ea * eb) - mu(ea) * mu(eb)) / np.sqrt(np.maximum(va * vb, 1e-12))
    if flat_ok:
        body = (a_rgba[..., 3] > 0.5) & (b_rgba[..., 3] > 0.5)
        if body.any():
            t = 0.02 * float(np.median(np.concatenate([va[body], vb[body]])))
            ncc = np.where((va < t) & (vb < t), 1.0, ncc)
    return ncc.astype(np.float32)


def structure_match(P, lo=0.25, hi=0.6):
    """Brightness-robust drawing-structure match of an aligned pair (mean over the body), on the same 0..1 scale as
    transfer's gate (edge_confidence with flat_ok=False)."""
    a = (P.ref[..., 3] > 0.5) & (P.neutral_w[..., 3] > 0.5)
    if not a.any():
        return 0.0
    ncc = edge_agreement(P.neutral_w, P.ref, P.Hf, win=0.04, flat_ok=False)
    c = np.clip((ncc - lo) / (hi - lo), 0, 1)
    return float(lowpass(c, a, max(1.0, 0.02 * P.Hf))[a].mean())


# ------------------------------------------------------------------ Pairs
class Pairs(_align.Pairs):
    """Neutral frame aligned into the reference frame + label-free fitting regions (drop-in for align.Pairs).

    Extra attributes: M (similarity), struct, flow_from (side whose flow is used: 'ref', 'neutral' or 'affine'),
    iou_flow (IoU per flow candidate), iou_affine_full, flow_used, conf / conf_fb / conf_struct (reference grid),
    fb_err (px), fold (of the map used), fold_flow (of the best flow even when it was rejected: a gate that relies on
    folds should read this one), info (numbers + timings)."""

    def __init__(self, neutral, ref, cell=0.12, regions=True, struct=STRUCT, flow_from="auto", confidence=True,
                 min_conf=MIN_CONF):
        if flow_from not in ("auto", "ref", "neutral", "affine"):
            raise ValueError(f"flow_from must be auto, ref, neutral or affine (got {flow_from!r})")
        t0 = time.time()
        self.neutral, self.ref = neutral, ref
        H, W = ref.shape[:2]
        self.shape = (H, W)
        self.Hf = figure_height(ref[..., 3])
        self.struct = struct
        M, self.iou_affine = _similarity(neutral[..., 3], ref[..., 3])
        self.M = M
        t1 = time.time()
        wa = cv2.warpAffine(neutral, M, (W, H))
        zero = np.zeros((H, W, 2), np.float32)
        cands = {"affine": zero}
        f_ref = f_neu = None
        if flow_from != "affine":
            Sa, Sr = structure(wa, struct, self.Hf), structure(ref, struct, self.Hf)
            if flow_from in ("auto", "ref") or confidence:
                f_ref = _dis(Sr, Sa, self.Hf)                       # template: the reference grid
            if flow_from in ("auto", "neutral") or confidence:
                f_neu = _dis(Sa, Sr, self.Hf)                       # template: the neutral (moving) side
            if flow_from in ("auto", "ref"):
                cands["ref"] = f_ref
            if flow_from in ("auto", "neutral"):
                cands["neutral"] = invert_flow(f_neu)
        t2 = time.time()
        ra = ref[..., 3]
        ious = {}
        for k, fl in cands.items():
            mx, my = _maps(M, fl)
            ious[k] = _iou(cv2.remap(neutral[..., 3], mx, my, cv2.INTER_LINEAR, borderValue=0), ra)
        self.iou_affine_full = ious["affine"]
        self.iou_flow = {k: round(v, 4) for k, v in ious.items() if k != "affine"}
        best = max((k for k in cands if k != "affine"), key=lambda k: ious[k], default="affine")
        self.fold_flow = fold_fraction(*_maps(M, cands[best]), ra > 0.5) if best != "affine" else 0.0
        if best != "affine" and ious[best] < ious["affine"]:       # flow would lower the IoU: affine only
            best = "affine"
        self.flow_from = best
        self.flow_used = best != "affine"
        self.flow = cands[best]
        self.mapx, self.mapy = _maps(M, self.flow)
        self.neutral_w = self.warp(neutral)
        a, r = self.neutral_w[..., 3] > 0.5, ra > 0.5
        self.iou = float((a & r).sum() / max((a | r).sum(), 1))
        self.fold = fold_fraction(self.mapx, self.mapy, r)
        t3 = time.time()

        # confidence: forward-backward consistency x local structure agreement, on the reference grid
        body = a & r
        ncc = edge_agreement(self.neutral_w, ref, self.Hf)
        cs = np.clip((ncc - 0.25) / 0.35, 0, 1)
        self.conf_struct = (lowpass(cs, body, max(1.0, 0.02 * self.Hf)) * body).astype(np.float32)
        if f_ref is not None and f_neu is not None and self.flow_used:
            self.fb_err = fb_error(f_ref, f_neu).astype(np.float32)
            tau = max(1.0, FB_TAU * self.Hf)
            # flat, textureless areas: the two flows are undetermined there and disagree, but a misplacement there is
            # harmless for colour; the forward-backward penalty applies in proportion to local texture
            e = 0.5 * (_edges(self.neutral_w, self.Hf) + _edges(ref, self.Hf))
            w = int(0.01 * self.Hf) | 1
            tex = cv2.blur(e, (w, w))
            wt = np.clip(tex / (float(np.median(tex[body])) + 1e-6), 0, 1) if body.any() else np.ones_like(tex)
            pen = 1 - np.exp(-(self.fb_err / tau) ** 2)
            self.conf_fb = ((1 - wt * pen) * body).astype(np.float32)
        else:
            self.fb_err = None
            self.conf_fb = body.astype(np.float32)
        c = self.conf_fb * self.conf_struct
        self.conf = (lowpass(c, body, max(1.0, 0.005 * self.Hf)) * body).astype(np.float32)

        both = (self.neutral_w[..., 3] > 0.9) & (ref[..., 3] > 0.9)
        self.reliable = cv2.erode(both.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.mat = materials(self.neutral_w)                       # line decision on the neutral (original) basis
        line = cv2.dilate(self.mat["line"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.reliable &= ~line & (self.conf >= min_conf)
        t4 = time.time()
        if regions:
            self._make_regions(cell)
        t5 = time.time()
        fb = self.fb_err[body] if self.fb_err is not None and body.any() else np.zeros(1)
        self.info = {
            "iou": round(self.iou, 4), "iou_affine": round(float(self.iou_affine), 4),
            "iou_affine_full": round(self.iou_affine_full, 4), "iou_flow": self.iou_flow, "flow_from": best,
            "struct": struct, "fold": round(self.fold, 4), "fold_flow": round(self.fold_flow, 4), "fb_median_px": round(float(np.median(fb)), 2),
            "fb_p90_px": round(float(np.percentile(fb, 90)), 2), "conf_mean": round(float(self.conf[body].mean()), 3) if body.any() else 0.0,
            "reliable_px": int(self.reliable.sum()), "pose_ok": self.iou >= POSE_IOU,
            "use": "pack" if self.iou >= PACK_IOU else ("scene" if self.iou >= POSE_IOU else "none"),
            "t": {"similarity": round(t1 - t0, 2), "flow": round(t2 - t1, 2), "maps": round(t3 - t2, 2),
                  "conf": round(t4 - t3, 2), "regions": round(t5 - t4, 2), "total": round(t5 - t0, 2)}}

    def _make_regions(self, cell):
        """As align.Pairs, but the local edge agreement uses log-luminance edges (brightness-robust) and the
        flow confidence enters through `reliable`."""
        ra = self.ref[..., 3]
        x0, y0, x1, y1 = _align._bbox(ra)
        Hf = float(y1 - y0 + 1)
        self.Hf = Hf
        c = max(8, int(round(cell * Hf)))
        en, er = _edges(self.neutral_w, Hf), _edges(self.ref, Hf)
        min_px = max(150, int((0.02 * Hf) ** 2))
        self.regions = []
        for yy in range(y0, y1 + 1, c):
            for xx in range(x0, x1 + 1, c):
                cellm = np.zeros(ra.shape, bool)
                cellm[yy:yy + c, xx:xx + c] = True
                inside = cellm & (ra > 0.5) & (self.neutral_w[..., 3] > 0.5)
                if inside.sum() < min_px:
                    continue
                a, b = en[inside], er[inside]
                ncc = float(((a - a.mean()) * (b - b.mean())).mean() / max(a.std() * b.std(), 1e-6))
                w = float(np.clip((ncc - 0.2) / 0.4, 0, 1))
                if w <= 0:
                    continue
                for mat in ("skin", "light"):
                    region = cellm & (self.mat[mat] > 0.5) & (ra > 0.5)
                    m = region & self.reliable & (self.mat[mat] > 0.7)
                    if m.sum() < min_px:
                        continue
                    self.regions.append({"name": f"{yy}_{xx}_{mat}", "mat": mat, "region": region, "mask": m, "w": w})


def overlay(P, mode="diff"):
    """Check images (float RGB, reference grid). 'diff': warped neutral L* vs reference L*, each rank-normalised
    (so a dark reference is comparable) -> red = neutral only, cyan = reference only. 'blend': 50/50 of the
    rank-normalised structure. 'conf': confidence map over the reference."""
    a = (P.ref[..., 3] > 0.5) | (P.neutral_w[..., 3] > 0.5)
    sn = structure(P.neutral_w, "rank", P.Hf).astype(np.float32) / 255
    sr = structure(P.ref, "rank", P.Hf).astype(np.float32) / 255
    if mode == "blend":
        g = 0.5 * (sn + sr)
        return np.dstack([g, g, g])
    if mode == "conf":
        g = sr * 0.6
        return np.dstack([g + 0.4 * (1 - P.conf) * a, g + 0.4 * P.conf, g])
    en, er = _edges(P.neutral_w, P.Hf), _edges(P.ref, P.Hf)
    norm = lambda e: np.clip(e / (np.percentile(e[a], 95) + 1e-6), 0, 1) if a.any() else e
    en, er = norm(en), norm(er)
    return np.dstack([en, er, er]) * 0.85 + 0.15 * np.dstack([sr, sr, sr])
