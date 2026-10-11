"""Synthetic ground-truth renderer #3 + recolours (copied from R_audit synth.py / common.py, our own research code; no art).

  lit(F, scene): ambient + wrapped-Lambert key(s) on a Poisson-inflated silhouette (low-res solve, a different geometry than
      the methods' features) + optional point light with falloff + optional desaturation / L* contrast curve. Bench
      extension: 'keys' = extra directional lights [(rgb, (az, el))]; defaults reproduce the original.
  night_from_ref / darkness_from_ref: the real hand-lit reference re-graded (Purkinje desaturation + dark blue gain).
  recolour: von Kries per-channel multiply in LINEAR RGB on a soft mask; commutes with any per-channel light, so recolouring
      a neutral and its lit reference the same way gives a consistent (neutral, lit) pair.
  warp_frame / downscale: mimic frames and small frames (premultiplied resize).
"""
import numpy as np
import cv2

from scenelight.color import srgb_to_linear, linear_to_srgb, to_lab, from_lab
from scenelight.shading import _poisson

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def lin(rgb):
    return srgb_to_linear(np.clip(rgb, 0, 1))


def srgb(x):
    return np.clip(linear_to_srgb(x), 0, 1).astype(np.float32)


def night_from_ref(R, gain=(0.07, 0.10, 0.22), desat=0.45):
    x = lin(R[..., :3])
    Y = (x * LUM).sum(2, keepdims=True)
    x = ((1 - desat) * x + desat * Y) * np.array(gain, np.float32)
    return np.dstack([srgb(x), R[..., 3]]).astype(np.float32)


def darkness_from_ref(R):
    return night_from_ref(R, gain=(0.018, 0.022, 0.04), desat=0.6)


_NCACHE = {}


def normals(alpha, low_h=300):
    key = (alpha.shape, float(alpha.sum()))
    if key in _NCACHE:
        return _NCACHE[key]
    a = alpha > 0.5
    ys = np.nonzero(a.any(1))[0]
    Hf = ys.max() - ys.min() + 1
    k = min(1.0, low_h / Hf)
    small = cv2.resize(a.astype(np.float32), None, fx=k, fy=k, interpolation=cv2.INTER_AREA) > 0.5
    h = np.sqrt(np.maximum(_poisson(small), 0)) / k
    h = cv2.resize(h.astype(np.float32), (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
    h = cv2.GaussianBlur(h, (0, 0), max(1.0, 0.004 * Hf)) * a
    gy, gx = np.gradient(h)
    N = np.dstack([-gx, -gy, np.ones_like(h)])
    N /= np.linalg.norm(N, axis=2, keepdims=True)
    if len(_NCACHE) > 64:
        _NCACHE.clear()
    _NCACHE[key] = (N, h, Hf)
    return N, h, Hf


def _dir(az, el):
    """az: screen angle in degrees (0 = right, 90 = up), el: elevation towards the viewer (image coords, y down)."""
    t, e = np.radians(az), np.radians(el)
    return np.array([np.cos(t) * np.cos(e), -np.sin(t) * np.cos(e), np.sin(e)], np.float32)


SCENES = {
    # the three R_audit scenes (verbatim)
    "night_lantern": dict(amb=(0.045, 0.06, 0.13), key=(0.20, 0.27, 0.45), key_dir=(140, 15), wrap=0.2,
                          point=(1.5, 0.80, 0.30), point_pos=(0.62, 0.52, 0.25), point_r=0.30, desat=0.25),
    "midday": dict(amb=(0.30, 0.33, 0.38), key=(0.95, 0.93, 0.88), key_dir=(75, 35), wrap=0.0,
                   point=None, desat=0.0, curve=1.25),
    "sunset_like": dict(amb=(0.42, 0.30, 0.24), key=(0.85, 0.50, 0.22), key_dir=(10, 10), wrap=0.3,
                        point=None, desat=0.0),
    # bench additions
    "night_moon": dict(amb=(0.045, 0.06, 0.13), key=(0.20, 0.27, 0.45), key_dir=(140, 15), wrap=0.2,
                       point=None, desat=0.25),
    "two_keys": dict(amb=(0.25, 0.25, 0.28), key=(0.90, 0.62, 0.35), key_dir=(25, 10), wrap=0.1,
                     keys=[((0.30, 0.40, 0.70), (165, 25))], point=None, desat=0.0),
}


def lit(F, scene):
    sc = SCENES[scene] if isinstance(scene, str) else scene
    a = F[..., 3]
    N, h, Hf = normals(a)
    ys, xs = np.nonzero(a > 0.5)
    x0, x1, y0 = xs.min(), xs.max(), ys.min()
    alb = lin(F[..., :3])
    l1 = _dir(*sc["key_dir"])
    w = sc.get("wrap", 0.0)
    key = np.clip(((N @ l1) + w) / (1 + w), 0, 1)[..., None] * np.array(sc["key"], np.float32)
    light = np.array(sc["amb"], np.float32) + key
    for rgb, kd in sc.get("keys", ()):
        light = light + np.clip(((N @ _dir(*kd)) + w) / (1 + w), 0, 1)[..., None] * np.array(rgb, np.float32)
    if sc.get("point") is not None:
        px, py, pz = sc["point_pos"]
        P = np.array([x0 + px * (x1 - x0), y0 + py * Hf, pz * Hf], np.float32)
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
        V = np.dstack([P[0] - xx, P[1] - yy, P[2] - h])
        r = np.linalg.norm(V, axis=2, keepdims=True)
        V /= r
        fall = 1 / (1 + (r / (sc["point_r"] * Hf)) ** 2)
        light = light + np.clip((N * V).sum(2, keepdims=True), 0, 1) * fall * np.array(sc["point"], np.float32)
    x = alb * light
    if sc.get("desat"):
        Y = (x * LUM).sum(2, keepdims=True)
        x = (1 - sc["desat"]) * x + sc["desat"] * Y
    out = srgb(x)
    if sc.get("curve"):
        lab = to_lab(out)
        lab[..., 0] = np.clip(55 + sc["curve"] * (lab[..., 0] - 55), 0, 100)
        out = from_lab(lab)
    return np.dstack([out, a]).astype(np.float32)


def soft(mask, s=1.2):
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), s), 0, 1)


def recolour(img, mask, k):
    """von Kries recolour: per-channel multiply in LINEAR RGB on a (soft) mask."""
    out = img.copy()
    x = lin(img[..., :3])
    w = soft(mask)[..., None]
    out[..., :3] = srgb(x * (1 - w) + x * np.array(k, np.float32) * w)
    return out


def warp_frame(F, s=0.82, rot=3.0, dx=0.0, bend=0.012, ref_alpha=None):
    """Mimic animation frame: scale / rotate about the feet + a smooth horizontal sway."""
    H, W = F.shape[:2]
    ys, xs = np.nonzero((F[..., 3] if ref_alpha is None else ref_alpha) > 0.5)
    cx, cy = (xs.min() + xs.max()) / 2, ys.max()
    Hf = ys.max() - ys.min()
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), rot, s)
    M[0, 2] += dx
    out = cv2.warpAffine(F, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sway = bend * Hf * np.sin((yy - ys.min()) / Hf * np.pi * 1.5)
    out = cv2.remap(out, (xx - sway).astype(np.float32), yy, cv2.INTER_LINEAR, borderValue=0)
    return np.clip(out, 0, 1).astype(np.float32)


def downscale(F, fig_h, like=None):
    """Resize so the figure is fig_h px tall (premultiplied area resize; like = take the scale from this frame)."""
    ys = np.nonzero((F if like is None else like)[..., 3].max(1) > 0.5)[0]
    k = fig_h / (ys.max() - ys.min() + 1)
    H, W = F.shape[:2]
    pm = np.dstack([F[..., :3] * F[..., 3:4], F[..., 3:4]])
    s = cv2.resize(pm, (max(1, round(W * k)), max(1, round(H * k))), interpolation=cv2.INTER_AREA)
    a = np.clip(s[..., 3], 0, 1)
    rgb = np.clip(s[..., :3] / np.maximum(a[..., None], 1e-4), 0, 1)
    return np.dstack([rgb, a]).astype(np.float32), k
