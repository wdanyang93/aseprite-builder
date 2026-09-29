"""Image I/O, colour-space helpers, material weights and the per-material colour base."""
import cv2
import numpy as np
from PIL import Image


# ------------------------------------------------------------------ io
def load_rgba(path):
    """float32 HxWx4 in [0,1], straight alpha, sRGB."""
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def save_rgba(path, x):
    Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(path)


def save_rgb(path, x):
    Image.fromarray((np.clip(x[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").save(path)


# ----------------------------------------------------------- colour space
def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def to_lab(rgb):
    """CIE Lab, L* in 0..100 (OpenCV float convention)."""
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def from_lab(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)


def lowpass(x, m, sigma):
    """Masked normalised Gaussian blur: values outside m don't bleed in."""
    m = m.astype(np.float32)
    num = cv2.GaussianBlur(x * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def figure_height(alpha):
    ys = np.nonzero(alpha.max(1) > 0.5)[0]
    return float(ys.max() - ys.min() + 1) if len(ys) else float(alpha.shape[0])


# -------------------------------------------------------------- materials
def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def materials(rgba):
    """Soft material weights of a neutral-lit frame, decided per pixel from colour only
    (no part names): skin (chromatic), light (white cloth / hair / fur), dark (dark cloth, line art).
    Returns dict of HxW weights summing to 1, plus boolean 'line'."""
    lab = to_lab(rgba[..., :3])
    L, C = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    dark = 1 - _smooth(44, 54, L)
    skin = (1 - dark) * _smooth(11, 17, C)
    light = 1 - dark - skin
    return {"skin": skin, "light": light, "dark": dark, "line": L < 38}


# ------------------------------------------------------------- colour base
def fit_color(pairs):
    """Per material and Lab channel mean/std matching between the aligned neutral and the reference.
    (Moment matching keeps shading contrast; least squares on imperfectly aligned pixels would shrink it.)
    Dark cloth gets a per-channel multiply so brown never turns purple."""
    nl, rl = to_lab(pairs.neutral_w[..., :3]), to_lab(pairs.ref[..., :3])
    prof = {}
    body = pairs.reliable & (pairs.mat["dark"] < 0.2)
    for mat in ("skin", "light"):
        m = pairs.reliable & (pairs.mat[mat] > 0.8)
        if m.sum() < 200:  # material absent in this drawing: fall back to all non-dark pixels
            m = body
        a, b = nl[m], rl[m]
        gain = np.clip(b.std(0) / np.maximum(a.std(0), 1e-3), 0.3, 3.0)
        W = np.zeros((4, 3))
        W[0, 0], W[1, 1], W[2, 2] = gain
        W[3] = b.mean(0) - a.mean(0) * gain
        prof[mat] = W.tolist()
    m = pairs.reliable & (pairs.mat["dark"] > 0.8)
    if m.sum() < 200:
        m = body
    ratio = np.median(pairs.ref[..., :3][m], 0) / np.maximum(np.median(pairs.neutral_w[..., :3][m], 0), 1e-3)
    prof["dark_mul"] = np.clip(ratio, 0.2, 3.0).tolist()
    return prof


def apply_color(rgba, prof, mats=None):
    mats = mats or materials(rgba)
    lab = to_lab(rgba[..., :3])
    X = np.concatenate([lab, np.ones(lab.shape[:2] + (1,), np.float32)], -1)
    out = np.zeros(lab.shape[:2] + (3,), np.float32)
    for mat in ("skin", "light"):
        out += mats[mat][..., None] * from_lab(X @ np.array(prof[mat], np.float32))
    out += mats["dark"][..., None] * np.clip(rgba[..., :3] * np.array(prof["dark_mul"], np.float32), 0, 1)
    return np.dstack([out, rgba[..., 3]]).astype(np.float32)


def defringe(rgb, alpha, radius=4):
    """Replace colours of semi-transparent edge pixels with interior colour (removes white halos)."""
    solid = (alpha > 0.95).astype(np.float32)
    num = cv2.GaussianBlur(rgb * solid[..., None], (0, 0), radius)
    den = cv2.GaussianBlur(solid, (0, 0), radius)[..., None]
    fill = num / np.maximum(den, 1e-4)
    k = radius * 2 + 1
    for _ in range(3):
        miss = den[..., 0] < 1e-3
        if not miss.any():
            break
        fill = np.where(miss[..., None], cv2.dilate(fill, np.ones((k, k), np.uint8)), fill)
        den = np.where(miss[..., None], 1.0, den)
    w = np.clip((alpha - 0.5) / 0.45, 0, 1)[..., None]
    return rgb * w + fill * (1 - w)
