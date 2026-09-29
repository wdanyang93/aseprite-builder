"""Place a relit sprite on a background: scale, contact shadow, light wrap, in-scene brightness."""
import cv2
import numpy as np

from .color import linear_to_srgb, srgb_to_linear


def crop_to_alpha(rgb, alpha, pad=2):
    ys, xs = np.where(alpha > 0.02)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, alpha.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, alpha.shape[1])
    return rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def resize_premult(rgb, alpha, height):
    scale = height / alpha.shape[0]
    size = (max(1, round(alpha.shape[1] * scale)), max(1, round(alpha.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    pm = cv2.resize(rgb * alpha[..., None], size, interpolation=interp)
    a = np.clip(cv2.resize(alpha, size, interpolation=interp), 0, 1)
    return np.clip(pm / np.maximum(a[..., None], 1e-4), 0, 1), a


def composite(bg, rgba, cx, foot_y, height, shadow=0.55, wrap=0.35, dim=0.95):
    """bg: float RGB. rgba: relit sprite. (cx, foot_y) = where the feet touch the ground, height = sprite px."""
    rgb, alpha = crop_to_alpha(rgba[..., :3], rgba[..., 3])
    rgb, alpha = resize_premult(rgb, alpha, height)
    h, w = alpha.shape
    x0, y0 = int(round(cx - w / 2)), int(round(foot_y - h))
    out = bg.copy()
    H, W = bg.shape[:2]
    if shadow > 0:
        # soft ellipse under the lowest opaque pixels (the feet)
        rows = np.where(alpha.max(1) > 0.5)[0]
        foot = alpha[rows[-1] - max(2, h // 25): rows[-1] + 1]
        cols = np.where(foot.max(0) > 0.5)[0]
        fw = (cols[-1] - cols[0]) * 0.65 + h * 0.08
        fcx = x0 + (cols[0] + cols[-1]) / 2
        sh = np.zeros((H, W), np.float32)
        cv2.ellipse(sh, (int(fcx), int(y0 + rows[-1])), (int(fw / 2 + 1), int(max(2, h * 0.018))), 0, 0, 360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, h * 0.012))
        dark = np.array([0.18, 0.10, 0.05])
        out = out * (1 - sh[..., None] * shadow) + dark * sh[..., None] * shadow * 0.3
    bx0, by0, bx1, by1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    spr = rgb[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
    a = alpha[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
    base = out[by0:by1, bx0:bx1]
    spr = linear_to_srgb(srgb_to_linear(spr) * dim ** 2.2)
    if wrap > 0:
        # blurred background bleeds onto the sprite's edge so it doesn't look cut out
        r = max(1.0, h / 90.0)
        bg_blur = cv2.GaussianBlur(bg, (0, 0), r * 2)[by0:by1, bx0:bx1]
        edge = np.clip(1 - cv2.GaussianBlur(a, (0, 0), r), 0, 1) * a
        spr = linear_to_srgb(srgb_to_linear(spr) + srgb_to_linear(bg_blur) * edge[..., None] * wrap)
    out[by0:by1, bx0:bx1] = base * (1 - a[..., None]) + spr * a[..., None]
    return out
