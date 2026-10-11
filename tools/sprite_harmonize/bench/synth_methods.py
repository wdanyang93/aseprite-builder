"""Synthetic ground-truth renderer #2 (copied from R_methods synth.py, our own research code; no art).

Poisson-inflation height field normals (low-res solve, 0.4 % H blur), wrapped-Lambert moon key, blue sky ambient, warm point
lantern (screen-space position relative to the figure box, 1 / (1 + (r / 0.35 H)^2) falloff), then Purkinje desaturation
(25 % toward luminance) and a faint blue haze on the darks. Bench extension: every constant is a parameter; the defaults
reproduce the original night(); lantern=None switches the lantern off (moon only).
"""
import numpy as np
import cv2

from scenelight.shading import _poisson
from scenelight.color import srgb_to_linear, linear_to_srgb


def fig_h(alpha):
    ys = np.nonzero(alpha.max(1) > 0.5)[0]
    return float(ys.max() - ys.min() + 1)


def normals(alpha, low_h=360):
    a = alpha > 0.5
    Hf = fig_h(alpha)
    k = min(1.0, low_h / Hf)
    small = cv2.resize(a.astype(np.float32), None, fx=k, fy=k, interpolation=cv2.INTER_AREA) > 0.5
    u = _poisson(small)
    h = np.sqrt(np.maximum(u, 0)) / k                        # px units at full res
    h = cv2.resize(h.astype(np.float32), (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
    h = cv2.GaussianBlur(h, (0, 0), 0.004 * Hf) * a
    gy, gx = np.gradient(h)
    N = np.dstack([-gx, -gy, np.ones_like(h)])
    N /= np.linalg.norm(N, axis=2, keepdims=True)
    return N, h


def night(x, moon_deg=135, moon_el=30, lantern=(1.15, 0.62, 0.25), amb=(0.05, 0.07, 0.15), moon_rgb=(0.32, 0.40, 0.62),
          wrap=0.25, lantern_rgb=(1.6, 0.85, 0.32), lantern_r=0.35, desat=0.25, haze=(0.004, 0.006, 0.014)):
    a = x[..., 3]
    N, h = normals(a)
    ys, xs = np.nonzero(a > 0.5)
    Hf = fig_h(a); x0, x1, y0 = xs.min(), xs.max(), ys.min()
    alb = srgb_to_linear(x[..., :3])
    amb = np.array(amb)
    th, el = np.radians(moon_deg), np.radians(moon_el)
    l1 = np.array([np.cos(th) * np.cos(el), -np.sin(th) * np.cos(el), np.sin(el)])   # image coords: y DOWN
    moon = np.clip(((N @ l1) + wrap) / (1 + wrap), 0, 1)[..., None] * np.array(moon_rgb)
    light = amb + moon
    if lantern is not None:
        P = np.array([x0 + lantern[0] * (x1 - x0), y0 + lantern[1] * Hf, lantern[2] * Hf])
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
        V = np.dstack([P[0] - xx, P[1] - yy, P[2] - h])
        r = np.linalg.norm(V, axis=2, keepdims=True); V /= r
        fall = 1 / (1 + (r / (lantern_r * Hf)) ** 2)
        light = light + (np.clip((N * V).sum(2, keepdims=True), 0, 1) * fall) * np.array(lantern_rgb)
    lin = alb * light
    Y = (lin * np.array([0.2126, 0.7152, 0.0722])).sum(2, keepdims=True)
    lin = (1 - desat) * lin + desat * Y + np.array(haze) * (1 - np.clip(Y * 8, 0, 1))
    return np.dstack([np.clip(linear_to_srgb(lin), 0, 1), a]).astype(np.float32)
