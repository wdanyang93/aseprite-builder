"""Synthetic ground-truth renderer #1 (copied from R_practice synth.py, our own research code; no art).

Pseudo-3D normals from a distance-transform dome (limb radius 3.5 % H), Lambert + wrap key, ambient, optional point light
(1 / (1 + (r / r0)^2) falloff), then a Lab grade (value compression L_lo..L_hi, Purkinje darkening of reds, saturation,
a*/b* shift). Bench extension (defaults reproduce the original exactly): an optional second directional light 'key2'.
Coordinates of this renderer: normals x right, y UP, z to the viewer; az = screen angle (0 = right, 90 = up).
"""
import numpy as np
import cv2

from scenelight.color import srgb_to_linear, linear_to_srgb, to_lab, materials


def fig_box(alpha):
    ys, xs = np.nonzero(alpha > 0.5)
    return xs.min(), ys.min(), xs.max(), ys.max()


def normals3d(alpha):
    a = (alpha > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(a, cv2.DIST_L2, 5)
    H = fig_box(alpha)[3] - fig_box(alpha)[1]
    R = 0.035 * H                                   # limb radius scale
    h = np.sqrt(np.clip(2 * R * np.minimum(d, R) - np.minimum(d, R) ** 2, 0, None))
    h = cv2.GaussianBlur(h.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8
    n = np.dstack([-gx, gy, np.ones_like(h)])       # x right, y UP, z to viewer
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def lvec(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


SCENES = {
    # the three R_practice scenes (verbatim)
    'midday': dict(amb=[0.62, 0.68, 0.80], key=[0.95, 0.92, 0.85], az=95, el=55, wrap=0.15,
                   grade=dict(L_lo=0, L_hi=100, sat=1.05, a_shift=0, b_shift=2, purkinje=0)),
    'night_moon': dict(amb=[0.10, 0.13, 0.22], key=[0.45, 0.55, 0.80], az=140, el=-25, wrap=0.05,
                       grade=dict(L_lo=6, L_hi=62, sat=0.45, a_shift=0, b_shift=-10, purkinje=0.5)),
    'night_torch': dict(amb=[0.10, 0.13, 0.22], key=[0.45, 0.55, 0.80], az=140, el=-25, wrap=0.05,
                        point=dict(pos=[0.42, -0.15, 0.25], col=[2.2, 1.1, 0.35], r0=0.18),
                        grade=dict(L_lo=6, L_hi=70, sat=0.6, a_shift=0, b_shift=-6, purkinje=0.4)),
    # bench additions (same renderer, new numbers)
    'sunset': dict(amb=[0.55, 0.40, 0.36], key=[1.05, 0.62, 0.30], az=40, el=5, wrap=0.2,
                   grade=dict(L_lo=4, L_hi=96, sat=1.1, a_shift=2, b_shift=6, purkinje=0)),
    'darkness': dict(amb=[0.035, 0.045, 0.08], key=[0.12, 0.15, 0.24], az=120, el=10, wrap=0.1,
                     grade=dict(L_lo=2, L_hi=30, sat=0.35, a_shift=0, b_shift=-6, purkinje=0.6)),
    'two_keys': dict(amb=[0.30, 0.30, 0.34], key=[0.95, 0.70, 0.40], az=30, el=15, wrap=0.1,
                     key2=dict(col=[0.35, 0.45, 0.75], az=160, el=20, wrap=0.0),
                     grade=dict(L_lo=0, L_hi=100, sat=1.0, a_shift=0, b_shift=0, purkinje=0)),
}


def light_scene(N_rgba, sc):
    Nl = srgb_to_linear(np.clip(N_rgba[..., :3], 0, 1))
    alpha = N_rgba[..., 3]
    n = normals3d(alpha)
    L = lvec(sc['az'], sc['el'])
    ndl = n @ L
    w = sc['wrap']
    lam = np.clip((ndl + w) / (1 + w), 0, None)
    E = np.array(sc['amb'])[None, None, :] + lam[..., None] * np.array(sc['key'])[None, None, :]
    if sc.get('key2'):
        k2 = sc['key2']
        lam2 = np.clip(((n @ lvec(k2['az'], k2['el'])) + k2['wrap']) / (1 + k2['wrap']), 0, None)
        E = E + lam2[..., None] * np.array(k2['col'])[None, None, :]
    if 'point' in sc:
        x0, y0, x1, y1 = fig_box(alpha); Hf = y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        yy, xx = np.mgrid[0:alpha.shape[0], 0:alpha.shape[1]].astype(np.float32)
        P = sc['point']
        px, py, pz = cx + P['pos'][0] * Hf, cy - P['pos'][1] * Hf, P['pos'][2] * Hf
        v = np.dstack([px - xx, -(py - yy), np.full_like(xx, pz)])     # to the light, y up
        r = np.linalg.norm(v, axis=-1)
        fall = 1.0 / (1.0 + (r / (P['r0'] * Hf)) ** 2)
        lamp = np.clip((n * (v / r[..., None])).sum(-1), 0, None)
        E = E + (fall * lamp)[..., None] * np.array(P['col'])[None, None, :]
    out = Nl * E
    rgb = linear_to_srgb(np.clip(out, 0, 1))
    lab = to_lab(rgb)
    g = sc['grade']
    lab[..., 0] = g['L_lo'] + (g['L_hi'] - g['L_lo']) * lab[..., 0] / 100.0
    if g['purkinje']:                     # reds lose lightness and chroma relative to blues in the dark
        red = np.clip(lab[..., 1] / 40.0, 0, 1)
        lab[..., 0] -= g['purkinje'] * 10 * red * (1 - lab[..., 0] / 100.0)
    lab[..., 1] = lab[..., 1] * g['sat'] + g['a_shift']
    lab[..., 2] = lab[..., 2] * g['sat'] + g['b_shift']
    rgb = np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)
    return np.dstack([rgb, alpha]).astype(np.float32)


def misalign(ref, dx=1.5, dy=-1.0, blur=0.7):
    """R_practice: real references are warped redraws -> 1.5 px shift + 0.7 px blur of the synthetic reference."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]))
    out[..., :3] = cv2.GaussianBlur(out[..., :3], (0, 0), blur)
    return out


def recolour_swap(n):
    """R_practice swap_test.recolour: white cloth / hair -> saturated red, dark cloth -> blue (colours the reference
    never showed). Returns (rgba, mask of the recoloured parts)."""
    mt = materials(n)
    out = n.copy()
    lin = srgb_to_linear(n[..., :3])
    red = lin * np.array([0.95, 0.12, 0.10]); blue = lin * np.array([0.6, 0.9, 2.6])
    w1 = mt['light'][..., None]; w2 = mt['dark'][..., None] * (1 - (to_lab(n[..., :3])[..., 0:1] < 30))
    lin2 = lin * (1 - w1 - w2) + red * w1 + np.clip(blue, 0, 1) * w2
    out[..., :3] = linear_to_srgb(np.clip(lin2, 0, 1))
    return out.astype(np.float32), (mt['light'] > 0.7) | (mt['dark'] > 0.7)
