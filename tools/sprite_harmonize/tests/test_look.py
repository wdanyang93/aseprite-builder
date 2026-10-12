"""Invariants of the reference-look transfer (scenelight.look) on synthetic dolls only.

Run: cd tools/sprite_harmonize && python -m pytest -q tests/test_look.py
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scenelight import look as LK  # noqa: E402
from scenelight.color import linear_to_srgb, srgb_to_linear, to_lab  # noqa: E402

CREAM = (0.90, 0.85, 0.74)
SKIN = (0.95, 0.76, 0.64)
HAIR = (0.62, 0.62, 0.66)
WARM = np.array([1.9, 1.3, 0.8], np.float32)   # lamp colour (linear ratio at full glow)


def doll(swap=False, shift=(0, 0), H=420, W=320):
    """Head, torso, two touching legs. swap: the anatomical left leg stands screen-right (feet swapped).
    -> rgba float32, parts {name: bool}, anatomical leg masks."""
    rng = np.random.default_rng(0)
    a = np.zeros((H, W), np.uint8)
    head = cv2.circle(np.zeros_like(a), (160, 70), 40, 1, -1)
    torso = cv2.ellipse(np.zeros_like(a), (160, 190), (62, 90), 0, 0, 360, 1, -1)
    leg_l = cv2.rectangle(np.zeros_like(a), (112, 260), (160, 390), 1, -1)
    leg_r = cv2.rectangle(np.zeros_like(a), (160, 260), (208, 390), 1, -1)
    if swap:
        leg_l, leg_r = leg_r, leg_l
    rgb = np.zeros((H, W, 3), np.float32)
    for m, c in ((leg_l | leg_r, SKIN), (torso, CREAM), (head, HAIR)):
        rgb[m > 0] = c
    rgb *= 1 + 0.06 * rng.standard_normal((H, W, 1)).astype(np.float32)     # painted texture
    alpha = (head | torso | leg_l | leg_r).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)                              # anti-aliased rim
    rgba = np.dstack([np.clip(rgb, 0, 1), alpha]).astype(np.float32)
    parts = {"hair": head > 0, "upper": (torso > 0) & (head == 0), "legs": (leg_l | leg_r) > 0}
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    mv = lambda x: cv2.warpAffine(x, M, (W, H), flags=cv2.INTER_NEAREST)
    rgba = mv(rgba)
    parts = {k: mv(v.astype(np.uint8)) > 0 for k, v in parts.items()}
    return rgba, parts, {"left": mv(leg_l) > 0, "right": mv(leg_r) > 0}


def lit(rgba):
    """Analytic reference: a warm lamp from screen-left (glow = (cos of the edge normal to the left)^2, decaying with
    the depth from the silhouette), on top of a cool, darker ambient."""
    body = (rgba[..., 3] > 0.5).astype(np.uint8)
    g = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 4.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    facing = np.clip(gx / np.maximum(np.hypot(gx, gy), 1e-6), 0, 1)        # outward normal points to screen-left
    d = cv2.distanceTransform(body, cv2.DIST_L2, 5)
    glow = (facing ** 2 * np.exp(-d / 6.0))[..., None]
    lin = srgb_to_linear(rgba[..., :3]) * (np.array([0.55, 0.6, 0.7]) + glow * (WARM - 0.4))
    return np.dstack([linear_to_srgb(np.clip(lin, 0, 1)), rgba[..., 3]]).astype(np.float32)


def _fit():
    o, parts, _ = doll()
    return o, parts, LK.fit_look(o, lit(o), parts, align=False)


def test_alpha_identical_and_transparent_untouched():
    o, parts, look = _fit()
    out = LK.apply_look(o, parts, look)
    assert out.dtype == np.float32 and np.array_equal(out[..., 3], o[..., 3])
    tr = o[..., 3] == 0
    assert np.array_equal(out[..., :3][tr], o[..., :3][tr])
    u8 = (o * 255 + 0.5).astype(np.uint8)
    out8 = LK.apply_look(u8, parts, look)
    assert out8.dtype == np.uint8 and np.array_equal(out8[..., 3], u8[..., 3])
    assert np.array_equal(out8[..., :3][u8[..., 3] == 0], u8[..., :3][u8[..., 3] == 0])


def test_azimuth_and_self_reproduction():
    o, parts, look = _fit()
    assert abs(((look["az"] - 180) + 180) % 360 - 180) < 20          # lamp from screen-left
    ref, out = lit(o), LK.apply_look(o, parts, look)
    d, ang, _ = LK.geometry(o[..., 3])
    lamp = (o[..., 3] > 0.5) & (d * 420 < 4) & (np.cos(ang - np.pi) > 0.7)
    inner = (o[..., 3] > 0.5) & (d * 420 > 30)
    L = lambda x: to_lab(x[..., :3])[..., 0]
    for m in (lamp, inner):
        assert abs((L(out) - L(ref))[m].mean()) < 3.0
    b = lambda x: to_lab(x[..., :3])[..., 2]
    assert (b(out) - b(o))[lamp].mean() > 8                          # the lamp's colour (warm), not just brightness


def test_light_follows_leg_position_not_label():
    """after the feet swap, the leg standing on the lamp side gets the lamp light."""
    _, _, look = _fit()
    res = {}
    for swap in (False, True):
        f, parts, legs = doll(swap=swap)
        out = LK.apply_look(f, parts, look)
        dL = to_lab(out[..., :3])[..., 0] - to_lab(f[..., :3])[..., 0]
        res[swap] = {k: float(dL[m].mean()) for k, m in legs.items()}
    assert res[False]["left"] > res[False]["right"] + 1.0            # anatomical left leg stands screen-left
    assert res[True]["right"] > res[True]["left"] + 1.0              # swapped: now the right leg is lit


def test_nothing_depends_on_screen_position():
    _, _, look = _fit()
    f0, p0, _ = doll()
    f1, p1, _ = doll(shift=(14, 9))
    o0 = (LK.apply_look(f0, p0, look) * 255).round()
    o1 = (LK.apply_look(f1, p1, look) * 255).round()
    assert np.abs(o1[9:, 14:] - o0[:-9, :-14]).max() <= 1


def test_no_seams_deep_inside_a_wide_figure():
    """a figure much deeper than 0.1 H: the normal confidence and the look fade smoothly toward the medial axis
    (no cut-off where the facing table suddenly stops)."""
    a = cv2.ellipse(np.zeros((600, 600), np.uint8), (300, 300), (250, 280), 0, 0, 360, 1, -1)
    alpha = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 0.7)
    d, _, conf = LK.geometry(alpha)
    assert d.max() > 0.3
    fac = 0.5 * np.cos(np.arange(LK.NF) * 2 * np.pi / LK.NF)                # strong facing pattern in every band
    nb = len(LK.D_CENTERS)
    tab = {"dir": np.tile(fac[None, :, None], (nb, 1, 3)).tolist(),
           "iso": np.zeros((nb, 3)).tolist(), "slope": np.zeros((nb, 3)).tolist()}
    look = {"az": 30.0, "eps": LK.EPS, "tone": [0.0, -9.0, 9.0], "tables": {"*": tab}}
    q = LK.log_ratio(alpha, {"upper": a > 0}, look)[..., 0]
    m = alpha > 0.5
    for x in (conf, q):
        jx = np.abs(np.diff(x, axis=1))[m[:, 1:] & m[:, :-1]]
        jy = np.abs(np.diff(x, axis=0))[m[1:] & m[:-1]]
        assert max(jx.max(), jy.max()) < 0.1

def test_lines_never_lighter():
    o, parts, look = _fit()
    lines = np.zeros(o.shape[:2], np.float32)
    lines[:, 113:115] = 1.0                                          # a stroke across the lamp-side edge region
    lines[200, :] = 1.0
    u8 = (o * 255 + 0.5).astype(np.uint8)
    for x in (o, u8):
        out = LK.apply_look(x, parts, look, lines=lines)
        held = (lines >= 0.5) & (x[..., 3] > 0)
        assert (out[..., :3][held].astype(np.float64) <= x[..., :3][held].astype(np.float64) + 1e-6).all()


def test_soft_knee_keeps_hue_and_never_whites_out():
    base = srgb_to_linear(np.array([[CREAM, SKIN, HAIR]], np.float32))
    over = base * WARM * 1.6                                         # far above white
    k = LK.soft_knee(over, base)
    assert k.max() <= 1.0 + 1e-6 and (k.max(-1) > 0.95).all()
    assert np.allclose(k / k.max(-1, keepdims=True), over / over.max(-1, keepdims=True), atol=1e-5)
    assert np.allclose(LK.soft_knee(base, base), base)               # an unchanged pixel stays unchanged


def test_look_is_json_and_unknown_parts_use_figure_table():
    o, parts, look = _fit()
    again = json.loads(json.dumps(look))
    assert np.allclose(LK.apply_look(o, parts, again), LK.apply_look(o, parts, look), atol=1e-6)
    renamed = {"cape": parts["upper"], "hair": parts["hair"], "legs": parts["legs"]}
    out = LK.apply_look(o, renamed, again)
    assert np.isfinite(out).all() and np.array_equal(out[..., 3], o[..., 3])


def test_fit_with_alignment_on_shifted_reference():
    o, parts, _ = doll()
    ref = lit(o)
    ref = cv2.warpAffine(ref, np.float32([[1, 0, 4], [0, 1, -3]]), (o.shape[1], o.shape[0]))
    look = LK.fit_look(o, ref, parts)
    assert abs(((look["az"] - 180) + 180) % 360 - 180) < 25
