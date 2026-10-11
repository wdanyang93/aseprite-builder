"""Flat-background matte (scenelight.matte) and brightness-robust alignment (scenelight.align2) on synthetic dolls.

Run: cd tools/sprite_harmonize && python -m pytest -q tests/test_matte_align2.py
"""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scenelight import align2, matte  # noqa: E402
from scenelight.align import Pairs as OldPairs  # noqa: E402
from scenelight.color import linear_to_srgb, srgb_to_linear, to_lab  # noqa: E402

SKIN, CLOTH, HAIR, INK = (0.95, 0.78, 0.68), (0.30, 0.42, 0.70), (0.85, 0.83, 0.80), (0.16, 0.10, 0.09)


def doll(h=420, w=320, gap=True, other=False):
    """Standing doll: head + hair, torso with striped cloth, arms, legs, drawn ink outline and interior lines, painted
    shading. With gap=True one arm meets the hip so a real background hole is enclosed."""
    a = np.zeros((h, w), np.uint8)
    cx = w // 2
    cv2.circle(a, (cx, 58), 36, 1, -1)
    cv2.ellipse(a, (cx, 175), (55, 88), 0, 0, 360, 1, -1)
    if other:                                                              # a different pose: arms up, legs apart
        cv2.fillConvexPoly(a, np.array([[cx - 30, 250], [cx - 6, 250], [cx - 70, 400], [cx - 100, 400]]), 1)
        cv2.fillConvexPoly(a, np.array([[cx + 6, 250], [cx + 30, 250], [cx + 100, 400], [cx + 70, 400]]), 1)
        cv2.fillConvexPoly(a, np.array([[cx - 50, 120], [cx - 36, 110], [cx - 110, 20], [cx - 126, 32]]), 1)
        cv2.fillConvexPoly(a, np.array([[cx + 36, 110], [cx + 50, 120], [cx + 126, 32], [cx + 110, 20]]), 1)
        gap = None
    else:
        cv2.rectangle(a, (cx - 46, 250), (cx - 10, 400), 1, -1)
        cv2.rectangle(a, (cx + 10, 250), (cx + 46, 400), 1, -1)
        cv2.rectangle(a, (cx - 80, 110), (cx - 62, 245), 1, -1)           # left arm, free
    if gap is None:
        pass
    elif gap:                                                                # right arm bent back onto the hip
        cv2.rectangle(a, (cx + 20, 100), (cx + 82, 122), 1, -1)               # shoulder: the gap is enclosed
        cv2.rectangle(a, (cx + 62, 110), (cx + 82, 200), 1, -1)
        cv2.rectangle(a, (cx + 40, 196), (cx + 82, 214), 1, -1)
    else:
        cv2.rectangle(a, (cx + 62, 110), (cx + 80, 245), 1, -1)
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[:] = SKIN
    cloth = np.zeros((h, w), np.uint8)
    cv2.ellipse(cloth, (cx, 185), (56, 70), 0, 0, 360, 1, -1)
    yy, xx = np.mgrid[0:h, 0:w]
    stripes = (np.sin(yy / 6.0) > 0.3) & (cloth > 0)
    rgb[cloth > 0] = CLOTH
    rgb[stripes] = np.array(CLOTH) * 0.7
    hair = np.zeros((h, w), np.uint8)
    cv2.ellipse(hair, (cx, 45), (38, 26), 0, 180, 360, 1, -1)
    rgb[hair > 0] = HAIR
    d = cv2.distanceTransform(a, cv2.DIST_L2, 5)
    rgb *= (0.8 + 0.2 * np.clip(d / 18, 0, 1))[..., None]
    ink = np.zeros((h, w), np.uint8)                                       # interior lines
    cv2.line(ink, (cx - 30, 150), (cx + 30, 150), 1, 2)
    cv2.line(ink, (cx, 260), (cx, 330), 1, 2)
    cv2.circle(ink, (cx - 12, 60), 4, 1, -1)
    cv2.circle(ink, (cx + 12, 60), 4, 1, -1)
    cv2.ellipse(ink, (cx, 80), (10, 4), 0, 0, 180, 1, 2)
    outline = (d > 0) & (d <= 2.5)
    rgb[(ink > 0) | outline] = INK
    from scipy import ndimage as ndi
    _, (iy, ix) = ndi.distance_transform_edt(a == 0, return_indices=True)
    rgb = rgb[iy, ix]                                                      # anti-aliased ring carries the ink colour
    af = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 0.7)
    return np.dstack([rgb, af]).astype(np.float32)


def place(f, s=1.0, sx=None, dx=0.0, dy=0.0, rot=0.0, canvas=None):
    """Affine copy of f (scale about the feet, separate x scale, shift). Returns (image, M forward 2x3)."""
    h, w = f.shape[:2]
    H, W = canvas or (h, w)
    ys, xs = np.nonzero(f[..., 3] > 0.5)
    cx, cy = (xs.min() + xs.max()) / 2, ys.max()
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), rot, s)
    if sx is not None:
        M[0, :2] *= sx / s
        M[0, 2] = cx - M[0, 0] * cx - M[0, 1] * cy
    M[0, 2] += dx
    M[1, 2] += dy
    return np.clip(cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0), 0, 1).astype(np.float32), M


def graded(f, gain=(0.07, 0.10, 0.22), desat=0.45):
    """Night-style grade of an RGBA (linear gain + desaturation), quantised to 8 bit like a delivered PNG."""
    l = srgb_to_linear(np.clip(f[..., :3], 0, 1))
    Y = (l * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2, keepdims=True)
    l = ((1 - desat) * l + desat * Y) * np.array(gain, np.float32)
    out = np.dstack([np.clip(linear_to_srgb(l), 0, 1), f[..., 3]])
    return (np.round(out * 255) / 255).astype(np.float32)


def on_flat(f, bg=(0.5, 0.5, 0.5), noise=0.004, grad=0.0, seed=0):
    """Composite an RGBA onto a flat colour (optional gentle gradient + noise) -> RGB without alpha, 8-bit."""
    h, w = f.shape[:2]
    rng = np.random.default_rng(seed)
    B = np.ones((h, w, 3), np.float32) * np.array(bg, np.float32)
    B += grad * np.linspace(-1, 1, w, dtype=np.float32)[None, :, None]
    B += rng.normal(0, noise, (h, w, 3)).astype(np.float32)
    a = f[..., 3:4]
    return np.round(np.clip(f[..., :3] * a + B * (1 - a), 0, 1) * 255).astype(np.uint8)


def iou(a, b):
    a, b = a > 0.5, b > 0.5
    return (a & b).sum() / max((a | b).sum(), 1)


def gt_map(M, shape):
    """Backward map (reference grid -> original coordinates) of a forward affine M."""
    Mi = cv2.invertAffineTransform(M)
    gx, gy = np.meshgrid(np.arange(shape[1], dtype=np.float32), np.arange(shape[0], dtype=np.float32))
    return Mi[0, 0] * gx + Mi[0, 1] * gy + Mi[0, 2], Mi[1, 0] * gx + Mi[1, 1] * gy + Mi[1, 2]


@pytest.fixture(scope="module")
def pair():
    """Original doll + a reference 'repainted' 98 % wide and shifted ~10 px (like a repainted reference)."""
    n = doll()
    r, M = place(n, s=1.0, sx=0.98, dx=10, dy=3)
    return n, r, M


# ------------------------------------------------------------------ matte
def test_matte_moves_original_alpha_with_the_right_direction(pair):
    n, r, M = pair
    ref = on_flat(r)
    out, info = matte.matte(n, ref, "orig")
    assert info["mode"] == "orig" and info["iou_aligned"] >= 0.98
    assert iou(out[..., 3], r[..., 3]) >= 0.98
    # WARP_INVERSE_MAP direction: the moved alpha sits where the reference is (shifted right), not mirrored back
    cx = lambda a: np.nonzero(a > 0.5)[1].mean()
    assert abs(cx(out[..., 3]) - cx(r[..., 3])) < 1.0 and cx(out[..., 3]) - cx(n[..., 3]) > 7
    T = np.array(info["affine"])                      # reference -> original: x scale ~ 1 / 0.98
    assert abs(T[0, 0] - 1 / 0.98) < 0.01 and abs(T[1, 1] - 1.0) < 0.01


def test_matte_key_mode_and_unmixing_recover_colours(pair):
    n, r, M = pair
    out, info = matte.matte(n, on_flat(r), "key")
    assert info["mode"] == "key" and iou(out[..., 3], r[..., 3]) >= 0.98
    inner = cv2.erode((r[..., 3] > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.abs(out[..., :3][inner] - r[..., :3][inner]).mean() < 0.01
    ring = (r[..., 3] > 0.3) & (r[..., 3] < 0.97) & (out[..., 3] > 0.3)
    err_unmixed = np.abs(out[..., :3][ring] - r[..., :3][ring]).mean()
    err_raw = np.abs(on_flat(r)[ring].astype(np.float32) / 255 - r[..., :3][ring]).mean()
    assert err_unmixed < 0.6 * err_raw                # the grey of the background is taken out of the edge colours


def test_matte_auto_reports_and_any_flat_colour_with_gradient(pair):
    n, r, M = pair
    ref = on_flat(r, bg=(0.25, 0.65, 0.30), grad=0.02, seed=3)
    out, info = matte.matte(n, ref, "auto")
    assert info["mode"] in ("key", "orig") and "auto" in info and info["iou_aligned"] >= 0.98
    assert abs(info["bg"][1] - 0.65 * 255) < 3 and info["bg_range"] > 5
    assert iou(out[..., 3], r[..., 3]) >= 0.97


def test_matte_refuses_a_different_outline_but_key_mode_works():
    n = doll(gap=True)
    other = doll(other=True)                          # same character, different pose
    ref = on_flat(other)
    with pytest.raises(matte.MatteError, match="does not keep the original outline"):
        matte.matte(n, ref, "auto")
    out, info = matte.matte(n, ref, "key")
    assert info["mode"] == "key" and iou(out[..., 3], other[..., 3]) >= 0.97


def test_matte_passes_real_alpha_through_and_refuses_textured_background(pair):
    n, r, M = pair
    out, info = matte.matte(n, r, "auto")
    assert info["mode"] == "alpha" and np.array_equal(out, r)
    rng = np.random.default_rng(1)
    tex = (rng.random(r.shape[:2] + (3,)) * 0.8).astype(np.float32)
    a = r[..., 3:4]
    busy = np.round((r[..., :3] * a + tex * (1 - a)) * 255).astype(np.uint8)
    with pytest.raises(matte.MatteError, match="not flat"):
        matte.matte(n, busy, "auto")


def test_key_keeps_real_gaps_open_and_fills_specks(pair):
    n, r, M = pair
    ref = on_flat(r)
    ref2 = ref.copy()
    cx = r.shape[1] // 2 + 10
    ref2[176:178, cx - 21:cx - 19] = ref[5, 5]        # 2x2 background-coloured speck inside the torso
    out, info = matte.matte(n, ref2, "key", return_masks=True)
    K = info["masks"]["key"]
    gap = (r[..., 3] < 0.05) & (cv2.dilate((r[..., 3] > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    from scipy import ndimage as ndi
    enclosed = ndi.binary_fill_holes(r[..., 3] > 0.5) & ~(r[..., 3] > 0.5) & gap
    assert enclosed.sum() > 200 and not K[enclosed].any()          # the arm-hip gap stays background
    assert K[176:178, cx - 21:cx - 19].all() and info["holes_kept"] >= 1


def test_defringe_nearest_never_lightens_a_dark_outline():
    f = doll()
    a = f[..., 3]
    halo = (a > 0.02) & (a < 0.9)
    f2 = f.copy()
    f2[..., :3][halo] = 0.95                          # light fringe (white-matte export) on the outline ring
    near = matte.defringe(f2, "nearest")
    inter = matte.defringe(f2, "interior")
    assert np.array_equal(near[..., 3], f2[..., 3]) and np.array_equal(matte.defringe(f2, None), f2)
    L = lambda x: to_lab(x[..., :3])[..., 0]
    ring = (a > 0.05) & (a < 0.9)
    assert L(near)[ring].mean() < L(inter)[ring].mean() - 5           # outline ink kept, not interior colour
    assert np.median(np.abs(near[..., :3][ring] - np.array(INK)).max(-1)) < 0.05
    rep = matte.edge_report(f2, near)
    assert rep["lighter5_px"] == 0


# ------------------------------------------------------------------ align2
def test_align2_recovers_known_shift_and_scale(pair):
    n, r, M = pair
    P = align2.Pairs(n, r)
    gx, gy = gt_map(M, r.shape)
    m = r[..., 3] > 0.5
    err = np.hypot(P.mapx - gx, P.mapy - gy)[m]
    assert P.iou >= 0.98 and np.median(err) < 1.0 and P.fold < 0.01


def test_align2_darkened_reference_aligns_like_the_bright_one(pair):
    n, r, M = pair
    m = r[..., 3] > 0.5
    gx, gy = gt_map(M, r.shape)
    bright = align2.Pairs(n, r, regions=False)
    for g in ((0.07, 0.10, 0.22), (0.018, 0.022, 0.04)):           # night, darkness
        rd = graded(r, g)
        P = align2.Pairs(n, rd, regions=False)
        assert abs(P.iou - bright.iou) <= 0.01
        assert np.median(np.hypot(P.mapx - gx, P.mapy - gy)[m]) < 1.0
        assert P.info["pose_ok"] and align2.structure_match(P) > 0.8


def test_align2_interface_matches_align_pairs(pair):
    n, r, M = pair
    P, O = align2.Pairs(n, r), OldPairs(n, r)
    for k in ("shape", "mapx", "mapy", "neutral_w", "iou", "iou_affine", "reliable", "mat", "regions", "Hf"):
        assert hasattr(P, k), k
    assert P.mapx.shape == O.mapx.shape and P.mapx.dtype == np.float32 and P.neutral_w.shape == O.neutral_w.shape
    assert set(P.mat) == set(O.mat) and len(P.regions) > 0 and set(P.regions[0]) == set(O.regions[0])
    assert np.allclose(P.warp(n), P.neutral_w)
    body = (P.neutral_w[..., 3] > 0.5) & (r[..., 3] > 0.5)
    assert P.conf.min() >= 0 and P.conf.max() <= 1 and not P.conf[~body].any()
    assert P.reliable.sum() > 0.5 * O.reliable.sum()


def test_align2_rejects_flow_that_lowers_iou(pair, monkeypatch):
    n, r, M = pair
    H, W = r.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bad = np.dstack([25 * np.sin(yy / 30), 25 * np.cos(xx / 25)]).astype(np.float32)
    monkeypatch.setattr(align2, "_dis", lambda I0, I1, Hf: bad)
    P = align2.Pairs(n, r, regions=False)
    assert not P.flow_used and P.flow_from == "affine" and abs(P.iou - P.iou_affine_full) < 1e-6


def test_align2_confidence_drops_where_the_drawings_differ(pair):
    n, r, M = pair
    r2 = r.copy()
    cx = r.shape[1] // 2 + 10
    for k in range(6):                                   # repainted detail: new strokes on the left leg only
        cv2.line(r2, (cx - 40, 270 + 20 * k), (cx - 18, 280 + 20 * k), (*INK, 1.0), 2)
    P = align2.Pairs(n, r2, regions=False)
    leg = np.zeros(r.shape[:2], bool)
    leg[265:395, cx - 42:cx - 16] = True
    rest = (P.conf > 0) & ~cv2.dilate(leg.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    assert P.conf[leg & (r[..., 3] > 0.9)].mean() < P.conf[rest].mean() - 0.2


def test_align2_deterministic_and_structure_gain_free(pair):
    n, r, M = pair
    P1, P2 = align2.Pairs(n, r, regions=False), align2.Pairs(n, r, regions=False)
    assert np.array_equal(P1.mapx, P2.mapx) and np.array_equal(P1.conf, P2.conf)
    for kind in ("rank", "grad", "mix"):
        s0 = align2.structure(r, kind).astype(np.float32)
        s1 = align2.structure(graded(r), kind).astype(np.float32)
        m = r[..., 3] > 0.5
        assert np.abs(s0 - s1)[m].mean() < 12, kind


def test_matte_then_align_on_a_dark_flat_background_reference(pair):
    n, r, M = pair
    ref = on_flat(graded(r), bg=(0.47, 0.47, 0.48))
    out, info = matte.matte(n, ref, "auto")
    P = align2.Pairs(n, out, regions=False)
    assert P.iou >= 0.97 and P.info["use"] == "pack"
