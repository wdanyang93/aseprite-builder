"""Invariants of the silhouette geometry (scenelight.geom: normals, depth, local width) on synthetic shapes only.

Run: cd tools/sprite_harmonize && python -m pytest -q tests
"""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scenelight import geom as G  # noqa: E402


def disc(R, pad=20):
    S = 2 * R + 2 * pad
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    c = S / 2 - 0.5
    r = np.hypot(xx - c, yy - c)
    return np.clip(R - r + 0.5, 0, 1).astype(np.float32), (xx - c), (yy - c)


def ellipse(a, b, pad=20):
    H, W = 2 * b + 2 * pad, 2 * a + 2 * pad
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2 - 0.5, H / 2 - 0.5
    # signed distance approximation for anti-aliasing
    f = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
    gx, gy = (xx - cx) / a ** 2, (yy - cy) / b ** 2
    gn = np.hypot(gx, gy) / np.maximum(f, 1e-6) + 1e-9
    sd = (1 - f) * np.maximum(f, 1e-6) / gn
    return np.clip(sd + 0.5, 0, 1).astype(np.float32), xx - cx, yy - cy


def doll_alpha(h=600):
    """Alpha of a standing doll: head, torso, arms, legs, a thin tail."""
    s = h / 600.0
    a = np.zeros((h + int(40 * s), int(420 * s)), np.float32)
    cx = a.shape[1] // 2
    P = lambda *v: tuple(int(round(t * s)) for t in v)
    cv2.circle(a, (cx, P(70)[0]), P(45)[0], 1, -1, cv2.LINE_AA)
    cv2.ellipse(a, (cx, P(230)[0]), P(70, 120), 0, 0, 360, 1, -1, cv2.LINE_AA)
    cv2.rectangle(a, (cx - P(55)[0], P(330)[0]), (cx - P(15)[0], P(620)[0]), 1, -1, cv2.LINE_AA)
    cv2.rectangle(a, (cx + P(15)[0], P(330)[0]), (cx + P(55)[0], P(620)[0]), 1, -1, cv2.LINE_AA)
    cv2.rectangle(a, (cx - P(100)[0], P(140)[0]), (cx - P(78)[0], P(350)[0]), 1, -1, cv2.LINE_AA)
    cv2.rectangle(a, (cx + P(78)[0], P(140)[0]), (cx + P(100)[0], P(350)[0]), 1, -1, cv2.LINE_AA)
    cv2.line(a, (cx + P(60)[0], P(300)[0]), (cx + P(190)[0], P(420)[0]), 1, P(9)[0], cv2.LINE_AA)   # thin tail
    return np.clip(a, 0, 1)


def downscale(a, h):
    """Area downscale of an alpha to figure height h (like a game-size frame)."""
    k = h / G.figure_height(a)
    sz = (max(1, round(a.shape[1] * k)), max(1, round(a.shape[0] * k)))
    return np.clip(cv2.resize(a, sz, interpolation=cv2.INTER_AREA), 0, 1)


# ------------------------------------------------------------------ geometry
def test_disc_normals_are_sphere_normals():
    a, x, y = disc(120)
    g = G.analyze(a)
    m = g["mask"]
    R = 120.0
    tz = np.sqrt(np.clip(1 - (x ** 2 + y ** 2) / R ** 2, 0, 1))
    dot = g["n"][..., 0] * x / R + g["n"][..., 1] * y / R + g["n"][..., 2] * tz
    err = np.degrees(np.arccos(np.clip(dot, -1, 1)))[m]
    assert g["H"] == pytest.approx(240, abs=1)
    assert err.mean() < 1.5 and np.percentile(err, 95) < 4.0
    c = a.shape[0] // 2
    assert g["n"][c, c, 2] > 0.99
    assert g["depth_px"][c, c] == pytest.approx(R - np.hypot(0.5, 0.5), abs=0.8)   # pixel (c, c) is 0.7 px off-centre
    # angle convention: 0 = screen right, 90 = screen up
    assert abs(g["angle"][c, c + 110] - 0) < 3 or abs(g["angle"][c, c + 110] - 360) < 3
    assert g["angle"][c - 110, c] == pytest.approx(90, abs=3)
    assert g["angle"][c, c - 110] == pytest.approx(180, abs=3)


def test_ellipse_contour_normals_and_no_seam():
    a, x, y = ellipse(160, 60)
    g = G.analyze(a)
    m = g["mask"]
    band = m & (g["depth_px"] < 2.0)
    tx, ty = x / 160 ** 2, y / 60 ** 2
    tn = np.hypot(tx, ty) + 1e-9
    n2 = g["n"][..., :2] / (np.linalg.norm(g["n"][..., :2], axis=-1, keepdims=True) + 1e-9)
    cosv = (n2[..., 0] * tx / tn + n2[..., 1] * ty / tn)[band]
    assert np.degrees(np.arccos(np.clip(cosv, -1, 1))).mean() < 3.0
    # symmetric about both axes
    n = g["n"]
    assert np.abs(n[..., 0] + n[:, ::-1, 0]).max() < 0.02
    assert np.abs(n[..., 1] + n[::-1, :, 1]).max() < 0.02
    # no seam on the medial axis (the long axis): the 3D normal turns smoothly through the middle
    c = a.shape[0] // 2
    seg = n[c - 3:c + 4, 60:-60]
    jump = np.degrees(np.arccos(np.clip((seg[1:] * seg[:-1]).sum(-1), -1, 1)))
    assert jump.max() < 6.0
    # distance-transform-gradient normals flip by ~180 deg there (what the old prototype printed as seams)
    d = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    gy = np.gradient(d, axis=0)
    assert np.sign(gy[c - 2, 200]) != np.sign(gy[c + 2, 200])


def test_limb_width_thin_parts_and_small_sizes():
    f = doll_alpha(600)
    g = G.analyze(f)
    H = g["H"]
    s = 600 / 600.0
    cx = f.shape[1] // 2
    leg_w = g["width_px"][int(500 * s), cx - int(35 * s)]
    assert leg_w == pytest.approx(40 * s, rel=0.2)
    tail = g["thin"][int(360 * s), cx + int(125 * s)]
    torso = g["thin"][int(240 * s), cx]
    assert tail > 0.9 and torso == 0.0
    assert g["width"][int(360 * s), cx + int(125 * s)] == pytest.approx(9 * s / H, rel=0.35)
    for h in (96, 48):
        gs = G.analyze(downscale(f, h))
        assert gs["H"] == pytest.approx(h, abs=2)
        assert np.isfinite(gs["n"]).all() and np.isfinite(gs["width"]).all()
        assert np.abs(np.linalg.norm(gs["n"][gs["mask"]], axis=-1) - 1).max() < 1e-3
    e = G.analyze(np.zeros((40, 30), np.float32))
    assert e["H"] == 0 and not e["mask"].any()


def test_arc_index_smooth_in_concave_corner():
    a = np.zeros((300, 300), np.float32)
    a[40:260, 40:120] = 1          # L-shape: reflex (concave) corner at (120, 180)
    a[180:260, 40:260] = 1
    m = a > 0.5
    d = G.signed_depth(a)
    arc, C = G.arc_index(m, d, 220.0, a, {"band": 0.2})
    # wedge inside the corner: interior pixels whose nearest silhouette point is the corner pixel
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(~(C["gi"] >= 0), return_indices=True)
    near = C["gi"][iy, ix]
    yy, xx = np.mgrid[0:300, 0:300]
    wedge = m & (yy >= 180) & (xx >= 120) & (yy < 215) & (xx < 155)
    vals_old = np.unique(near[wedge])
    vals_new = arc[wedge]
    assert np.isfinite(vals_new).all()
    assert len(vals_old) <= 0.25 * wedge.sum()             # the nearest index stamps few values over the wedge
    assert np.unique(np.round(vals_new, 1)).size > 5 * len(vals_old)
    # smooth: neighbouring pixels in the wedge differ by a small arc step
    blk = arc[180:215, 120:155]
    assert np.nanmax(np.abs(np.diff(blk, axis=0))) < 3.0 and np.nanmax(np.abs(np.diff(blk, axis=1))) < 3.0


# ------------------------------------------------------------------ lines
