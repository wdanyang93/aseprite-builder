"""Fast invariants for scenelight on a synthetic figure (no real art needed).

Run: cd tools/sprite_harmonize && python -m pytest -q tests
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scenelight import fit_scene, relight  # noqa: E402
from scenelight.color import to_lab  # noqa: E402
from scenelight.composite import composite  # noqa: E402
import relight as cli  # noqa: E402

SKIN = (0.97, 0.80, 0.70)  # neutral skin of the real art: L* ~87, C* ~20
CLOTH = (0.95, 0.95, 0.94)


def figure(h=360, w=260):
    """Simple standing figure: head, torso, white top with two cups, arms, legs; soft painted shading."""
    a = np.zeros((h, w), np.float32)
    rgb = np.zeros((h, w, 3), np.float32)
    cx = w // 2
    cv2.circle(a, (cx, 45), 28, 1, -1)
    cv2.ellipse(a, (cx, 140), (44, 70), 0, 0, 360, 1, -1)
    cv2.rectangle(a, (cx - 36, 200), (cx - 8, 340), 1, -1)
    cv2.rectangle(a, (cx + 8, 200), (cx + 36, 340), 1, -1)
    cv2.rectangle(a, (cx - 62, 85), (cx - 48, 200), 1, -1)
    cv2.rectangle(a, (cx + 48, 85), (cx + 62, 200), 1, -1)
    rgb[:] = SKIN
    top = np.zeros((h, w), np.uint8)
    cv2.ellipse(top, (cx - 18, 112), (18, 15), 0, 0, 360, 1, -1)
    cv2.ellipse(top, (cx + 18, 112), (18, 15), 0, 0, 360, 1, -1)
    rgb[top > 0] = CLOTH
    # mild painted shading: darker toward the silhouette
    d = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    rgb *= (0.85 + 0.15 * np.clip(d / 20, 0, 1))[..., None]
    # drawn chest form (what the artist painted): cleavage shadow, shadows under the cups, highlights on the cups
    form = np.zeros((h, w), np.float32)
    cv2.ellipse(form, (cx, 100), (4, 14), 0, 0, 360, -0.25, -1)
    cv2.ellipse(form, (cx - 18, 130), (16, 5), 0, 0, 360, -0.22, -1)
    cv2.ellipse(form, (cx + 18, 130), (16, 5), 0, 0, 360, -0.22, -1)
    cv2.circle(form, (cx - 22, 106), 6, 0.12, -1)
    cv2.circle(form, (cx + 14, 106), 6, 0.12, -1)
    form = cv2.GaussianBlur(form, (0, 0), 2.5)
    rgb *= (1 + form)[..., None]
    a = cv2.GaussianBlur(a, (0, 0), 0.7)
    return np.dstack([rgb, a]).astype(np.float32)


def warm_reference(f):
    """Same figure lit warm from the upper right: Lambert shading on an inflated silhouette, orange tint."""
    a = (f[..., 3] > 0.5).astype(np.uint8)
    h = np.sqrt(cv2.distanceTransform(a, cv2.DIST_L2, 5))
    h = cv2.GaussianBlur(h, (0, 0), 2)
    gy, gx = np.gradient(h)
    n = np.dstack([-gx, -gy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    l = np.array([0.75, -0.35, 0.55]); l /= np.linalg.norm(l)
    lit = 0.55 + 0.6 * np.clip(n @ l, 0, 1)
    r = f.copy()
    r[..., :3] = np.clip(f[..., :3] * np.array([1.0, 0.8, 0.62]) * lit[..., None], 0, 1)
    return r


@pytest.fixture(scope="module")
def fitted():
    n = figure()
    r = warm_reference(n)
    prof, pairs = fit_scene(n, r, "front")
    return n, r, prof, pairs


def test_alignment_of_identical_pose(fitted):
    _, _, _, pairs = fitted
    assert pairs.iou > 0.95


def test_profile_is_small_json(fitted):
    _, _, prof, _ = fitted
    s = json.dumps(prof)
    assert len(s) < 32768
    assert prof["version"] == 2


def test_alpha_unchanged_and_deterministic(fitted):
    n, _, prof, _ = fitted
    o1 = relight(n, prof, "front")
    o2 = relight(n, prof, "front")
    assert o1.shape == n.shape
    assert np.array_equal(o1[..., 3], n[..., 3])
    assert np.array_equal(o1, o2)
    assert np.isfinite(o1).all() and o1.min() >= 0 and o1.max() <= 1


def test_learns_warm_light_from_the_right(fitted):
    n, _, prof, _ = fitted
    o = relight(n, prof, "front")
    lab_o, lab_n = to_lab(o[..., :3]), to_lab(n[..., :3])
    a = (n[..., 3] > 0.5).astype(np.uint8)
    inner = cv2.erode(a, np.ones((5, 5), np.uint8)) > 0
    gx = cv2.Sobel(cv2.GaussianBlur(a.astype(np.float32), (0, 0), 4), cv2.CV_32F, 1, 0)
    band = inner & (cv2.distanceTransform(a, cv2.DIST_L2, 5) < 10)
    facing_right = band & (gx < -0.02)   # alpha falls off to the right -> surface faces right
    facing_left = band & (gx > 0.02)
    d = lab_o[..., 0] - lab_n[..., 0]
    assert d[facing_right].mean() > d[facing_left].mean() + 3.0, (d[facing_left].mean(), d[facing_right].mean())
    body = n[..., 3] > 0.9
    assert lab_o[..., 2][body].mean() > lab_n[..., 2][body].mean() + 5  # warmer (more yellow/orange)


def test_breast_detection_is_view_aware(fitted):
    from scenelight.shading import geometry
    n, _, prof, _ = fitted
    geo = dict(prof["geo"], bh=0.6)  # ellipsoids are optional (off by default)
    assert geometry(n, "back", geo)["breasts"] == []
    assert len(geometry(n, "front", geo)["breasts"]) == 2
    assert len(geometry(n, "left", geo)["breasts"]) == 1


def test_default_keeps_drawn_form(fitted):
    """Relighting must not change the drawn shape: never subtract painted shading, no ellipsoid bumps."""
    from scenelight.report import drawn_form_kept
    n, _, prof, _ = fitted
    assert prof["geo"]["bh"] == 0.0
    assert prof["light"][15] >= 0 and prof["light"][16] >= 0
    assert drawn_form_kept(n, relight(n, prof, "front")) > 0.95


def test_works_at_other_scales(fitted):
    n, _, prof, _ = fitted
    small = cv2.resize(n, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    o = relight(small, prof, "front")
    assert o.shape == small.shape and np.isfinite(o).all()


def test_composite_places_sprite(fitted):
    n, _, prof, _ = fitted
    bg = np.full((300, 400, 3), 0.3, np.float32)
    out = composite(bg, relight(n, prof, "front"), 200, 280, 150)
    assert out.shape == bg.shape
    assert not np.allclose(out, bg)


@pytest.mark.parametrize("name,view", [
    ("S231_roll_SE_03.png", "front_right"), ("walk_front_left_2.png", "front_left"),
    ("backstep_n_01.png", "back"), ("idle-east.png", "right"), ("lyn_07.png", None), ("구르기_남서_1.png", "front_left"),
])
def test_view_from_file_name(name, view):
    assert cli.view_from_name(name) == view
