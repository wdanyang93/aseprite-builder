"""Bench metrics on synthetic dolls: known answers, the NaN guard and the angle convention (no user art, no data paths)."""
import os
import sys

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenelight.color import to_lab, from_lab, srgb_to_linear, linear_to_srgb  # noqa: E402
from bench import metrics as MX  # noqa: E402
from bench import parts as PARTS  # noqa: E402
from bench import synth_audit as SA, synth_practice as SP, synth_methods as SMe  # noqa: E402
from bench import frames as FR  # noqa: E402


def doll(h=320, w=240, line=True, patch=False):
    """Upright ellipse: left half pink skin, right half white cloth, 3-px dark outline, a thin interior stroke, optional
    wide dark patch (dark cloth, not a line)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx, ry, rx = h / 2, w / 2, 0.42 * h, 0.36 * w
    r = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    a = np.clip((1.0 - r) * ry / 1.5, 0, 1).astype(np.float32)
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[:] = (0.93, 0.74, 0.66)
    rgb[xx > cx] = (0.92, 0.92, 0.90)
    if patch:
        rgb[(np.abs(yy - cy - 0.15 * h) < 0.08 * h) & (np.abs(xx - cx) < 0.18 * w)] = (0.18, 0.14, 0.12)
    if line:
        ring = (r > 0.93) & (r <= 1.0)
        rgb[ring] = (0.12, 0.10, 0.10)
        stroke = (np.abs(xx - cx + 0.12 * w) < 1.2) & (np.abs(yy - cy) < 0.25 * h)
        rgb[stroke] = (0.12, 0.10, 0.10)
    return np.dstack([rgb, a]).astype(np.float32)


def lit_from(F, deg, gain=0.6):
    """Multiply light that brightens the side facing screen direction `deg` (outward normal . light > 0)."""
    a = F[..., 3]
    d, nx, ny = MX.edge_geometry(a)
    u = MX.unit(deg)
    face = np.clip(nx * u[0] + ny * u[1], 0, 1) * np.exp(-d / 12.0)
    lin = srgb_to_linear(F[..., :3]) * (0.7 + gain * face)[..., None]
    return np.dstack([np.clip(linear_to_srgb(lin), 0, 1), a]).astype(np.float32)


# ------------------------------------------------------------------ angle convention
def test_angle_convention():
    assert np.allclose(MX.unit(0), [1, 0], atol=1e-6)
    assert np.allclose(MX.unit(90), [0, -1], atol=1e-6)          # screen up = image row index decreasing
    assert np.allclose(MX.unit(180), [-1, 0], atol=1e-6)
    assert MX.angle_of(1.0, 0.0) == pytest.approx(0)
    assert MX.angle_of(0.0, -1.0) == pytest.approx(90)
    assert MX.angle_of(0.0, 1.0) == pytest.approx(-90)
    assert MX.ang_diff(350, 10) == pytest.approx(20)
    assert MX.ang_diff(-170, 170) == pytest.approx(20)


def test_edge_normals_point_outward_in_screen_angles():
    F = doll()
    d, nx, ny = MX.edge_geometry(F[..., 3])
    a = F[..., 3] > 0.5
    ys, xs = np.nonzero(a)
    cy = int(np.median(ys))
    xr = xs[ys == cy].max() - 1
    yt = ys.min() + 1
    xc = int(np.median(xs[ys == ys.min()]))
    assert MX.angle_of(nx[cy, xr], ny[cy, xr]) == pytest.approx(0, abs=10)     # right edge faces 0 deg
    assert MX.angle_of(nx[yt, xc], ny[yt, xc]) == pytest.approx(90, abs=10)    # top edge faces 90 deg


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_renderers_light_the_side_they_say(deg):
    """All three copied research renderers: a key at screen angle deg brightens the edge facing deg."""
    F = doll(line=False)
    a = F[..., 3] > 0.5
    d, nx, ny = MX.edge_geometry(F[..., 3])
    u = MX.unit(deg)
    face = nx * u[0] + ny * u[1]
    band = a & (d <= 6)
    sun, shd = band & (face > 0.7), band & (face < -0.7)
    outs = [SA.lit(F, dict(amb=(0.2, 0.2, 0.2), key=(0.9, 0.9, 0.9), key_dir=(deg, 10), wrap=0.0, point=None, desat=0.0)),
            SP.light_scene(F, dict(amb=[0.2] * 3, key=[0.9] * 3, az=deg, el=10, wrap=0.0,
                                   grade=dict(L_lo=0, L_hi=100, sat=1.0, a_shift=0, b_shift=0, purkinje=0))),
            SMe.night(F, moon_deg=deg, moon_el=10, lantern=None, amb=(0.2, 0.2, 0.2), moon_rgb=(0.9, 0.9, 0.9), wrap=0.0,
                      desat=0.0, haze=(0, 0, 0))]
    for o in outs:
        L = MX.L_of(o)
        assert L[sun].mean() > L[shd].mean() + 10


# ------------------------------------------------------------------ known answers
def test_perfect_output_scores_zero():
    F = doll()
    T = lit_from(F, 30)
    r = MX.score(T.copy(), T, F, light_deg=30)
    assert r["dEa"] == pytest.approx(0, abs=1e-4)
    assert r["dEst"] == pytest.approx(0, abs=1e-4)
    assert r["sun"] == pytest.approx(0, abs=1e-4) and r["shd"] == pytest.approx(0, abs=1e-4)
    assert r["hp"] == pytest.approx(1, abs=1e-4)
    assert r["alpha_same"] and not r["nan_out"] and r["nan_keys"] == []
    assert r["con"] == pytest.approx(r["con_ref"]) and r["con_ref"] > 3        # lit side really is brighter
    assert r["edge_mode"] == "angle"


def test_contrast_sign_follows_the_light_azimuth():
    F = doll()
    T = lit_from(F, 0)
    assert MX.score(T, T, F, light_deg=0)["con"] > 3
    assert MX.score(T, T, F, light_deg=180)["con"] < -3


def test_constant_offset_is_colour_error_not_structure():
    F = doll()
    T = lit_from(F, 30)
    lab = to_lab(T[..., :3])
    lab[..., 0] += 5.0
    O = np.dstack([from_lab(lab), T[..., 3]]).astype(np.float32)
    r = MX.score(O, T, F, light_deg=30)
    assert r["dEa"] == pytest.approx(5.0, abs=0.3)
    assert r["dEst"] < 0.3
    assert r["det"] > 0.99


def test_identity_output_misses_the_light():
    F = doll()
    T = lit_from(F, 0, gain=1.0)
    r = MX.score(F, T, F, light_deg=0)
    assert r["sun"] > r["shd"]                     # the lit edge is where the identity is most wrong
    assert r["con"] < r["con_ref"] - 3


def test_alpha_change_and_lightened_lines_are_caught():
    F = doll()
    O = F.copy()
    O[10:12, 10:12, 3] = np.clip(O[10:12, 10:12, 3] + 1 / 255.0, 0, 1)
    O[F[..., 3] == 0.5] += 0                       # no-op: only the alpha bit flip matters
    r = MX.score(O, F, F)
    assert not r["alpha_same"]
    lines = MX.thin_lines(F)
    d = cv2.distanceTransform((F[..., 3] > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    O = F.copy()
    O[lines & (d > 6), :3] = 0.6
    r = MX.score(O, F, F)
    assert r["lit_lines"] > 50 and r["lit_share"] > 0.5
    assert r["lnC"] < 0.75


def test_thin_lines_are_strokes_not_dark_areas():
    F = doll(patch=True)
    lines = MX.thin_lines(F)
    h, w = F.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    stroke = (np.abs(xx - w / 2 + 0.12 * w) < 1.2) & (np.abs(yy - h / 2) < 0.25 * h)
    patch_core = (np.abs(yy - h / 2 - 0.15 * h) < 0.08 * h - 4) & (np.abs(xx - w / 2) < 0.18 * w - 4)
    patch = (np.abs(yy - h / 2 - 0.15 * h) < 0.08 * h + 3) & (np.abs(xx - w / 2) < 0.18 * w + 3)
    assert lines[stroke & ~patch].mean() > 0.8      # (where the stroke crosses the dark patch it is invisible)
    assert lines[patch_core].mean() < 0.05


# ------------------------------------------------------------------ NaN guard
def test_nan_output_is_flagged_never_silent():
    F = doll()
    T = lit_from(F, 30)
    O = T.copy()
    O[100:110, 100:110, :3] = np.nan
    r = MX.score(O, T, F, light_deg=30)
    assert r["nan_out"] and "nan_out" in r["nan_keys"]
    assert np.isfinite(r["dEa"])                   # numbers are still computed on the finite part, and flagged


def test_uncomputable_metric_is_listed():
    F = doll(h=24, w=18)                           # too few reliable pixels for the low-pass colour error
    T = lit_from(F, 30)
    r = MX.score(T, T, F, light_deg=30)
    assert not np.isfinite(r["dEa"])
    assert "dEa" in r["nan_keys"]


def test_edge_metric_uses_parts_when_labels_exist():
    F = doll()
    T = lit_from(F, 0)
    lab = np.zeros(F.shape[:2], np.uint8)
    lab[:, : F.shape[1] // 2] = PARTS.NAMES.index("arm_back") + 1
    lab[:, F.shape[1] // 2:] = PARTS.NAMES.index("arm_front") + 1
    r = MX.score(T, T, F, light_deg=0, labels=lab)
    assert r["edge_mode"] == "parts"
    assert r["sun"] == pytest.approx(0, abs=1e-4)


# ------------------------------------------------------------------ compat replicas
def test_compat_replicas_identity():
    F = doll(h=480, w=360)
    T = lit_from(F, 30)
    ev = MX.eval_audit(T, T, {"all": F[..., 3] > 0.5}, MX.fig_h(F[..., 3]))
    assert ev["all"]["lp_dE"] == 0 and ev["all"]["hp_ratio"] == pytest.approx(1)
    valid = MX.practice_valid(F, T)
    s = MX.score_practice(T, T, valid, MX.fig_h(F[..., 3]))
    assert s["dE_mean"] == pytest.approx(0) and s["L_corr"] == pytest.approx(1)


def test_heldout_frame_generator_is_seeded():
    F = doll()
    rig = {"fixed": [(120, 60), (120, 160)], "J": {"hand": (60, 200), "head": (120, 60)}}
    a1, m1 = FR.make_frame(F, rig, {"hand": (10, -5)}, seed=1)
    a2, m2 = FR.make_frame(F, rig, {"hand": (10, -5)}, seed=1)
    a3, m3 = FR.make_frame(F, rig, {"hand": (10, -5)}, seed=2)
    assert np.array_equal(m1, m2) and not np.array_equal(m1, m3)
    assert m1.shape == F.shape[:2] + (2,)
