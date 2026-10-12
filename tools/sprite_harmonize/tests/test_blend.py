"""Invariants of the scene blend (scenelight/blend.py) on synthetic dolls only.

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

from scenelight import blend as BL  # noqa: E402
from scenelight.color import from_lab, to_lab  # noqa: E402

GAME = 500          # game height of the 1000-row doll canvas


def doll(H=1000, W=600, outline=(0.55, 0.42, 0.34), seed=0):
    """Head (skin) with a hair cap, body (clothes), tail; textured fills, a 3 px outline, soft alpha.
    Returns rgba (float), parts, lines (the outline + one interior stroke, as Adobe would give them)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    head = ((yy - 0.2 * H) / (0.12 * H)) ** 2 + ((xx - 0.5 * W) / (0.12 * H)) ** 2 <= 1
    body = ((yy - 0.58 * H) / (0.3 * H)) ** 2 + ((xx - 0.5 * W) / (0.16 * H)) ** 2 <= 1
    tail = ((yy - 0.72 * H) / (0.07 * H)) ** 2 + ((xx - 0.8 * W) / (0.12 * H)) ** 2 <= 1
    hair = head & (yy < 0.17 * H)
    parts = {"hair": hair, "skin": head & ~hair, "clothes": body & ~head, "tail": tail & ~body}
    rgb = np.zeros((H, W, 3), np.float32)
    for k, c, amp in (("skin", (0.95, 0.78, 0.68), 0.05), ("hair", (0.86, 0.85, 0.83), 0.08),
                      ("clothes", (0.38, 0.27, 0.2), 0.12), ("tail", (0.9, 0.87, 0.84), 0.1)):
        m = parts[k]
        rgb[m] = np.clip(np.float32(c) + amp * rng.standard_normal((int(m.sum()), 1)).astype(np.float32), 0, 1)
    sil = head | body | tail
    depth = cv2.distanceTransform(sil.astype(np.uint8), cv2.DIST_L2, 5)
    ring = sil & (depth <= 3)
    rgb[ring] = outline
    stroke = np.zeros((H, W), bool)
    stroke[int(0.45 * H):int(0.7 * H), int(0.5 * W) - 1:int(0.5 * W) + 1] = True   # interior line (Adobe path)
    rgb[stroke] = (0.2, 0.14, 0.1)
    alpha = np.clip(cv2.GaussianBlur(sil.astype(np.float32), (0, 0), 0.8), 0, 1)
    alpha[alpha < 0.02] = 0
    rgba = np.dstack([rgb, alpha]).astype(np.float32)
    return rgba, parts, (ring | stroke).astype(np.float32)


def painted_target(rgba, parts, line_lab=(22.0, 14.0, 18.0)):
    """The 'painter' frame at game scale: darks lifted, lights compressed, warmer, a dark outer line; on grey."""
    small = BL.downscale(rgba, GAME)
    lab = to_lab(small[..., :3])
    lab[..., 0] = 18 + 0.72 * lab[..., 0]
    lab[..., 1] += 4.0
    lab[..., 2] += 9.0
    sil = small[..., 3] >= 0.5
    lab[sil & (BL._depth(sil) <= 1.0)] = line_lab        # the outermost pixels (incl. diagonal neighbours)
    rgb = from_lab(lab) * small[..., 3:] + 0.5 * (1 - small[..., 3:])
    tparts = {k: BL._resize(v.astype(np.float32), small.shape[1::-1], "area") >= 0.5 for k, v in parts.items()}
    return rgb, tparts, sil


@pytest.fixture(scope="module")
def case():
    rgba, parts, lines = doll()
    target, tparts, tsil = painted_target(rgba, parts)
    look = BL.fit_blend(rgba, parts, target, tparts, GAME, lines=lines, target_sil=tsil)
    out = BL.apply_blend(rgba, parts, look, GAME, lines=lines)
    return dict(rgba=rgba, parts=parts, lines=lines, target=target, tparts=tparts, tsil=tsil, look=look, out=out)


def test_alpha_is_never_changed(case):
    full = BL.apply_blend(case["rgba"], case["parts"], case["look"], None, lines=case["lines"])
    assert np.array_equal(full[..., 3], case["rgba"][..., 3])                       # full res: the input alpha
    assert np.array_equal(case["out"][..., 3], BL.downscale(case["rgba"], GAME)[..., 3])   # game res: plain downscale
    u8 = (case["rgba"] * 255 + 0.5).astype(np.uint8)
    full8 = BL.apply_blend(u8, case["parts"], case["look"], None, lines=case["lines"])
    assert np.array_equal((full8[..., 3] * 255 + 0.5).astype(np.uint8), u8[..., 3])        # uint8 in: same bytes


def test_line_art_never_lighter():
    rgba, parts, lines = doll()
    # a target much LIGHTER than the sprite: the curves lift, the Adobe line pixels must not follow
    small = BL.downscale(rgba, GAME)
    lab = to_lab(small[..., :3])
    lab[..., 0] = 40 + 0.6 * lab[..., 0]
    tparts = {k: BL._resize(v.astype(np.float32), small.shape[1::-1], "area") >= 0.5 for k, v in parts.items()}
    look = BL.fit_blend(rgba, parts, from_lab(lab), tparts, GAME, lines=lines, target_sil=small[..., 3] >= 0.5)
    full = BL.apply_blend(rgba, parts, look, None, lines=lines)
    m = (lines > 0) & (rgba[..., 3] > 0)
    dL = to_lab(full[..., :3])[..., 0][m] - to_lab(rgba[..., :3])[..., 0][m]
    assert dL.max() <= 1e-3
    body = (rgba[..., 3] > 0.99) & (lines == 0) & parts["clothes"]
    assert np.median(to_lab(full[..., :3])[..., 0][body] - to_lab(rgba[..., :3])[..., 0][body]) > 5   # the rest lifts


def test_parts_match_target_tone_and_colour(case):
    out, target, tsil = case["out"], case["target"], case["tsil"]
    osil = out[..., 3] >= 0.5
    lo, lt = to_lab(out[..., :3]), to_lab(target)
    do = cv2.distanceTransform(osil.astype(np.uint8), cv2.DIST_L2, 5)
    dt = cv2.distanceTransform(tsil.astype(np.uint8), cv2.DIST_L2, 5)
    for k in ("skin", "hair", "clothes", "tail"):
        core = cv2.erode(case["tparts"][k].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # away from other parts
        mo = core & osil & (do > 4)
        mt = core & tsil & (dt > 4)
        for q in (10, 50, 90):
            assert abs(np.percentile(lo[..., 0][mo], q) - np.percentile(lt[..., 0][mt], q)) < 3.0, (k, q)
        assert abs(lo[..., 1][mo].mean() - lt[..., 1][mt].mean()) < 1.5, k
        assert abs(lo[..., 2][mo].mean() - lt[..., 2][mt].mean()) < 1.5, k


def test_outer_line_takes_target_darkness(case):
    out, target = case["out"], case["target"]
    sil = out[..., 3] >= 0.5
    first = sil & (BL._depth(sil) <= 1.0)
    on_grey = lambda x: x[..., :3] * x[..., 3:] + 0.5 * (1 - x[..., 3:])       # the target is on grey too
    L = to_lab(on_grey(out))[..., 0]
    Lt = to_lab(target)[..., 0]
    plain = to_lab(on_grey(BL.downscale(case["rgba"], GAME)))[..., 0]
    assert np.median(plain[first]) - np.median(Lt[first]) > 15  # the doll's own outline: much lighter
    assert abs(np.median(L[first]) - np.median(Lt[first])) < 4.0
    assert case["look"]["parts"]["hair"]["line"][0] < 35


def test_outer_line_never_lighter_than_input():
    rgba, parts, lines = doll(outline=(0.08, 0.06, 0.05))      # sprite outline darker than the target line
    target, tparts, tsil = painted_target(rgba, parts, line_lab=(45.0, 10.0, 15.0))
    look = BL.fit_blend(rgba, parts, target, tparts, GAME, lines=lines, target_sil=tsil)
    out = BL.apply_blend(rgba, parts, look, GAME, lines=lines)
    ref = BL.downscale(rgba, GAME)
    a = out[..., 3]
    d = cv2.distanceTransform((a >= 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    band = (a > 0) & (d <= 1.0)
    assert (to_lab(out[..., :3])[..., 0][band] - to_lab(ref[..., :3])[..., 0][band]).max() <= 1e-3


def test_look_is_json_and_reapplies_identically(case):
    look2 = json.loads(json.dumps(case["look"]))
    out2 = BL.apply_blend(case["rgba"], case["parts"], look2, GAME, lines=case["lines"])
    assert np.array_equal(out2, case["out"])
    assert set(look2["parts"]) >= {"hair", "skin", "clothes", "tail", "_all"}


def test_other_frame_with_missing_parts_runs(case):
    rgba, parts, lines = doll(seed=3)
    parts = {"hair": parts["hair"], "tail": parts["tail"]}     # Adobe found fewer parts on this frame
    out = BL.apply_blend(rgba, parts, case["look"], GAME, lines=lines)
    assert np.array_equal(out[..., 3], BL.downscale(rgba, GAME)[..., 3])
    assert np.isfinite(out).all()


def test_curve_is_monotone_and_smooth():
    lut = BL._lut([30, 30, 31, 60, 59, 80, 95, 95], [40, 38, 45, 62, 70, 82, 90, 89])   # ties / noisy knots
    assert np.all(np.diff(lut) >= -1e-6)
    assert lut[0] == pytest.approx(0, abs=1e-6) and lut[-1] == pytest.approx(100, abs=1e-6)


def test_hair_flecks_cleaned_lines_and_strokes_kept():
    rgba, parts, lines = doll()
    rgba = rgba.copy()
    H, W = rgba.shape[:2]
    cy, cx = int(0.12 * H), int(0.5 * W)
    fleck = np.zeros((H, W), bool)
    fleck[cy:cy + 3, cx - 30:cx - 27] = True                    # a 3x3 dark speck in the hair
    stroke = np.zeros((H, W), bool)
    stroke[cy - 20:cy + 20, cx + 20] = True                     # a long drawn strand, not in Adobe's paths
    adobe = np.zeros((H, W), bool)
    adobe[cy - 20:cy + 20, cx - 10] = True                      # a strand that IS an Adobe path
    for m in (fleck, stroke, adobe):
        rgba[m, :3] = (0.25, 0.18, 0.14)
    lines = np.maximum(lines, adobe.astype(np.float32))
    _, _, _, _, lab = BL._prep(rgba, parts, lines, dict(BL.BLEND))
    L0 = to_lab(rgba[..., :3])[..., 0]
    assert (lab[..., 0][fleck] - L0[fleck]).mean() > 30        # speck pulled to the hair around it
    assert np.abs(lab[..., 0][stroke] - L0[stroke]).max() < 1e-3
    assert np.abs(lab[..., 0][adobe] - L0[adobe]).max() < 1e-3
    assert np.abs(lab[..., 0][~parts["hair"]] - L0[~parts["hair"]]).max() < 1e-3   # nothing outside the hair


def test_downscale_ignores_transparent_colour():
    rgba, parts, lines = doll()
    junk = rgba.copy()
    junk[rgba[..., 3] == 0, :3] = (1.0, 0.0, 1.0)              # magenta under alpha 0
    a = BL.downscale(rgba, GAME)
    b = BL.downscale(junk, GAME)
    vis = a[..., 3] > 0
    assert np.abs(a[vis] - b[vis]).max() < 1e-5
