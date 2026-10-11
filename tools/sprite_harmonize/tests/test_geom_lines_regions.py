"""Invariants for the per-frame analysis (geom / lines / regions / character card) on synthetic dolls only.

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
from scenelight import lines as LN  # noqa: E402
from scenelight import regions as RG  # noqa: E402
from scenelight import character as CH  # noqa: E402
from scenelight.color import to_lab  # noqa: E402

SKIN = (0.97, 0.80, 0.70)
DARKSKIN = (0.45, 0.30, 0.22)
NAVY = (0.10, 0.12, 0.33)
BLACK = (0.06, 0.06, 0.07)
WHITE = (0.93, 0.93, 0.91)
BROWN_LINE = (0.55, 0.40, 0.33)   # L* ~47: the "brown line" style the old absolute threshold never found
INK = (0.20, 0.13, 0.11)


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


def doll(h=600, skin=SKIN, cloth=NAVY, line=BROWN_LINE, outline=INK, thick=None):
    """Standing doll: head, torso in a wide cloth dress, arms, legs, a thin tail. Drawn lines: silhouette outline
    (ink), interior strokes (line colour): dress hem, belt, arm-over-dress contour, two fingers. Returns (rgba, line
    truth coverage HxW)."""
    s = h / 600.0
    w = int(420 * s)
    H = h + int(40 * s)
    a = np.zeros((H, w), np.float32)
    cx = w // 2
    P = lambda *v: tuple(int(round(t * s)) for t in v)
    cv2.circle(a, (cx, P(70)[0]), P(45)[0], 1, -1, cv2.LINE_AA)
    cv2.ellipse(a, (cx, P(230)[0]), P(70, 120), 0, 0, 360, 1, -1, cv2.LINE_AA)
    cv2.rectangle(a, (cx - P(55)[0], P(330)[0]), (cx - P(15)[0], P(620)[0]), 1, -1, cv2.LINE_AA)
    cv2.rectangle(a, (cx + P(15)[0], P(330)[0]), (cx + P(55)[0], P(620)[0]), 1, -1, cv2.LINE_AA)
    cv2.rectangle(a, (cx - P(100)[0], P(140)[0]), (cx - P(78)[0], P(350)[0]), 1, -1, cv2.LINE_AA)
    cv2.rectangle(a, (cx + P(78)[0], P(140)[0]), (cx + P(100)[0], P(350)[0]), 1, -1, cv2.LINE_AA)
    cv2.line(a, (cx + P(60)[0], P(300)[0]), (cx + P(190)[0], P(420)[0]), 1, P(9)[0], cv2.LINE_AA)   # thin tail
    a = np.clip(a, 0, 1)
    rgb = np.zeros((H, w, 3), np.float32)
    rgb[:] = skin
    dress = np.zeros((H, w), np.float32)
    cv2.ellipse(dress, (cx, P(250)[0]), P(70, 100), 0, 0, 360, 1, -1, cv2.LINE_AA)
    rgb = rgb * (1 - dress[..., None]) + np.array(cloth, np.float32) * dress[..., None]
    d = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    rgb *= (0.88 + 0.12 * np.clip(d / (20 * s), 0, 1))[..., None]    # soft painted shading
    lw = max(1, int(round(2 * s)))
    strokes = np.zeros((H, w), np.float32)
    cv2.ellipse(strokes, (cx, P(250)[0]), P(70, 100), 0, 20, 160, 1, lw, cv2.LINE_AA)        # hem
    cv2.line(strokes, (cx - P(60)[0], P(210)[0]), (cx + P(60)[0], P(210)[0]), 1, lw, cv2.LINE_AA)   # belt
    cv2.line(strokes, (cx - P(95)[0], P(330)[0]), (cx - P(95)[0], P(348)[0]), 1, lw, cv2.LINE_AA)   # fingers
    cv2.line(strokes, (cx - P(85)[0], P(330)[0]), (cx - P(85)[0], P(348)[0]), 1, lw, cv2.LINE_AA)
    cv2.line(strokes, (cx - P(40)[0], P(380)[0]), (cx - P(30)[0], P(450)[0]), 1, lw, cv2.LINE_AA)   # knee mark
    out = np.zeros((H, w), np.float32)
    cnts = cv2.findContours((a > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(out, cnts, -1, 1, lw, cv2.LINE_AA)
    out *= (a > 0.5)
    strokes = np.clip(strokes, 0, 1) * (a > 0.5)
    rgb = rgb * (1 - strokes[..., None]) + np.array(line, np.float32) * strokes[..., None]
    rgb = rgb * (1 - out[..., None]) + np.array(outline, np.float32) * out[..., None]
    return np.dstack([np.clip(rgb, 0, 1), a]).astype(np.float32), np.maximum(strokes, out)


def downscale(f, t, h):
    """Premultiplied area downscale to figure height h (like a game-size frame), truth coverage the same way."""
    H = G.figure_height(f[..., 3])
    k = h / H
    sz = (max(1, round(f.shape[1] * k)), max(1, round(f.shape[0] * k)))
    pm = cv2.resize(np.dstack([f[..., :3] * f[..., 3:4], f[..., 3:4]]), sz, interpolation=cv2.INTER_AREA)
    al = np.clip(pm[..., 3], 0, 1)
    rgb = np.clip(pm[..., :3] / np.maximum(al[..., None], 1e-4), 0, 1)
    tt = cv2.resize(t * f[..., 3], sz, interpolation=cv2.INTER_AREA) / np.maximum(cv2.resize(f[..., 3], sz, interpolation=cv2.INTER_AREA), 1e-4)
    return np.dstack([rgb, al]).astype(np.float32), np.clip(tt, 0, 1)


def soft_pr(w, t, fig):
    w, t = w * fig, np.clip(t, 0, 1) * fig
    i = np.minimum(w, t).sum()
    return i / max(w.sum(), 1e-6), i / max(t.sum(), 1e-6)


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
    f, _ = doll(600)
    g = G.analyze(f[..., 3])
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
        fs, _ = downscale(f, np.zeros(f.shape[:2], np.float32), h)
        gs = G.analyze(fs[..., 3])
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
def test_lines_brown_strokes_found_navy_cloth_not_line():
    f, t = doll(600)
    lm = LN.line_map(f)
    fig = f[..., 3] > 0.5
    P, R = soft_pr(lm["weight"], t, fig)
    assert R > 0.75 and P > 0.75
    lab = to_lab(f[..., :3])
    navy = fig & (np.abs(lab[..., 2] + 20) < 15) & (t < 0.05)
    navy = cv2.erode(navy.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert navy.sum() > 3000
    assert lm["mask"][navy].mean() < 0.02          # a wide dark garment is not line art
    for clo, skin in ((BLACK, SKIN), (NAVY, DARKSKIN), (WHITE, DARKSKIN)):
        f2, t2 = doll(600, skin=skin, cloth=clo)
        lm2 = LN.line_map(f2)
        P2, R2 = soft_pr(lm2["weight"], t2, fig)
        assert R2 > 0.6 and P2 > 0.6, (clo, skin, P2, R2)


def test_lines_survive_small_sizes_and_resampling():
    f, t = doll(600)
    card = {"ink_L": LN.line_map(f)["ink_L"], "width": 2 / 600.0}
    t_ratio = (t * (f[..., 3] > 0.5)).sum() / (f[..., 3] > 0.5).sum()
    for h in (192, 96, 48):
        fs, ts = downscale(f, t, h)
        fig = fs[..., 3] > 0.5
        lm = LN.line_map(fs, p=card)
        ratio = lm["weight"][fig].mean()
        truth = ts[fig].mean()
        assert 0.4 * truth < ratio < 2.5 * truth, (h, ratio, truth, t_ratio)
        P, R = soft_pr(lm["weight"], ts, fig)
        assert R > 0.4, (h, R)
    # a slightly blurred / shifted copy (resampled frame) keeps its decision at full size
    M = np.float32([[1, 0, 0.5], [0, 1, 0.3]])
    fw = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), flags=cv2.INTER_LINEAR)
    tw = cv2.warpAffine(t, M, (f.shape[1], f.shape[0]), flags=cv2.INTER_LINEAR)
    P, R = soft_pr(LN.line_map(fw, p=card)["weight"], tw, fw[..., 3] > 0.5)
    assert R > 0.7 and P > 0.7


def test_line_map_deterministic_and_alpha_free():
    f, _ = doll(300)
    a1, a2 = LN.line_map(f), LN.line_map(f.copy())
    assert np.array_equal(a1["weight"], a2["weight"])
    assert not (a1["weight"][f[..., 3] <= 0.5] > 0).any()


# ------------------------------------------------------------------ regions / palette
def test_palette_regions_and_assignment():
    f, t = doll(600)
    H = G.figure_height(f[..., 3])
    lm = LN.line_map(f)
    pal = RG.fit_palette(RG.palette_pixels(f, lm["weight"], H))
    lab = np.array(pal["centres"])
    assert 2 <= len(lab) <= 5
    navy_l = to_lab(np.array([[NAVY]], np.float32))[0, 0]
    skin_l = to_lab(np.array([[SKIN]], np.float32))[0, 0]
    dn = np.linalg.norm(lab[:, 1:] - navy_l[1:], axis=1).min()
    ds = np.linalg.norm(lab[:, 1:] - skin_l[1:], axis=1).min()
    assert dn < 8 and ds < 8
    seg = RG.segment(f, pal["centres"], lm["weight"], H)
    fig = f[..., 3] > 0.5
    assert (seg["labels"][fig] > 0).all() and (seg["labels"][~fig] == 0).all()
    assert 3 <= seg["n"] <= 30
    # the dress is one region in the navy colour, cut from the skin by the drawn hem / belt / outline
    cx = f.shape[1] // 2
    r_dress = seg["labels"][260, cx]
    assert seg["labels"][300, cx] == r_dress
    assert seg["labels"][120, cx] != r_dress             # chest skin above the belt
    # an unseen colour (saturated red) is flagged -1 against the card palette
    g = f.copy()
    g[250:270, cx - 10:cx + 10, :3] = (0.9, 0.05, 0.05)
    idx, dist = RG.assign(g, pal["centres"])
    assert (idx[255:265, cx - 5:cx + 5] == -1).all()
    assert (idx[~fig] == -1).all()


def test_texture_oriented_detail_only():
    h, w = 300, 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.full((h, w, 3), 0.8, np.float32)
    rng = np.random.default_rng(0)
    strands = 0.8 + 0.12 * np.sin(xx * 1.3 + 0.2 * np.sin(yy * 0.05)) * (rng.random((h, w)) * 0.3 + 0.7)
    rgb[:, :100] = strands[:, :100, None]                       # fur-like: oriented fine strokes
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.0)
    rgb[:, 200:] = (0.8 + 0.12 * noise / noise.std())[:, 200:, None]   # isotropic print / noise
    f = np.dstack([np.clip(rgb, 0, 1), np.ones((h, w), np.float32)])
    tw, tm = RG.texture(f, H=300.0)
    assert tm[50:250, 20:80].mean() > 0.9
    assert tm[50:250, 120:180].mean() < 0.05
    assert tm[50:250, 220:280].mean() < 0.2


# ------------------------------------------------------------------ character card
def test_character_card_roundtrip_and_frame_analysis(tmp_path):
    f, _ = doll(500)
    card = CH.build_card([f], name="doll")
    assert card["palette"]["lab"] and 0.003 <= card["lines"]["scale"] <= 0.005
    assert card["lines"]["ink_L"] is not None and card["lines"]["ink_L"] < 60
    assert card["gate"]["calibrated"] is False and card["gate"]["default"] == 0.68
    folder = tmp_path / "chars" / "doll"
    CH.save_card(str(folder), card)
    assert (folder / "card.json").exists() and (folder / "maps.npz").exists()
    back = CH.load_card(str(folder), maps=True)
    assert back["palette"]["lab"] == card["palette"]["lab"]
    assert np.array_equal(back["_maps"][0]["labels"], card["_maps"][0]["labels"])
    fs, _ = downscale(f, np.zeros(f.shape[:2], np.float32), 96)
    fa = CH.analyze_frame(fs, back)
    assert fa["new_colour_share"] < 0.05
    assert fa["lines"]["weight"].shape == fs.shape[:2]
    # cards hold data derived from the user's art: never inside the (public) repository
    with pytest.raises(ValueError):
        CH.save_card(os.path.join(ROOT, "tests", "_card_should_not_exist"), card)
    assert not os.path.exists(os.path.join(ROOT, "tests", "_card_should_not_exist"))
