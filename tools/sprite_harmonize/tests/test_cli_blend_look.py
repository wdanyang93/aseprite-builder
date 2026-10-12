"""CLI smoke tests of blend-fit / blend / look-fit / look (relight.py) on synthetic dolls only.

Run: cd tools/sprite_harmonize && python -m pytest -q tests
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import relight  # noqa: E402
import test_blend as TB  # noqa: E402
import test_look as TL  # noqa: E402
from scenelight import blend as BL, look as LK  # noqa: E402
from scenelight.parts import load_lines, load_parts  # noqa: E402


def u8(x):
    return (np.clip(np.asarray(x, np.float32), 0, 1) * 255 + 0.5).astype(np.uint8)


def save_adobe(folder, parts, lines=None):
    """masks as Adobe would give them (white = selected) + a pre-rendered vectorize line art (dark paths on white)."""
    os.makedirs(folder, exist_ok=True)
    for k, m in parts.items():
        Image.fromarray(u8(m)).save(os.path.join(folder, k + ".png"))
    if lines is not None:
        Image.fromarray(u8(np.dstack([1 - lines] * 3))).save(os.path.join(folder, "vector.png"))


def thin_lines(rgba):
    """the doll's outline (2 px) + one interior stroke: thin enough to pass load_lines' width test."""
    sil = rgba[..., 3] >= 0.5
    d = cv2.distanceTransform(sil.astype(np.uint8), cv2.DIST_L2, 5)
    ln = (sil & (d <= 2)).astype(np.float32)
    ln[450:700, 299:301] = 1
    return ln


def test_blend_fit_and_blend(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rgba, parts, _ = TB.doll()
    lines = thin_lines(rgba)
    target, tparts, tsil = TB.painted_target(rgba, parts)
    Image.fromarray(u8(rgba)).save("frame.png")
    Image.fromarray(u8(target)).save("scene.png")
    save_adobe("masks/frame", parts, lines)
    save_adobe("masks/scene", dict(tparts, subject=tsil))
    other, oparts, _ = TB.doll(seed=3)
    Image.fromarray(u8(other)).save("other.png")
    Image.fromarray(u8(other)).save("unmasked.png")
    save_adobe("masks/other", {k: v for k, v in oparts.items() if k != "clothes"}, lines)   # sparse parts

    relight.main(["blend-fit", "frame.png", "scene.png", "-o", "blend.json", "--sheet", "check.png"])
    log = capsys.readouterr().out
    look = json.load(open("blend.json"))
    assert abs(look["height"] - TB.GAME) <= 3 and look["rows"] == rgba.shape[0]     # figure as tall as in the scene
    assert set(look["parts"]) == {"hair", "skin", "clothes", "tail", "_all"}
    assert "off by > 3" not in log and "line px lighter 0 of" in log and "alpha = plain downscale True" in log
    assert Image.open("check.png").size[1] == 720

    relight.main(["blend", "blend.json", "frame.png", "other.png", "unmasked.png", "-o", "out", "--sheet", "ba.png"])
    log = capsys.readouterr().out
    assert "unmasked.png: FAILED" in log and "1 frame(s) failed" in log                 # the other frames are kept
    assert "under 50 %" in log                                                           # sparse masks are flagged
    out = np.asarray(Image.open("out/frame.png"))
    assert out.shape[0] == look["height"] and os.path.isfile("out/other.png") and os.path.isfile("ba.png")
    f = np.asarray(Image.open("frame.png")).astype(np.float32) / 255
    assert np.array_equal(out[..., 3], u8(BL.downscale(f, look["height"])[..., 3]))      # alpha: plain downscale
    lines_png = load_lines("masks/frame/vector.png", f.shape)
    api = BL.apply_blend(f, load_parts("masks/frame"), look, look["height"], lines_png)
    assert np.abs(out.astype(int) - u8(api).astype(int)).max() <= 1                     # same as the Python API


def test_first_claim_makes_parts_disjoint():
    legs = np.zeros((10, 10), np.float32); legs[2:10] = 1
    shoes = np.zeros((10, 10), np.float32); shoes[7:10] = 1
    p = relight._first_claim({"shoes": shoes, "leftleg": legs, "subject": np.ones((10, 10), np.float32)})
    assert list(p) == ["shoes", "leftleg"]                                               # 'subject' dropped
    assert p["leftleg"][7:].max() == 0 and p["leftleg"][2:7].min() == 1                  # legs minus shoes


def test_look_fit_and_look(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    o, parts, _ = TL.doll()
    lines = np.zeros(o.shape[:2], np.float32)
    lines[:, 113:115] = 1
    lines[200, :] = 1
    Image.fromarray(u8(o)).save("orig.png")
    Image.fromarray(u8(TL.lit(o))).save("lamp.png")
    save_adobe("masks/orig", parts, lines)
    w, wparts, _ = TL.doll(shift=(14, 9))
    Image.fromarray(u8(w)).save("walk.png")
    save_adobe("masks/walk", wparts, lines)

    relight.main(["look-fit", "orig.png", "lamp.png", "-o", "look.json", "--sheet", "check.png"])
    log = capsys.readouterr().out
    look = json.load(open("look.json"))
    assert abs(((look["az"] - 180) + 180) % 360 - 180) < 20                             # lamp from screen-left
    assert "CHECK FAILED" not in log and "line px lighter 0 of" in log and os.path.isfile("check.png")
    o8, r8 = np.asarray(Image.open("orig.png")), np.asarray(Image.open("lamp.png"))
    api = LK.fit_look(o8, r8, load_parts("masks/orig"), load_lines("masks/orig/vector.png", o8.shape))
    assert json.loads(json.dumps(api)) == look                                           # one alignment, same look

    relight.main(["look", "look.json", "orig.png", "walk.png", "-o", "out", "--sheet", "ba.png"])
    log = capsys.readouterr().out
    assert log.count("alpha identical True, line px lighter 0") == 2 and "CHECK FAILED" not in log
    for name in ("orig", "walk"):
        src, out = np.asarray(Image.open(name + ".png")), np.asarray(Image.open(f"out/{name}.png"))
        held = (lines >= 0.5) & (src[..., 3] > 0)
        assert np.array_equal(out[..., 3], src[..., 3]) and not np.array_equal(out, src)
        assert not (out[..., :3][held].astype(int) > src[..., :3][held].astype(int)).any()


def test_opaque_reference_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    o, parts, _ = TL.doll()
    Image.fromarray(u8(o)).save("orig.png")
    Image.fromarray(u8(TL.lit(o))[..., :3]).save("lamp.png")                            # RGB: no cut-out
    save_adobe("masks/orig", parts, np.zeros(o.shape[:2], np.float32))
    with pytest.raises(SystemExit, match="image_remove_background"):
        relight.main(["look-fit", "orig.png", "lamp.png", "-o", "look.json"])
