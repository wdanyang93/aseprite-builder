"""Adobe mask / line-art loaders (scenelight.parts) on synthetic files only.

Run: cd tools/sprite_harmonize && python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scenelight import parts as PT  # noqa: E402


def _disc(h=60, w=40, r=12):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.hypot(yy - h / 2, xx - w / 2) < r


def test_load_parts_formats_order_and_prefix(tmp_path):
    m = _disc()
    u8 = (m * 255).astype(np.uint8)
    Image.fromarray(u8).save(tmp_path / "subject.png")                                   # grey
    Image.fromarray(u8).save(tmp_path / "tail.png")
    Image.fromarray(np.dstack([u8] * 3)).save(tmp_path / "Hair.png")                   # RGB, upper-case stem
    cut = np.dstack([np.full(m.shape + (3,), 90, np.uint8), u8])                      # RGBA cutout: alpha is the mask
    Image.fromarray(cut, "RGBA").save(tmp_path / "upper.png")
    opq = np.dstack([u8] * 3 + [np.full(m.shape, 255, np.uint8)])                      # RGBA, opaque: RGB is the mask
    Image.fromarray(opq, "RGBA").save(tmp_path / "shoes.png")
    Image.fromarray((m * 65535).astype(np.uint16)).save(tmp_path / "belt.png")        # 16-bit grey, extra name
    Image.fromarray(u8).save(tmp_path / "apron.png")
    (tmp_path / "notes.txt").write_text("not a mask")
    P = PT.load_parts(str(tmp_path))
    assert list(P) == ["hair", "upper", "shoes", "tail", "apron", "belt", "subject"]   # NAMES, extras, subject last
    for k, v in P.items():
        assert v.dtype == np.float32 and v.shape == m.shape, k
        assert np.array_equal(v >= 0.5, m) and v.min() == 0.0 and v.max() == 1.0, k

    # one folder for several frames: a per-frame prefix
    Image.fromarray(u8).save(tmp_path / "a_hair.png")
    Image.fromarray(255 - u8).save(tmp_path / "a_tail.png")
    Image.fromarray(u8).save(tmp_path / "b_hair.png")
    A = PT.load_parts(str(tmp_path), prefix="a_")
    assert list(A) == ["hair", "tail"] and np.array_equal(A["tail"] >= 0.5, ~m)
    assert PT.load_parts(str(tmp_path), prefix="zz_") == {}


def _drawing(h, w):
    """White canvas (RGBA): thin dark strokes 2 / 6 / 10 px wide, a wide dark blob, a mid-grey (not dark) block,
    a transparent black hole (not dark) and a dark stroke drawn semi-transparent (an anti-aliased path seam)."""
    x = np.ones((h, w, 4), np.float32)
    x[10:-10, 4:6, :3] = 0.1          # 2 px stroke
    x[10:-10, 12:18, :3] = 0.1        # 6 px
    x[10:-10, 24:34, :3] = 0.1        # 10 px
    x[20:60, 40:80, :3] = 0.05        # wide blob (cloth shadow)
    x[80:120, 40:80, :3] = 0.55       # mid grey, L* ~ 58: not dark
    x[130:170, 40:80] = 0.0           # fully transparent, RGB black
    x[10:-10, 90:92, :3] = 0.1
    x[10:-10, 90:92, 3] = 0.75        # seam alpha
    return x


def _truth(shape):
    t = np.zeros(shape, bool)
    for c0, c1 in ((4, 6), (12, 18), (90, 92)):
        t[10:-10, c0:c1] = True
    return t


def test_load_lines_png_width_test(tmp_path):
    h, w = 1536, 100
    x = _drawing(h, w)
    Image.fromarray((x * 255 + 0.5).astype(np.uint8), "RGBA").save(tmp_path / "v.png")
    L = PT.load_lines(str(tmp_path / "v.png"), (h, w, 4))
    assert L.dtype == bool and L.shape == (h, w)
    # 2 and 6 px strokes and the seam stroke are lines; the 10 px stroke, the blob, grey and transparent areas are not
    assert np.array_equal(L, _truth((h, w)))
    # the width limit is a fraction of the frame height: at twice the height the 10 px stroke is a line too
    L2 = PT.load_lines(str(tmp_path / "v.png"), (2 * h, w))      # PNG resized to the frame: rows doubled
    assert L2[2 * 100, 24:34].all() and not L2[2 * 40, 40:80].any()
    # small frames keep 1-2 px lines (floor min_r_px) and drop a 4 px wide one
    s = np.ones((96, 40, 4), np.float32)
    s[10:-10, 4, :3] = 0.1
    s[10:-10, 10:12, :3] = 0.1
    s[10:-10, 20:24, :3] = 0.1
    Image.fromarray((s * 255 + 0.5).astype(np.uint8), "RGBA").save(tmp_path / "s.png")
    Ls = PT.load_lines(str(tmp_path / "s.png"), (96, 40))
    assert Ls[10:-10, 4].all() and Ls[10:-10, 10:12].all() and not Ls[:, 20:24].any()


def test_load_lines_svg_matches_png(tmp_path):
    pytest.importorskip("cairosvg")
    h, w = 1536, 100
    rects = [(4, 10, 2, h - 20), (12, 10, 6, h - 20), (24, 10, 10, h - 20), (40, 20, 40, 40)]
    body = "".join('<path d="M%d,%d h%d v%d h%d Z" fill="#1a1a1a"/>' % (x0, y0, ww, hh, -ww) for x0, y0, ww, hh in rects)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (w, h) +
           '<path d="M0,0 h%d v%d h%d Z" fill="#ffffff"/>' % (w, h, -w) +
           '<path d="M40,80 h40 v40 h-40 Z" fill="#8c8c8c"/>' + body + "</svg>")
    (tmp_path / "v.svg").write_text(svg)
    L = PT.load_lines(str(tmp_path / "v.svg"), (h, w))
    t = np.zeros((h, w), bool)
    t[10:-10, 4:6] = t[10:-10, 12:18] = True
    assert np.array_equal(L, t)
    # rendered at another size: still the same strokes, scaled
    Lh = PT.load_lines(str(tmp_path / "v.svg"), (h // 2, w // 2))
    assert Lh[400, 2:3].all() and not Lh[20, 20:40].any()


def test_load_lines_svg_needs_cairosvg(tmp_path, monkeypatch):
    (tmp_path / "v.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>')
    monkeypatch.setitem(sys.modules, "cairosvg", None)          # import fails as if not installed
    with pytest.raises(ImportError, match="cairosvg"):
        PT.load_lines(str(tmp_path / "v.svg"), (4, 4))
