"""Adobe part masks and line art for blend / look. Parts and lines come ONLY from Adobe, never from our own detectors:
part masks = image_select_subject bodyParts / image_select_by_prompt, line art = the thin dark paths of image_vectorize.

    parts = load_parts("masks/frame_a")                  # {name: HxW float32 0..1}, one <name>.png per part
    lines = load_lines("masks/frame_a/vector.svg", rgba.shape)    # HxW bool

Naming: one folder per frame (or one folder and a per-frame `prefix`, e.g. prefix="a_" for a_hair.png), file stem =
part name: hair, face, skin, neck, hands, upper, lower, coat, dress, shoes, legs, leftleg, rightleg, lefthand,
righthand, tail, subject. Other names load too. `<part>_<anything>.png` joins <part> (union): upper_shirt.png +
upper_sleeves.png -> upper. Masks: 8/16-bit grey, RGB or RGBA PNGs, white = selected (an RGBA cutout: its alpha).
"""
import io
import os

import cv2
import numpy as np
from PIL import Image

from .color import to_lab

NAMES = ("hair", "face", "skin", "neck", "hands", "upper", "lower", "coat", "dress", "shoes", "legs", "leftleg",
         "rightleg", "lefthand", "righthand", "tail", "subject")

LINES = {
    "dark_L": 50.0,           # a vectorize path darker than this L* is a line candidate ...
    "max_r": 3.0 / 1536,      # ... and stays one unless a pixel within max_r of it lies deeper than max_r inside the
                              # dark (fraction of the frame height: 3 px on a 1536-row frame); such wide dark shapes
                              # are cloth shadow / fur stripes, not line art
    "min_r_px": 1.5,          # never below this (px): small frames keep their 1-3 px lines
}


def _read_mask(path):
    """float32 HxW 0..1 from a mask PNG (grey / RGB: brightest channel; RGBA with transparency: the alpha)."""
    m = np.asarray(Image.open(path))
    m = m.astype(np.float32) / (65535.0 if m.dtype == np.uint16 else 255.0 if m.dtype == np.uint8 else 1.0)
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 and m[..., 3].min() < 1.0 else m[..., :3].max(-1)
    return np.clip(m, 0, 1)


def load_parts(folder, prefix=""):
    """{name: float32 HxW mask} for every <prefix><name>.png in `folder`; <prefix><name>_<x>.png files of a NAMES part
    join it (pixel-wise max). Order: NAMES order, then other names alphabetically, 'subject' (the whole figure) last,
    so a first-claim consumer gives it only unclaimed pixels."""
    found = {}
    for f in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(f)
        if ext.lower() == ".png" and stem.startswith(prefix) and len(stem) > len(prefix):
            name = stem[len(prefix):].lower()
            group = name.split("_", 1)[0]
            found.setdefault(group if group in NAMES and group != name else name, []).append(os.path.join(folder, f))
    order = [n for n in NAMES[:-1] if n in found] + sorted(n for n in found if n not in NAMES)
    order += ["subject"] if "subject" in found else []
    return {n: np.max([_read_mask(p) for p in found[n]], 0) for n in order}


def _render(path, shape):
    """RGB float of an SVG (cairosvg) or PNG at `shape`: the paths' own (straight) colours, white where the render is
    mostly transparent. Not composited: the semi-transparent seams between abutting paths keep their path colour."""
    h, w = shape[:2]
    if path.lower().endswith(".svg"):
        try:
            import cairosvg
        except ImportError as e:
            raise ImportError("load_lines needs cairosvg for an SVG (pip install cairosvg) or a pre-rendered PNG") from e
        with open(path, "rb") as f:
            im = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=f.read(), output_width=w, output_height=h)))
    else:
        im = Image.open(path)
    x = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0
    if x.shape[:2] != (h, w):
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
    return np.where(x[..., 3:] >= 0.5, x[..., :3], 1.0)


def load_lines(path, shape, params=None):
    """Adobe line art as an HxW bool mask: the dark paths (L* < dark_L) of an image_vectorize SVG (rendered with
    cairosvg) or of its pre-rendered PNG, minus wide dark shapes (width test on Adobe's own shapes)."""
    P = dict(LINES, **(params or {}))
    dark = to_lab(_render(path, shape))[..., 0] < P["dark_L"]
    r = max(P["min_r_px"], P["max_r"] * shape[0])
    dt = cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 5)
    k = 2 * int(round(r)) + 1
    wide = cv2.dilate(dt, np.ones((k, k), np.uint8)) > r
    return dark & ~wide
