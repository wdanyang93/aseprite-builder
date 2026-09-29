#!/usr/bin/env python3
"""Relight 2D character sprites so they belong in a scene.

  fit      neutral frame + hand-lit reference of the same pose  ->  scene profile (JSON, ~2 KB)
  apply    scene profile + any frames (any pose / direction)     ->  relit frames (RGBA PNG)
  preview  relit (or neutral) sprite on a background with contact shadow and light wrap
  check    quality sheet + numbers for a scene profile (neutral | relit | reference)

Examples
  python relight.py fit neutral.png warm_reference.png -o scenes/forest_sunset.json --sheet check.png
  python relight.py apply scenes/forest_sunset.json frames/*.png -o out/ --view-from-name --sheet ba.png
  python relight.py preview scenes/forest_sunset.json frames/walk_03.png --bg bg.png --place 640,602,240 -o shot.png
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenelight import fit_scene, load_profile, relight, save_profile  # noqa: E402
from scenelight.color import defringe, load_rgb, load_rgba, save_rgb, save_rgba  # noqa: E402
from scenelight.composite import composite  # noqa: E402
from scenelight import report  # noqa: E402
from scenelight.shading import VIEWS  # noqa: E402

# file-name tokens -> view hint (8-direction sprite conventions: S = facing camera)
VIEW_ALIASES = {
    "front": "front", "s": "front", "south": "front", "down": "front", "정면": "front", "남": "front",
    "front_left": "front_left", "sw": "front_left", "southwest": "front_left", "남서": "front_left",
    "front_right": "front_right", "se": "front_right", "southeast": "front_right", "남동": "front_right",
    "left": "left", "w": "left", "west": "left", "서": "left",
    "right": "right", "e": "right", "east": "right", "동": "right",
    "back": "back", "n": "back", "north": "back", "up": "back", "뒤": "back", "북": "back",
    "back_left": "back_left", "nw": "back_left", "northwest": "back_left", "북서": "back_left",
    "back_right": "back_right", "ne": "back_right", "northeast": "back_right", "북동": "back_right",
    "walk34": "walk34",
}


def view_from_name(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    toks = [t for t in re.split(r"[^0-9a-z가-힣]+", stem) if t]
    # two-token forms first (front_left, back_right ...)
    for a, b in zip(toks, toks[1:]):
        if f"{a}_{b}" in VIEW_ALIASES:
            return VIEW_ALIASES[f"{a}_{b}"]
    for t in toks:
        if t in VIEW_ALIASES:
            return VIEW_ALIASES[t]
    return None


def load_sprite(path, do_defringe=True):
    f = load_rgba(path)
    if do_defringe:
        f[..., :3] = defringe(f[..., :3], f[..., 3])
    return f


def cmd_fit(a):
    n, r = load_sprite(a.neutral, not a.no_defringe), load_sprite(a.reference, not a.no_defringe)
    t = time.time()
    prof, pairs = fit_scene(n, r, a.view)
    save_profile(a.out, prof)
    print(f"scene profile -> {a.out}  ({time.time() - t:.0f}s, alignment IoU {pairs.iou:.3f}, {len(pairs.regions)} regions)")
    if pairs.iou < 0.85:
        print("WARNING: neutral and reference line up poorly (IoU < 0.85). Use the same pose / framing for both.")
    relit = relight(n, prof, a.view)
    m = report.metrics(relit, pairs)
    print("shape agreement {shape_agreement:.3f} (neutral {shape_agreement_neutral:.3f})  "
          "colour error {color_error_lowfreq:.2f}  mean L* {mean_L:.1f} (reference {mean_L_reference:.1f})".format(**m))
    if a.sheet:
        report.sheet(relit, pairs, a.sheet)
        print("sheet ->", a.sheet)


def cmd_apply(a):
    prof = load_profile(a.scene)
    multi = len(a.frames) > 1 or os.path.isdir(a.out) or a.out.endswith(os.sep)
    if multi:
        os.makedirs(a.out, exist_ok=True)
    ins, outs = [], []
    for p in a.frames:
        view = view_from_name(p) if a.view_from_name else a.view
        if a.view_from_name and view is None:
            view = a.view
        f = load_sprite(p, not a.no_defringe)
        t = time.time()
        o = relight(f, prof, view)
        dst = os.path.join(a.out, os.path.splitext(os.path.basename(p))[0] + ".png") if multi else a.out
        save_rgba(dst, o)
        print(f"{p} -> {dst}  view={view}  {time.time() - t:.1f}s")
        if a.sheet:
            ins.append(f); outs.append(o)
    if a.sheet:
        report.before_after(ins, outs, a.sheet)
        print("before|after sheet ->", a.sheet)


def cmd_preview(a):
    bg = load_rgb(a.bg)
    f = load_sprite(a.sprite, not a.no_defringe)
    if not a.already_relit:
        f = relight(f, load_profile(a.scene), a.view)
    cx, fy, hh = (float(v) for v in a.place.split(","))
    out = composite(bg, f, cx, fy, int(hh), a.shadow, a.wrap, a.dim)
    save_rgb(a.out, out)
    print("preview ->", a.out)


def cmd_check(a):
    from scenelight.align import Pairs
    prof = load_profile(a.scene)
    n, r = load_sprite(a.neutral, not a.no_defringe), load_sprite(a.reference, not a.no_defringe)
    pairs = Pairs(n, r)
    relit = relight(n, prof, a.view)
    m = report.metrics(relit, pairs)
    for k, v in m.items():
        print(f"{k:28s} {v:.3f}" if isinstance(v, float) else f"{k:28s} {v}")
    report.sheet(relit, pairs, a.out)
    print("sheet ->", a.out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    views = f"view hint: {', '.join(VIEWS)}"

    f = sub.add_parser("fit", help="learn a scene light from neutral + hand-lit reference (same pose)")
    f.add_argument("neutral"); f.add_argument("reference")
    f.add_argument("-o", "--out", required=True, help="scene profile JSON")
    f.add_argument("--view", default="walk34", help=views)
    f.add_argument("--sheet", help="write a check sheet PNG")

    ap = sub.add_parser("apply", help="relight frames with a scene profile")
    ap.add_argument("scene"); ap.add_argument("frames", nargs="+")
    ap.add_argument("-o", "--out", required=True, help="output PNG (one frame) or directory")
    ap.add_argument("--view", default=None, help=views)
    ap.add_argument("--view-from-name", action="store_true",
                    help="take the view from file-name tokens (front/s, se, e/right, back/n ...)")
    ap.add_argument("--sheet", help="write a before|after contact sheet PNG")

    pv = sub.add_parser("preview", help="put a sprite on a background (contact shadow, light wrap)")
    pv.add_argument("scene"); pv.add_argument("sprite")
    pv.add_argument("--bg", required=True)
    pv.add_argument("--place", required=True, help="cx,foot_y,height in background pixels")
    pv.add_argument("-o", "--out", required=True)
    pv.add_argument("--view", default=None, help=views)
    pv.add_argument("--already-relit", action="store_true", help="sprite is already relit; only composite")
    pv.add_argument("--shadow", type=float, default=0.55, help="contact shadow opacity")
    pv.add_argument("--wrap", type=float, default=0.35, help="light wrap amount")
    pv.add_argument("--dim", type=float, default=0.95, help="sprite brightness in the scene")

    c = sub.add_parser("check", help="quality sheet + numbers for a profile")
    c.add_argument("scene"); c.add_argument("neutral"); c.add_argument("reference")
    c.add_argument("-o", "--out", required=True, help="sheet PNG")
    c.add_argument("--view", default="walk34", help=views)

    for sp_ in (f, ap, pv, c):
        sp_.add_argument("--no-defringe", action="store_true", help="keep semi-transparent edge colours as they are")
    a = p.parse_args(argv)
    {"fit": cmd_fit, "apply": cmd_apply, "preview": cmd_preview, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
