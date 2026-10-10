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

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenelight import fit_scene, load_profile, relight, save_profile  # noqa: E402
from scenelight.color import defringe, load_rgb, load_rgba, save_rgb, save_rgba  # noqa: E402
from scenelight.composite import composite  # noqa: E402
from scenelight import report, transfer  # noqa: E402
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
          "drawn form kept {drawn_form_kept:.3f}  colour error {color_error_lowfreq:.2f}  "
          "mean L* {mean_L:.1f} (reference {mean_L_reference:.1f})".format(**m))
    if m["drawn_form_kept"] < 0.95:
        print("WARNING: the relit chest shading departs from the drawing (drawn form kept < 0.95) - check the sheet.")
    if a.sheet:
        report.sheet(relit, pairs, a.sheet)
        print("sheet ->", a.sheet)
    if a.pack:
        pk = transfer.build_pack(n, r, prof)
        transfer.save_pack(a.pack, pk)
        lp = transfer.load_pack(a.pack)
        soft, info = transfer.relight_transfer(n, prof, lp, a.view, chest=False)
        tr, cinfo = transfer.relight_transfer(n, prof, lp, a.view)
        ms, mt = report.metrics(soft, pairs), report.metrics(tr, pairs)
        print(f"reference pack -> {a.pack}  (contains your art: keep it out of public repos)")
        print("  with pack: shape agreement {shape_agreement:.3f}  colour error {color_error_lowfreq:.2f}  "
              "mean L* {mean_L:.1f}".format(**mt) + f"  reference coverage {info['coverage']:.2f}")
        print("  soft transfer (body): drawn form kept {:.3f}".format(ms["drawn_form_kept"]))
        if cinfo.get("chest") == "ok":
            print("  chest stage: on - the breasts take the reference's light, cut by the drawn lines "
                  "(drawn form kept {:.3f} there by design; line art is never lightened)".format(mt["drawn_form_kept"]))
        else:
            print(f"  chest stage: skipped ({cinfo.get('chest', 'frame failed the gates')})")
        if not info["passed"]:
            print("WARNING: the neutral itself does not pass the pack gates - the pack would never be used.")
        if ms["drawn_form_kept"] < 0.95:
            print("WARNING: with the pack the body shading departs from the drawing (drawn form kept < 0.95).")
        if a.sheet:
            root, ext = os.path.splitext(a.sheet)
            report.sheet(tr, pairs, root + "_pack" + ext)
            print("sheet ->", root + "_pack" + ext)


def _gates(a):
    return {"max_fold": a.max_fold, "min_match": a.min_match, "min_iou_affine": a.min_iou_affine}


def _why(info):
    return f"fold {info.get('fold', 1):.4f}, match {info.get('match', 0):.2f}, iou {info.get('iou_affine', 0):.2f}"


def relight_one(f, prof, pack, view, a):
    """Reference look when a pack is given and the frame passes the gates, else wash+rim."""
    if pack is None:
        return relight(f, prof, view), "wash+rim"
    o, info = transfer.relight_transfer(f, prof, pack, view, _gates(a), chest=not a.no_chest)
    if info["passed"]:
        return o, f"reference {info['coverage']:.0%} ({_why(info)}){_chest(info)}"
    return o, f"wash+rim fallback ({_why(info)})"


def _chest(info):
    c = info.get("chest")
    return "" if c is None else ("  chest light: on" if c == "ok" else f"  chest light: skipped ({c})")


def _thumb(x, h=480):
    s = h / x.shape[0]
    return cv2.resize(x, (max(1, int(x.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def cmd_apply(a):
    prof = load_profile(a.scene)
    multi = len(a.frames) > 1 or os.path.isdir(a.out) or a.out.endswith(os.sep)
    if multi:
        os.makedirs(a.out, exist_ok=True)
    pack = transfer.load_pack(a.pack) if a.pack else None   # fails fast on a missing / old / damaged pack
    jobs = []
    for p in a.frames:
        view = view_from_name(p) if a.view_from_name else a.view
        jobs.append({"p": p, "view": a.view if (a.view_from_name and view is None) else view})

    use_ref = {}
    if pack is not None:
        # pass 1: alignment + gates only (fast). A clip is one view group; the group can only DOWNGRADE:
        # frames that fail their own gates never get the reference, and if too few frames of a group pass,
        # the whole group gets wash+rim so one animation never flickers between the two looks.
        for j in jobs:
            try:
                j["info"] = transfer.gate(load_sprite(j["p"], not a.no_defringe), pack, _gates(a))
            except Exception as e:  # unreadable frame etc.: handled again in pass 2
                j["info"] = {"passed": False, "error": f"{e.__class__.__name__}: {e}"}
        groups = {}
        for j in jobs:
            groups.setdefault(j["view"], []).append(j)
        for view, js in groups.items():
            share = sum(j["info"]["passed"] for j in js) / len(js)
            ok = a.per_frame or share >= a.group_ratio
            if not a.per_frame:
                print(f"view={view}: {share:.0%} of {len(js)} frames pass the pack gates -> "
                      f"{'reference look for the passing frames' if ok else 'wash+rim for the whole group'}")
            for j in js:
                use_ref[j["p"]] = ok and j["info"]["passed"]

    # pass 2: render and save each frame immediately (a failure never loses the frames already done)
    thumbs_in, thumbs_out, failed = [], [], 0
    for j in jobs:
        p, view = j["p"], j["view"]
        dst = os.path.join(a.out, os.path.splitext(os.path.basename(p))[0] + ".png") if multi else a.out
        t = time.time()
        try:
            f = load_sprite(p, not a.no_defringe)
            if use_ref.get(p):
                o, info = transfer.relight_transfer(f, prof, pack, view, _gates(a), chest=not a.no_chest)
                how = ((f"reference {info['coverage']:.0%}" if info["passed"] else "wash+rim fallback")
                       + f" ({_why(info)})" + (_chest(info) if info["passed"] else ""))
            else:
                o = relight(f, prof, view)
                how = "wash+rim" if pack is None else f"wash+rim fallback ({_why(j['info'])})"
            save_rgba(dst, o)
        except Exception as e:
            failed += 1
            print(f"{p}: FAILED ({e.__class__.__name__}: {e}) - skipped")
            continue
        print(f"{p} -> {dst}  view={view}  {how}  {time.time() - t:.1f}s")
        if a.sheet:
            thumbs_in.append(_thumb(f)); thumbs_out.append(_thumb(o))
    if a.sheet and thumbs_in:
        report.before_after(thumbs_in, thumbs_out, a.sheet)
        print("before|after sheet ->", a.sheet)
    if failed:
        print(f"{failed} frame(s) failed - see above")


def cmd_preview(a):
    bg = load_rgb(a.bg)
    f = load_sprite(a.sprite, not a.no_defringe)
    if not a.already_relit:
        f, how = relight_one(f, load_profile(a.scene), transfer.load_pack(a.pack) if a.pack else None, a.view, a)
        print("light:", how, "(single-frame rule; `apply` decides per view group)" if a.pack else "")
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
    f.add_argument("--pack", help="also build a reference pack (directory): apply/preview with --pack then copy the "
                                  "reference's own shading onto frames that line up with the neutral")

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

    for sp_ in (ap, pv):
        sp_.add_argument("--pack", help="reference pack from `fit --pack` (reference look where frames line up)")
        sp_.add_argument("--min-match", type=float, default=transfer.GATES["min_match"],
                         help="pack gate: drawing-structure match with the neutral (0..1)")
        sp_.add_argument("--max-fold", type=float, default=transfer.GATES["max_fold"],
                         help="pack gate: max share of folded (impossible) correspondence")
        sp_.add_argument("--min-iou-affine", type=float, default=transfer.GATES["min_iou_affine"],
                         help="pack gate: silhouette overlap before optical flow")
        sp_.add_argument("--no-chest", action="store_true",
                         help="pack: skip the chest stage (fine reference light on the breasts, cut by the drawn lines)")
    ap.add_argument("--group-ratio", type=float, default=0.8,
                    help="pack: use the reference for a whole view group when this share of its frames pass")
    ap.add_argument("--per-frame", action="store_true", help="pack: decide frame by frame (may flicker)")
    for sp_ in (f, ap, pv, c):
        sp_.add_argument("--no-defringe", action="store_true", help="keep semi-transparent edge colours as they are")
    a = p.parse_args(argv)
    {"fit": cmd_fit, "apply": cmd_apply, "preview": cmd_preview, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
