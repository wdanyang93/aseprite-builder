#!/usr/bin/env python3
"""Relight 2D character sprites so they belong in a scene.

  fit      neutral frame + hand-lit reference of the same pose  ->  scene profile (JSON, ~2 KB)
  apply    scene profile + any frames (any pose / direction)     ->  relit frames (RGBA PNG)
  preview  relit (or neutral) sprite on a background with contact shadow and light wrap
  check    quality sheet + numbers for a scene profile (neutral | relit | reference)

  blend-fit  one frame + the painter's scene picture of it (+ Adobe masks of both) -> scene blend look (JSON)
  blend      scene blend look + frames (+ their Adobe masks)                         -> game-height frames (RGBA PNG)
  look-fit   original + lit reference of the same pose (+ Adobe masks of the original) -> reference look (JSON)
  look       reference look + frames / motions (+ their Adobe masks)                    -> lit frames (RGBA PNG)
  Adobe outputs of image X.png live in <masks>/X/: <part>.png masks (scenelight.parts names) + vector.svg line art.

Examples
  python relight.py fit neutral.png warm_reference.png -o scenes/forest_sunset.json --sheet check.png
  python relight.py apply scenes/forest_sunset.json frames/*.png -o out/ --view-from-name --sheet ba.png
  python relight.py preview scenes/forest_sunset.json frames/walk_03.png --bg bg.png --place 640,602,240 -o shot.png
  python relight.py blend-fit frames/idle.png painter_scene.png --masks masks -o blend.json --sheet blend_check.png
  python relight.py blend blend.json frames/*.png --masks masks -o out/ --sheet blend_ba.png
  python relight.py look-fit orig.png lamp_ref.png --masks masks -o look_lamp.json --sheet look_check.png
  python relight.py look look_lamp.json frames/*.png --masks masks -o out/ --sheet look_ba.png
"""
import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenelight import fit_scene, load_profile, relight, save_profile  # noqa: E402
from scenelight.color import defringe, figure_height, load_rgb, load_rgba, save_rgb, save_rgba, to_lab  # noqa: E402
from scenelight.composite import composite  # noqa: E402
from scenelight import blend as BL, look as LK, report, transfer  # noqa: E402
from scenelight.parts import load_lines, load_parts  # noqa: E402
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


# ------------------------------------------------------------------ scene blend / reference look (Adobe parts + lines)
def _adobe(root, img_path, shape, lines=True):
    """Adobe outputs of one image, at its resolution: <root>/<image stem>/<part>.png masks (+ vector.svg or
    vector.png, the image_vectorize line art). -> (parts, lines or None)"""
    d = os.path.join(root, os.path.splitext(os.path.basename(img_path))[0])
    if not os.path.isdir(d):
        raise FileNotFoundError(f"no Adobe mask folder {d} - do the Adobe steps for this image first")
    h, w = shape[:2]
    parts = {k: v if v.shape == (h, w) else cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
             for k, v in load_parts(d).items() if k != "vector"}
    if not lines:
        return parts, None
    vec = [os.path.join(d, f) for f in ("vector.svg", "vector.png") if os.path.isfile(os.path.join(d, f))]
    if not vec:
        raise FileNotFoundError(f"no line art {os.path.join(d, 'vector.svg')} (Adobe image_vectorize): "
                                "without it line art could get lighter")
    return parts, load_lines(vec[0], shape).astype(np.float32)


def _first_claim(parts):
    """Overlapping masks made disjoint in load order (shoes before legs ...); 'subject' dropped (blend averages
    overlaps, look lets the first claim win: both then see the same parts)."""
    out, claimed = {}, None
    for k, m in parts.items():
        if k == "subject":
            continue
        out[k] = m if claimed is None else m * ~claimed
        claimed = (m >= 0.5) if claimed is None else claimed | (m >= 0.5)
    return out


def _with_rest(parts, fig, names):
    """`parts` limited to `names` + 'rest' = the figure pixels none of them covers (in Lyn's case the clothes): no Adobe
    call needed for it, and frame and scene split the figure the same way."""
    out = {k: parts[k] for k in names if k in parts}
    u = np.zeros(fig.shape, bool)
    for m in out.values():
        u |= m >= 0.5
    out["rest"] = (fig & ~u).astype(np.float32)
    return out


def _coverage(parts, alpha, names):
    """share of the figure covered by the parts the look knows (the rest takes the whole-figure curve)."""
    fig = alpha >= 0.5
    u = np.zeros(fig.shape, bool)
    for k in names:
        if k in parts:
            u |= parts[k] >= 0.5
    return float((u & fig).sum()) / max(int(fig.sum()), 1)


def _u8(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def _need_alpha(x, path):
    if _f32(x)[..., 3].min() >= 1.0:
        raise SystemExit(f"{path} has no transparent background: cut it out with Adobe image_remove_background first")


def _f32(x):
    return x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)


def _outputs(a):
    multi = len(a.frames) > 1 or os.path.isdir(a.out) or a.out.endswith(os.sep) or not os.path.splitext(a.out)[1]
    if multi:
        os.makedirs(a.out, exist_ok=True)
    return [(p, os.path.join(a.out, os.path.splitext(os.path.basename(p))[0] + ".png") if multi else a.out)
            for p in a.frames]


def _panels(tiles, path, h=720):
    """Check sheet, one row at `h` rows: each (rgb, figure mask) tile cropped to its figure, then the top 30 % of each
    figure (head / hair: outline and colour; small images are enlarged with visible pixels)."""
    row = []
    for frac in (1.0, 0.3):
        for rgb, m in tiles:
            ys = np.nonzero(m.any(1))[0]
            y0 = ys.min()
            y1 = y0 + max(2, int((ys.max() + 1 - y0) * frac))
            xs = np.nonzero(m[y0:y1].any(0))[0]
            p = max(2, int(0.04 * (y1 - y0)))
            c = rgb[max(0, y0 - p):y1 + p, max(0, xs.min() - p):xs.max() + p + 1]
            s = h / c.shape[0]
            row += [cv2.resize(c, (max(1, int(c.shape[1] * s)), h),
                               interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST),
                    np.full((h, 8, 3), 0.1, np.float32)]
    Image.fromarray((np.clip(np.hstack(row), 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def _rgba_tile(x):
    x = _f32(x)
    return report.on_grey(x), x[..., 3] >= 0.5


def cmd_blend_fit(a):
    f = load_rgba(a.frame)
    _need_alpha(f, a.frame)
    parts, lines = _adobe(a.masks, a.frame, f.shape)
    tgt = load_rgb(a.scene)
    tparts, _ = _adobe(a.masks, a.scene, tgt.shape, lines=False)
    tsil = np.zeros(tgt.shape[:2], bool)
    for m in tparts.values():                  # subject + every part: Adobe's subject alone can miss e.g. the tail
        tsil |= m >= 0.5
    if not tsil.any():
        raise SystemExit(f"no figure in the scene masks of {a.scene} (save subject.png and the part masks)")
    parts, tparts = _first_claim(parts), _first_claim(tparts)
    named = [k for k in parts if k in tparts]
    one = sorted(set(parts) ^ set(tparts))
    if one:
        print("  only in one picture's masks (counted as rest there):", ", ".join(one))
    parts, tparts = _with_rest(parts, f[..., 3] >= 0.5, named), _with_rest(tparts, tsil, named)
    height = a.height or int(round(f.shape[0] * figure_height(tsil) / figure_height(f[..., 3])))
    t = time.time()
    look = BL.fit_blend(f, parts, tgt, tparts, height, lines=lines, target_sil=tsil)
    look["height"], look["rows"] = height, f.shape[0]     # frames of other canvas sizes scale with the same factor
    look["rest_of"] = named                                # 'rest' = the figure minus these parts, on every frame
    with open(a.out, "w") as fh:
        json.dump(look, fh)
    out = BL.apply_blend(f, parts, look, height, lines)
    print(f"scene blend look -> {a.out}  ({time.time() - t:.0f}s, game height {height} rows"
          f"{'' if a.height else ', matched to the figure in the scene picture'})")
    P = look["params"]
    near = lambda k: cv2.dilate(tsil.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    bg = np.median(tgt[near(11) & ~near(7)], 0) if (near(11) & ~near(7)).any() else 0.5   # scene just outside
    sl = to_lab(out[..., :3] * out[..., 3:] + bg * (1 - out[..., 3:]))      # the blend as it shows in the scene
    ssil, tl = out[..., 3] >= 0.5, to_lab(tgt)
    size = out.shape[1::-1]
    print("part        px game/scene   L* median (scene)   a* (scene)    b* (scene)    outer edge L* (scene)")
    n_fig = look["parts"].get("_all", {"n": [int(ssil.sum())]})["n"][0]
    for k, p in look["parts"].items():
        sm = ssil.astype(np.float32) if k == "_all" else cv2.resize(parts[k], size, interpolation=cv2.INTER_AREA)
        tm = tsil if k == "_all" else tparts[k] >= 0.5
        xs, sa, sb, _ = BL._stats(sl, ssil, sm, P["edge"], (50,))
        ys, ta, tb, _ = BL._stats(tl, tsil, tm, P["edge"], (50,))
        xr = BL._stats(sl, ssil, sm, P["edge"], (50,), rim=True)[0]
        yr = BL._stats(tl, tsil, tm, P["edge"], (50,), rim=True)[0]
        off = xs is None or ys is None or max(abs(xs[0] - ys[0]), abs(sa - ta), abs(sb - tb)) > 3
        # checked: parts of >= 5 % of the figure (thin ones mix with their neighbours' curves across the feather);
        # 'whole' is information only: with 'rest' every figure pixel has a part curve
        info = k == "_all" or p["n"][0] < 0.05 * n_fig
        fmt = lambda v: "  -  " if v is None else f"{v[0]:5.1f}"
        print(f"{'whole' if k == '_all' else k:10s} {p['n'][0]:7d}/{p['n'][1]:<7d}  {fmt(xs)} ({fmt(ys)})"
              f"      {sa:5.1f} ({ta:5.1f})  {sb:5.1f} ({tb:5.1f})  {fmt(xr)} ({fmt(yr)})"
              + ("" if k == "_all" else "   (small part: information only)" if info
                 else "   <- off by > 3" if off else ""))
    miss = [k for k in parts if k not in look["parts"]]
    if miss:
        print("  too small for an own curve (whole-figure curve):", ", ".join(miss))
    cov = _coverage(parts, f[..., 3], named)
    print(f"named Adobe parts cover {cov:.0%} of the figure (the rest: 'rest')"
          + ("  <- under 50 %: check the masks" if cov < 0.5 else ""))
    full = BL.apply_blend(f, parts, look, None, lines)     # the line hold acts at full resolution
    held = (lines >= 0.5) & (f[..., 3] > 0)
    dl = (to_lab(full[..., :3])[..., 0] - to_lab(f[..., :3])[..., 0])[held]
    alpha_ok = np.array_equal(out[..., 3], BL.downscale(f, height)[..., 3])
    print(f"alpha = plain downscale {alpha_ok}, line px lighter {int((dl > 0.5).sum())} of {int(held.sum())}")
    if a.sheet:
        _panels([_rgba_tile(BL.downscale(f, height)), _rgba_tile(out), (tgt, tsil)], a.sheet)
        print("sheet (plain downscale | blend | scene) ->", a.sheet)


def cmd_blend(a):
    with open(a.look) as fh:
        look = json.load(fh)
    thumbs_in, thumbs_out, failed = [], [], 0
    for p, dst in _outputs(a):
        t = time.time()
        try:
            f = load_rgba(p)
            parts, lines = _adobe(a.masks, p, f.shape)
            parts = _first_claim(parts)
            if "rest_of" in look:
                parts = _with_rest(parts, f[..., 3] >= 0.5, look["rest_of"])
            if a.height:
                height = a.height
            elif "height" in look:
                height = int(round(f.shape[0] * look["height"] / look["rows"]))
            else:
                raise ValueError("the look has no game height: give --height")
            o = BL.apply_blend(f, parts, look, height, lines)
            save_rgba(dst, o)
        except Exception as e:
            failed += 1
            print(f"{p}: FAILED ({e.__class__.__name__}: {e}) - skipped")
            continue
        known = [k for k in look["parts"] if k not in ("_all", "rest")]
        cov = _coverage(parts, f[..., 3], known)
        gone = [k for k in known if k not in parts or not (parts[k] >= 0.5).any()]
        print(f"{p} -> {dst}  {o.shape[1]}x{o.shape[0]}  named Adobe parts cover {cov:.0%} of the figure"
              + (f"  <- no mask for: {', '.join(gone)} (takes the rest curve)" if gone else "")
              + f"  {time.time() - t:.1f}s")
        if a.sheet:
            thumbs_in.append(_thumb(BL.downscale(f, height))); thumbs_out.append(_thumb(o))
    if a.sheet and thumbs_in:
        report.before_after(thumbs_in, thumbs_out, a.sheet)
        print("before|after sheet (plain downscale | blend) ->", a.sheet)
    if failed:
        print(f"{failed} frame(s) failed - see above")


def _look_checks(src, out, lines):
    """alpha bit-identical, Adobe line px never lighter in any channel, no new pure-white px."""
    fig = src[..., 3] > 0
    held = (lines >= 0.5) & fig
    alpha_ok = np.array_equal(out[..., 3], src[..., 3])
    lighter = int((out[..., :3][held].astype(int) > src[..., :3][held].astype(int)).any(-1).sum())
    white = int(((out[..., :3] >= 250).all(-1) & ~(src[..., :3] >= 250).all(-1) & fig).sum())
    ok = alpha_ok and lighter == 0 and white == 0
    return (f"alpha identical {alpha_ok}, line px lighter {lighter} of {int(held.sum())}, new white px {white}"
            + ("" if ok else "  <- CHECK FAILED")), ok


def cmd_look_fit(a):
    o, r = _u8(a.original), _u8(a.reference)
    _need_alpha(o, a.original); _need_alpha(r, a.reference)
    parts, lines = _adobe(a.masks, a.original, o.shape)
    t = time.time()
    rw, conf = LK.align_ref(o, r)                          # once: for the fit and for the self check
    valid = (rw[..., 3] > 0.5) & (conf > 0.3) & (o[..., 3] > 0) & (lines < 0.5)
    share = valid.sum() / max(int((o[..., 3] > 0).sum()), 1)
    if share < 0.5:                                        # the self check would measure (almost) nothing
        raise SystemExit(f"CHECK FAILED: the reference lines up with the original on {share:.0%} of the figure "
                         "(under half) - same pose needed; no look written")
    look = LK.fit_look(o, rw, parts, lines, align=False, align_conf=conf)
    with open(a.out, "w") as fh:
        json.dump(look, fh)
    x = LK.apply_look(o, parts, look, lines)
    print(f"reference look -> {a.out}  ({time.time() - t:.0f}s)  light from {look['az']:.0f} deg "
          "(0 = screen right, 90 = top, 180 = left)")
    lab_r = to_lab(np.clip(rw[..., :3], 0, 1))
    de = lambda y: float(np.linalg.norm(to_lab(_f32(y)[..., :3]) - lab_r, axis=-1)[valid].mean())
    d0, d1 = de(o), de(x)
    good = d1 <= 8 and d1 <= 0.5 * d0
    print(f"self colour error dE {d1:.2f} (unchanged {d0:.2f}; pass: <= 8 and <= half of unchanged)"
          f"{'' if good else '  <- CHECK FAILED'}")
    print(_look_checks(o, x, lines)[0] + f"  (reference lines up on {share:.0%} of the figure)")
    if a.sheet:
        ref = np.dstack([rw[..., :3], _f32(o)[..., 3]])     # shown with the original's alpha
        _panels([_rgba_tile(o), _rgba_tile(x), _rgba_tile(ref)], a.sheet)
        print("sheet (original | look | reference aligned) ->", a.sheet)


def cmd_look(a):
    with open(a.look) as fh:
        look = json.load(fh)
    thumbs_in, thumbs_out, failed = [], [], 0
    for p, dst in _outputs(a):
        t = time.time()
        try:
            f = _u8(p)
            parts, lines = _adobe(a.masks, p, f.shape)
            x = LK.apply_look(f, parts, look, lines)
            Image.fromarray(x, "RGBA").save(dst)
        except Exception as e:
            failed += 1
            print(f"{p}: FAILED ({e.__class__.__name__}: {e}) - skipped")
            continue
        print(f"{p} -> {dst}  {_look_checks(f, x, lines)[0]}  {time.time() - t:.1f}s")
        if a.sheet:
            thumbs_in.append(_thumb(_f32(f))); thumbs_out.append(_thumb(_f32(x)))
    if a.sheet and thumbs_in:
        report.before_after(thumbs_in, thumbs_out, a.sheet)
        print("before|after sheet ->", a.sheet)
    if failed:
        print(f"{failed} frame(s) failed - see above")


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

    masks = "Adobe outputs: image X.png -> <masks>/X/<part>.png + vector.svg (default: masks)"
    bf = sub.add_parser("blend-fit", help="scene blend: fit once on one frame + the painter's scene picture of it")
    bf.add_argument("frame"); bf.add_argument("scene", help="the scene picture with the painted character (game scale)")
    bf.add_argument("-o", "--out", required=True, help="scene blend look JSON")
    bf.add_argument("--height", type=int,
                    help="game height of the frame canvas (default: the figure as tall as in the scene picture)")
    bf.add_argument("--sheet", help="write a check sheet PNG (plain downscale | blend | scene)")
    bl = sub.add_parser("blend", help="scene blend: apply a blend look to frames (output at game height)")
    bl.add_argument("look"); bl.add_argument("frames", nargs="+")
    bl.add_argument("-o", "--out", required=True, help="output PNG (one frame) or directory")
    bl.add_argument("--height", type=int, help="game height (default: the look's, scaled by the canvas height)")
    bl.add_argument("--sheet", help="write a before|after contact sheet PNG")
    lf = sub.add_parser("look-fit", help="reference look: fit on an original + its lit reference (same pose)")
    lf.add_argument("original")
    lf.add_argument("reference", help="lit reference of the same pose, transparent background")
    lf.add_argument("-o", "--out", required=True, help="reference look JSON (one per light)")
    lf.add_argument("--sheet", help="write a check sheet PNG (original | look | reference aligned)")
    lk = sub.add_parser("look", help="reference look: apply a look to frames / motions")
    lk.add_argument("look"); lk.add_argument("frames", nargs="+")
    lk.add_argument("-o", "--out", required=True, help="output PNG (one frame) or directory")
    lk.add_argument("--sheet", help="write a before|after contact sheet PNG")
    for sp_ in (bf, bl, lf, lk):
        sp_.add_argument("--masks", default="masks", help=masks)
    a = p.parse_args(argv)
    {"fit": cmd_fit, "apply": cmd_apply, "preview": cmd_preview, "check": cmd_check, "blend-fit": cmd_blend_fit,
     "blend": cmd_blend, "look-fit": cmd_look_fit, "look": cmd_look}[a.cmd](a)


if __name__ == "__main__":
    main()
