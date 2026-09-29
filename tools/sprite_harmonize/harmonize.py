#!/usr/bin/env python3
"""Relight a 2D sprite so it sits naturally on a given background.

Pipeline (all steps optional / tunable):
  1. defringe   - replace color of semi-transparent edge pixels with interior
                  color (removes white halos left over from a white BG).
  2. grade      - either
                    a) auto: tint by the background's key-light color and
                       match exposure to the background, or
                    b) --ref: transfer the color statistics of a reference
                       sprite that already has the desired look.
  3. rim light  - directional back/rim light along the silhouette, colored
                  by the background's key light.
  4. composite  - (when --place is given) scale, contact shadow, light wrap.

Examples:
  # grade only, write a transparent sprite
  python harmonize.py sprite.png --bg background.png -o out.png

  # grade + place on background for a preview
  python harmonize.py sprite.png --bg background.png -o out.png \
      --place 640,602,240 --composite-out preview.png

  # learn a look from a hand-made reference and reuse it on every frame
  python harmonize.py sprite.png --bg background.png --ref target_sprite.png \
      --save-look forest_sunset.json -o out.png
  python harmonize.py frames/*.png --look forest_sunset.json -o graded_frames/
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

LUMA = np.array([0.2126, 0.7152, 0.0722])


# ---------------------------------------------------------------- color utils
def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def load_rgba(path):
    im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    return im[..., :3], im[..., 3]


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def save_rgba(path, rgb, alpha):
    out = np.dstack([np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)])
    Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA").save(path)


def save_rgb(path, rgb):
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").save(path)


def to_lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def from_lab(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)


# ------------------------------------------------------------ analysis steps
def analyze_background(bg, region=None):
    """Return key-light color (linear, max channel = 1) and reference luminance."""
    lin = srgb_to_linear(bg)
    lum = lin @ LUMA
    thr = np.percentile(lum, 90)
    key = lin[lum >= thr].mean(0)
    key = key / key.max()
    area = bg if region is None else region
    area_lum_srgb = area @ LUMA
    return {
        "key_light": key.tolist(),
        # a lit foreground character reads about as bright as the BG's
        # brightest few percent (fitted on a hand-made reference)
        "target_luma": float(np.percentile(bg @ LUMA, 96)),
        "ambient": srgb_to_linear(area).reshape(-1, 3).mean(0).tolist(),
        "shadow": srgb_to_linear(area[area_lum_srgb <= np.percentile(area_lum_srgb, 10)]).mean(0).tolist(),
    }


def defringe(rgb, alpha, radius=4):
    """Fill edge pixel colors from solid interior pixels (kills white halos)."""
    solid = (alpha > 0.95).astype(np.float32)
    k = radius * 2 + 1
    num = cv2.GaussianBlur(rgb * solid[..., None], (0, 0), radius)
    den = cv2.GaussianBlur(solid, (0, 0), radius)[..., None]
    fill = num / np.maximum(den, 1e-4)
    # repeat for pixels still too far from solid area
    for _ in range(3):
        miss = den[..., 0] < 1e-3
        if not miss.any():
            break
        fill = np.where(miss[..., None], cv2.dilate(fill, np.ones((k, k), np.uint8)), fill)
        den = np.where(miss[..., None], 1.0, den)
    w = np.clip((alpha - 0.5) / 0.45, 0, 1)[..., None]  # keep own color when mostly opaque
    return rgb * w + fill * (1 - w)


def grade_auto(rgb, alpha, info, tint=0.7, exposure=None, haze=0.08):
    lin = srgb_to_linear(rgb)
    key = np.array(info["key_light"])
    lin = lin * (1 - tint + tint * key)  # sprite lit by the scene's key light
    mask = alpha > 0.5
    if exposure is None:
        cur = (linear_to_srgb(lin)[mask] @ LUMA).mean()
        # gain in linear space that moves the mean sRGB luma to target
        exposure = (info["target_luma"] / max(cur, 1e-4)) ** 2.2
        exposure = float(np.clip(exposure, 0.4, 1.6))
    lin = lin * exposure
    # lift shadows toward the scene's shadow color (atmosphere / bounce light)
    shadow = np.array(info["shadow"])
    l = (lin @ LUMA)[..., None]
    lin = lin + haze * (1 - np.clip(l * 3, 0, 1)) * shadow * 4
    return linear_to_srgb(lin), exposure


def fit_look(src_rgb, src_a, ref_rgb, ref_a):
    """LAB transfer: L linear (mean/std), a/b by mean/std per lightness band."""
    s = to_lab(src_rgb)[src_a > 0.8]
    r = to_lab(ref_rgb)[ref_a > 0.8]
    # linear L transfer: histogram matching between different drawings creates
    # steep curve segments that turn soft skin shading into blotches
    look = {"L_gain": float(np.clip(r[:, 0].std() / max(s[:, 0].std(), 1e-3), 0.5, 2.0))}
    look["L_offset"] = float(r[:, 0].mean() - s[:, 0].mean() * look["L_gain"])
    # chroma transfer per lightness band (shadows/mids/highlights tint separately)
    bands = []
    s_q = np.percentile(s[:, 0], [0, 25, 50, 75, 100])
    r_q = np.percentile(r[:, 0], [0, 25, 50, 75, 100])
    for i in range(4):
        sm = (s[:, 0] >= s_q[i]) & (s[:, 0] <= s_q[i + 1])
        rm = (r[:, 0] >= r_q[i]) & (r[:, 0] <= r_q[i + 1])
        bands.append({
            "L_center": float(np.median(s[sm, 0])),
            "src_mean": s[sm, 1:].mean(0).tolist(), "src_std": s[sm, 1:].std(0).tolist(),
            "ref_mean": r[rm, 1:].mean(0).tolist(), "ref_std": r[rm, 1:].std(0).tolist(),
        })
    look["bands"] = bands
    return look


def apply_look(rgb, look, strength=1.0):
    lab = to_lab(rgb)
    L = lab[..., 0]
    newL = L * look["L_gain"] + look["L_offset"]
    centers = np.array([b["L_center"] for b in look["bands"]])
    # soft weights across lightness bands
    d = np.abs(L[..., None] - centers)
    w = np.exp(-(d / 30.0) ** 2) + 1e-6
    w = w / w.sum(-1, keepdims=True)
    ab = np.zeros(lab.shape[:2] + (2,), np.float32)
    for i, b in enumerate(look["bands"]):
        sm, ss = np.array(b["src_mean"]), np.maximum(np.array(b["src_std"]), 0.5)
        rm, rs = np.array(b["ref_mean"]), np.array(b["ref_std"])
        gain = np.clip(rs / ss, 0.7, 1.6)
        ab += w[..., i:i + 1] * ((lab[..., 1:] - sm) * gain + rm)
    out = np.dstack([newL, ab])
    out = lab + (out - lab) * strength
    return from_lab(out)


def rim_light(rgb, alpha, key, angle_deg=160, width=None, strength=1.2, spread=1.2):
    """Additive rim light on the silhouette side that faces the light."""
    h = np.count_nonzero(alpha.max(1) > 0.1)
    if width is None:
        width = max(1.5, h / 60.0)
    a = alpha.astype(np.float32)
    blur = cv2.GaussianBlur(a, (0, 0), width * 1.5)
    gy, gx = np.gradient(cv2.GaussianBlur(a, (0, 0), width * 2.5))
    n = np.hypot(gx, gy) + 1e-6
    nx, ny = -gx / n, -gy / n  # outward normal (image coords, y down)
    t = math.radians(angle_deg)
    lx, ly = math.cos(t), -math.sin(t)
    facing = np.clip(nx * lx + ny * ly, 0, 1) ** spread
    band = np.clip((1 - blur) * 1.6, 0, 1) ** 1.5 * a  # inside edge only, soft falloff
    # thin bright line right at the silhouette on top of the soft glow
    thin = np.clip((1 - cv2.GaussianBlur(a, (0, 0), max(1.0, width * 0.3))) * 2.5, 0, 1) * a
    rim = np.maximum(band * 0.7, thin) * facing
    lin = srgb_to_linear(rgb)
    lin = lin + rim[..., None] * strength * np.array(key)
    return linear_to_srgb(lin), rim


# ---------------------------------------------------------------- composite
def crop_to_alpha(rgb, alpha, pad=2):
    ys, xs = np.where(alpha > 0.02)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, alpha.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, alpha.shape[1])
    return rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def resize_premult(rgb, alpha, height):
    scale = height / alpha.shape[0]
    size = (max(1, round(alpha.shape[1] * scale)), max(1, round(alpha.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    pm = cv2.resize(rgb * alpha[..., None], size, interpolation=interp)
    a = np.clip(cv2.resize(alpha, size, interpolation=interp), 0, 1)
    return np.clip(pm / np.maximum(a[..., None], 1e-4), 0, 1), a


def composite(bg, rgb, alpha, cx, foot_y, height, shadow=0.55, wrap=0.35, dim=0.92):
    rgb, alpha = crop_to_alpha(rgb, alpha)
    rgb, alpha = resize_premult(rgb, alpha, height)
    h, w = alpha.shape
    x0, y0 = int(round(cx - w / 2)), int(round(foot_y - h))
    out = bg.copy()
    H, W = bg.shape[:2]

    def region(arr_h, arr_w, ox, oy):
        bx0, by0 = max(ox, 0), max(oy, 0)
        bx1, by1 = min(ox + arr_w, W), min(oy + arr_h, H)
        return (by0, by1, bx0, bx1), (by0 - oy, by1 - oy, bx0 - ox, bx1 - ox)

    # contact shadow: soft ellipse under the lowest opaque pixels (the feet)
    if shadow > 0:
        rows = np.where(alpha.max(1) > 0.5)[0]
        foot_rows = alpha[rows[-1] - max(2, h // 25): rows[-1] + 1]
        cols = np.where(foot_rows.max(0) > 0.5)[0]
        fw = (cols[-1] - cols[0]) * 0.65 + h * 0.08
        fcx = x0 + (cols[0] + cols[-1]) / 2
        sh = np.zeros((H, W), np.float32)
        cv2.ellipse(sh, (int(fcx), int(y0 + rows[-1])), (int(fw / 2 + 1), int(max(2, h * 0.018))),
                    0, 0, 360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, h * 0.012))
        dark = np.array([0.18, 0.10, 0.05])
        out = out * (1 - sh[..., None] * shadow) + dark * sh[..., None] * shadow * 0.3

    (a0, a1, b0, b1), (s0, s1, t0, t1) = region(h, w, x0, y0)
    spr, a = rgb[s0:s1, t0:t1], alpha[s0:s1, t0:t1]
    base = out[a0:a1, b0:b1]
    # in-scene the character sits a little below the key-lit reference
    spr = linear_to_srgb(srgb_to_linear(spr) * dim ** 2.2)
    # light wrap: blurred background bleeds onto the sprite's edge
    if wrap > 0:
        r = max(1.0, h / 90.0)
        bg_blur = cv2.GaussianBlur(bg, (0, 0), r * 2)[a0:a1, b0:b1]
        edge = np.clip(1 - cv2.GaussianBlur(a, (0, 0), r), 0, 1) * a
        spr = linear_to_srgb(srgb_to_linear(spr) + srgb_to_linear(bg_blur) * edge[..., None] * wrap)
    out[a0:a1, b0:b1] = base * (1 - a[..., None]) + spr * a[..., None]
    return out


# ---------------------------------------------------------------------- cli
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sprites", nargs="+", help="sprite(s) with transparency (PNG/WebP RGBA)")
    p.add_argument("-o", "--out", required=True,
                   help="graded sprite (RGBA PNG); a directory when several sprites are given")
    p.add_argument("--bg", help="background image (used for light color / exposure / preview)")
    p.add_argument("--ref", help="reference sprite with the desired look; fits a look from it")
    p.add_argument("--look", help="apply a look JSON saved with --save-look")
    p.add_argument("--save-look", help="write the fitted look (+ BG light info) to JSON")
    p.add_argument("--tint", type=float, default=0.7, help="auto grade: key-light tint amount 0..1")
    p.add_argument("--exposure", type=float, help="auto grade: linear gain (default: auto from BG)")
    p.add_argument("--haze", type=float, default=0.08, help="auto grade: shadow lift toward BG shadow color")
    p.add_argument("--look-strength", type=float, default=1.0)
    p.add_argument("--light-angle", type=float, default=160,
                   help="direction the rim light comes FROM, degrees (0=right, 90=up, 180=left)")
    p.add_argument("--rim", type=float, default=1.2, help="rim light strength (0 = off)")
    p.add_argument("--rim-width", type=float, help="rim width in px (default: sprite height/60)")
    p.add_argument("--no-defringe", action="store_true")
    p.add_argument("--place", help="cx,foot_y,height - composite onto --bg at this spot")
    p.add_argument("--shadow", type=float, default=0.55, help="contact shadow opacity")
    p.add_argument("--wrap", type=float, default=0.35, help="light wrap amount")
    p.add_argument("--dim", type=float, default=0.92, help="composite: brightness of sprite in scene")
    p.add_argument("--composite-out", help="where to write the composite preview (single sprite only)")
    args = p.parse_args(argv)

    bg = load_rgb(args.bg) if args.bg else None
    look = None
    if args.look:
        with open(args.look) as f:
            look = json.load(f)
    info = look.get("bg") if look else None
    if info is None:
        if bg is None:
            p.error("--bg is required unless --look contains background info")
        info = analyze_background(bg)
    look = dict(look or {})
    if args.exposure is not None:
        look["exposure"] = args.exposure
    look.setdefault("tint", args.tint)
    look.setdefault("haze", args.haze)
    if args.rim_width is not None:
        look["rim_width"] = args.rim_width

    multi = len(args.sprites) > 1
    if multi:
        os.makedirs(args.out, exist_ok=True)
    for i, path in enumerate(args.sprites):
        rgb, alpha = load_rgba(path)
        if not args.no_defringe:
            rgb = defringe(rgb, alpha)

        if args.ref and i == 0:
            ref_rgb, ref_a = load_rgba(args.ref)
            look.update(fit_look(rgb, alpha, defringe(ref_rgb, ref_a), ref_a))
        if "L_gain" in look:
            rgb = apply_look(rgb, look, args.look_strength)
            mode = "look (reference-based)"
        else:
            # exposure is fixed by the first sprite so animation frames don't flicker
            rgb, look["exposure"] = grade_auto(rgb, alpha, info, look["tint"], look.get("exposure"), look["haze"])
            mode = f"auto  key_light={np.round(info['key_light'], 3).tolist()} exposure={look['exposure']:.3f}"

        if args.rim > 0:
            if "rim_width" not in look:
                look["rim_width"] = max(1.5, np.count_nonzero(alpha.max(1) > 0.1) / 60.0)
            rgb, _ = rim_light(rgb, alpha, info["key_light"], args.light_angle, look["rim_width"], args.rim)

        out = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0] + ".png") if multi else args.out
        save_rgba(out, rgb, alpha)
        print(f"[{mode}] {path} -> {out}")

        if args.place and not multi:
            if bg is None:
                p.error("--place needs --bg")
            cx, fy, hh = (float(v) for v in args.place.split(","))
            comp = composite(bg, rgb, alpha, cx, fy, int(hh), args.shadow, args.wrap, args.dim)
            cpath = args.composite_out or out.rsplit(".", 1)[0] + "_composite.png"
            save_rgb(cpath, comp)
            print("composite saved:", cpath)

    if args.save_look:
        look["bg"] = info
        with open(args.save_look, "w") as f:
            json.dump(look, f, indent=1)
        print("look saved:", args.save_look)


if __name__ == "__main__":
    sys.exit(main())
