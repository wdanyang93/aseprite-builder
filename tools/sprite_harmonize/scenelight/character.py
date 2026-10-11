"""Character card (design 5.3): what is learned from a character's ORIGINAL drawings alone (no reference needed), once
per character, and used for every frame of that character:
  - line scale: the black-hat radius (fraction of H) from the character's own stroke width, the ink L* (line colour)
    and the stroke width that sets how the line tests scale on small frames (lines.py);
  - palette: Lab centres of the flat colour regions (regions.py), pooled over all originals;
  - thin-part scale (local width below thin_max x H) and the share of thin parts;
  - the region map, line weight and texture map of each original (maps.npz);
  - a placeholder for the gate calibration (the structure-match threshold refitted from this character's frames; until
    that is measured the default 0.68 applies).
The card folder holds data derived from the user's art (region maps, preview): it is written OUTSIDE any git work tree
(the tool repo is public) unless allow_repo=True is passed explicitly.

    card = build_card([rgba, ...], name="lyn_outfit2")
    save_card("chars/lyn_outfit2", card)       # card.json + maps.npz + preview.png
    card = load_card("chars/lyn_outfit2")
    fa = analyze_frame(frame_rgba, card)       # per-frame geometry, lines, palette index, regions, texture
Line maps here are two-pass (lines_with_texture): strokes inside fur / hair texture must be nearly ink-dark.
"""
import json
import os

import numpy as np

from .color import load_rgba, from_lab, save_rgb
from .geom import GEOM, analyze, figure_height
from .lines import LINES, line_map, stroke_width, estimate_ink
from .regions import REGIONS, palette_pixels, fit_palette, segment, texture, assign

CARD_VERSION = 1
CARD = {
    "scale_k": 1.5,                 # black-hat radius = scale_k x stroke width, clamped to [scale_min, scale_max]
    "scale_min": 0.003, "scale_max": 0.005,
    "gate_default": 0.68,           # structure-match gate until it is calibrated on this character's frames
}


def _as_rgba(x):
    if isinstance(x, (str, os.PathLike)):
        return load_rgba(x), os.path.basename(str(x))
    return np.asarray(x, np.float32), None


def calibrate_lines(origs, p=None):
    """Line parameters of a character from its originals: ink L* (median of the originals' estimates), stroke width
    (median, fraction of H) and the black-hat scale derived from it."""
    c = dict(CARD, **(p or {}))
    inks, widths = [], []
    for f in origs:
        H = figure_height(f[..., 3])
        lm = line_map(f, H)
        e = estimate_ink(lm["L"], lm["bh"], f[..., 3] > 0.5)
        if e is not None:
            inks.append(e)
        lm = line_map(f, H, {"ink_L": e}) if e is not None else lm
        w = stroke_width(lm["weight"], lm["mask"], H)
        if w is not None:
            widths.append(w)
    width = float(np.median(widths)) if widths else LINES["width"]
    return {"ink_L": float(np.median(inks)) if inks else None,
            "width": width,
            "scale": float(np.clip(c["scale_k"] * width, c["scale_min"], c["scale_max"])),
            "blur": LINES["blur"]}


def lines_with_texture(rgba, H, lp=None):
    """Line map in two passes: lines -> fur / hair texture regions -> lines again with the strict test inside them
    (lines.line_map strict). The texture mask is returned in the dict as "texture"."""
    lm0 = line_map(rgba, H, lp)
    tm = texture(rgba, H, lm0)[1]
    lm = line_map(rgba, H, lp, strict=tm)
    lm["texture"] = tm
    return lm


def line_params(card):
    """Parameters for lines.line_map from a card (None -> defaults)."""
    if not card:
        return None
    L = card["lines"]
    return {k: L[k] for k in ("ink_L", "width", "scale", "blur") if L.get(k) is not None}


def build_card(origs, name="character", p=None):
    """Character card from >= 1 original (RGBA float arrays or image paths). Returns the card dict; the per-original
    maps are in card["_maps"] (saved to maps.npz by save_card, not to the json)."""
    items = [_as_rgba(o) for o in origs]
    if not items:
        raise ValueError("a character card needs at least one original drawing")
    frames = [f for f, _ in items]
    lines_cal = calibrate_lines(frames, p)
    lp = {k: v for k, v in lines_cal.items() if v is not None}
    per, pix, lms = [], [], []
    for f, fn in items:
        H = figure_height(f[..., 3])
        lm = lines_with_texture(f, H, lp)
        lms.append(lm)
        pix.append(palette_pixels(f, lm["weight"], H))
    pal = fit_palette(np.concatenate(pix) if pix else np.zeros((0, 3)))
    maps = []
    for (f, fn), lm in zip(items, lms):
        H = figure_height(f[..., 3])
        fig = f[..., 3] > 0.5
        seg = segment(f, pal["centres"], lm["weight"], H)
        tm = lm["texture"]
        g = analyze(f[..., 3], arc=False)
        per.append({"file": fn, "shape": list(f.shape[:2]), "H": H,
                    "line_share": round(float(lm["weight"][fig].mean()), 4) if fig.any() else 0.0,
                    "regions": seg["n"], "texture_share": round(float(tm[fig].mean()), 4) if fig.any() else 0.0,
                    "thin_share": round(float((g["thin"][fig] > 0.5).mean()), 4) if fig.any() else 0.0})
        maps.append({"labels": seg["labels"], "colour": seg["colour"], "index": seg["index"],
                     "line": (lm["weight"] * 255 + 0.5).astype(np.uint8), "texture": tm, "thin": g["thin"] > 0.5})
    card = {
        "version": CARD_VERSION, "name": name,
        "conventions": {"length": "fraction of figure height H", "lab": "CIE L*a*b*, L* 0..100"},
        "originals": per,
        "lines": lines_cal,
        "palette": {"lab": np.round(pal["centres"], 2).tolist(), "counts": [int(x) for x in pal["counts"]],
                    "k_bic": pal["k_bic"], "bic": pal["bic"], "w_L": REGIONS["w_L"], "merge_dE": REGIONS["merge_dE"],
                    "max_dE": REGIONS["max_dE"]},
        "thin": {"thin_max": GEOM["thin_max"],
                 "share": round(float(np.mean([o["thin_share"] for o in per])), 4)},
        "gate": {"min_match": None, "default": CARD["gate_default"], "calibrated": False,
                 "note": "structure-match distribution of this character's frames not measured yet"},
        "_maps": maps,
    }
    return card


def _inside_git(folder):
    d = os.path.abspath(folder)
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            return d
        up = os.path.dirname(d)
        if up == d:
            return None
        d = up


def save_card(folder, card, allow_repo=False):
    """Write card.json (+ maps.npz and preview.png when the card has maps). Refuses a folder inside a git work tree
    (cards hold data derived from the user's art) unless allow_repo=True."""
    repo = _inside_git(folder)
    if repo and not allow_repo:
        raise ValueError(f"refusing to write a character card inside the git work tree {repo}: "
                         "cards contain data derived from the user's art; choose a folder outside the repository")
    os.makedirs(folder, exist_ok=True)
    maps = card.get("_maps")
    js = {k: v for k, v in card.items() if k != "_maps"}
    with open(os.path.join(folder, "card.json"), "w") as fh:
        json.dump(js, fh, indent=1)
    if maps:
        z = {}
        for i, m in enumerate(maps):
            for k, v in m.items():
                z[f"{i}_{k}"] = v
        np.savez_compressed(os.path.join(folder, "maps.npz"), **z)
        save_rgb(os.path.join(folder, "preview.png"), palette_card(card))
    return folder


def load_card(folder, maps=False):
    """Read a card folder (card.json; maps.npz too when maps=True, as card["_maps"])."""
    with open(os.path.join(folder, "card.json")) as fh:
        card = json.load(fh)
    if card.get("version") != CARD_VERSION:
        raise ValueError(f"unsupported character card version {card.get('version')}")
    if maps and os.path.exists(os.path.join(folder, "maps.npz")):
        z = np.load(os.path.join(folder, "maps.npz"))
        n = len(card["originals"])
        card["_maps"] = [{k.split("_", 1)[1]: z[k] for k in z.files if k.split("_", 1)[0] == str(i)} for i in range(n)]
    return card


def palette_card(card, sw=48):
    """Palette swatches as an RGB image (one square per palette colour, dark to light)."""
    lab = np.array(card["palette"]["lab"], np.float32).reshape(-1, 1, 3)
    rgb = from_lab(lab).reshape(-1, 3)
    img = np.full((sw + 8, len(rgb) * (sw + 4) + 4, 3), 0.15, np.float32)
    for i, c in enumerate(rgb):
        img[4:4 + sw, 4 + i * (sw + 4): 4 + i * (sw + 4) + sw] = c
    return img


def analyze_frame(rgba, card=None, geometry=True):
    """Per-frame analysis with a character card (design 2.7-1): geometry (geom.analyze), line map with the card's line
    parameters, nearest card palette colour per pixel (-1 = colour the card has never seen; new_colour_share counts
    the non-line pixels that are -1, the warning for a palette the card does not know), the frame's own
    line-walled regions in card colours, and the texture map. Without a card everything uses the defaults and the
    palette is fitted on the frame itself."""
    rgba = np.asarray(rgba, np.float32)
    H = figure_height(rgba[..., 3])
    out = {"H": H}
    lm = lines_with_texture(rgba, H, line_params(card))
    out["lines"] = lm
    if card:
        centres = np.array(card["palette"]["lab"], np.float32)
        pal = card["palette"]
        rp = {"w_L": pal.get("w_L", REGIONS["w_L"]), "max_dE": pal.get("max_dE", REGIONS["max_dE"])}
    else:
        centres = fit_palette(palette_pixels(rgba, lm["weight"], H))["centres"]
        rp = {}
    idx, dist = assign(rgba, centres, rp)
    fig = rgba[..., 3] > 0.5
    out["palette_index"], out["palette_dist"] = idx, dist
    surf = fig & (lm["weight"] <= REGIONS["line_w"])      # line art is not a region colour (black outlines)
    out["new_colour_share"] = round(float((idx[surf] < 0).mean()), 4) if surf.any() else 0.0
    out["regions"] = segment(rgba, centres, lm["weight"], H, rp)
    out["texture"] = lm["texture"]
    if geometry:
        out["geom"] = analyze(rgba[..., 3])
    return out
