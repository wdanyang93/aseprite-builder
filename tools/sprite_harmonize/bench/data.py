"""Loading the private data (paths from config) + the cached derived arrays every case shares.

Every frame is float RGBA, straight alpha, defringed like the research scripts (scenelight.color.defringe).
References are warped into their neutral's geometry with the tool's own aligner (Pairs(ref, neutral).warp), exactly as
R_methods / R_audit / R_practice / the pack do (their cached arrays are bit-identical to this).
"""
import os

import numpy as np
import cv2

from scenelight.color import load_rgba, defringe, materials, to_lab
from scenelight.align import Pairs

from . import config
from . import parts as PARTS

_MEMO = {}


def memo(key, fn):
    if key not in _MEMO:
        _MEMO[key] = fn()
    return _MEMO[key]


def ld(path, do_defringe=True):
    x = load_rgba(path)
    if do_defringe:
        x[..., :3] = defringe(x[..., :3], x[..., 3])
    return x


def _aligned(name, ref, neutral):
    """Reference warped into the neutral geometry (+ the warp maps and IoU), cached as float32."""
    p = os.path.join(config.cache_dir("align"), name + ".npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["ref"], z["mapx"], z["mapy"], float(z["iou"]), float(z["iou_affine"])
    P = Pairs(ref, neutral, regions=False)
    rn = P.warp(ref).astype(np.float32)
    tmp = p + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, ref=rn, mapx=P.mapx, mapy=P.mapy, iou=P.iou, iou_affine=P.iou_affine)
    os.replace(tmp, p)
    return rn, P.mapx, P.mapy, float(P.iou), float(P.iou_affine)


# ------------------------------------------------------------------ real pairs
LYN = {"R": ("lyn_R_neutral", "lyn_R_ref", "old"), "L": ("lyn_L_neutral", "lyn_L_ref", "left")}


def lyn(side):
    """Lyn sunset pair: neutral, raw reference, reference in neutral geometry, part labels (sunrim polygons)."""
    def make():
        kn, kr, kind = LYN[side]
        n, r = ld(config.data(kn)), ld(config.data(kr))
        rn, mx, my, iou, iou_a = _aligned(f"lyn_{side}", r, n)
        return {"neutral": n, "ref_raw": r, "ref": rn, "mapx": mx, "mapy": my, "iou": iou, "iou_affine": iou_a,
                "kind": kind, "labels": PARTS.labels(kind, n.shape[:2]), "neutral_path": config.data(kn)}
    return memo(("lyn", side), make)


P01_REFS = {"night_rgba": "P01_night_rgba", "night_key": "P01_night_key",
            "lantern_rgba": "P01_lantern_rgba", "lantern_key": "P01_lantern_key"}


def p01(variant):
    """P01 clothed Lyn + a matted night / lantern reference (s598)."""
    def make():
        n, r = ld(config.data("P01_orig")), ld(config.data(P01_REFS[variant]))
        rn, mx, my, iou, iou_a = _aligned(f"P01_{variant}", r, n)
        return {"neutral": n, "ref_raw": r, "ref": rn, "iou": iou, "iou_affine": iou_a, "kind": None, "labels": None,
                "neutral_path": config.data("P01_orig")}
    return memo(("p01", variant), make)


TURN_VIEWS = ("front", "front_right", "right", "back_right", "back", "back_left", "left", "front_left")
TURN_HINT = {"front": "front", "front_right": "front_right", "right": "right", "back_right": "back_right", "back": "back",
             "back_left": "back_left", "left": "left", "front_left": "front_left"}


def drawing(name):
    """A neutral drawing by name: lyn_R, lyn_L, P01, P02, P03, turn_<view>."""
    def make():
        if name == "lyn_R":
            return lyn("R")["neutral"]
        if name == "lyn_L":
            return lyn("L")["neutral"]
        if name in ("P01", "P02", "P03"):
            return ld(config.data(f"{name}_orig"))
        if name.startswith("turn_"):
            return ld(os.path.join(config.data("turn_dir"), f"{name}_rgba.png"))
        raise KeyError(name)
    return memo(("drawing", name), make)


def drawing_path(name):
    if name == "lyn_R":
        return config.data("lyn_R_neutral")
    if name == "lyn_L":
        return config.data("lyn_L_neutral")
    if name in ("P01", "P02", "P03"):
        return config.data(f"{name}_orig")
    return os.path.join(config.data("turn_dir"), f"{name}_rgba.png")


def drawing_labels(name, shape):
    if name == "lyn_R":
        return PARTS.labels("old", shape)
    if name == "lyn_L":
        return PARTS.labels("left", shape)
    return None


# ------------------------------------------------------------------ R_audit semantic masks (Lyn right, neutral geometry)
PART = {"tail": 1, "back_leg": 2, "skirt": 3, "belly": 4, "chest": 5, "front_leg": 6, "left_arm": 7, "right_arm": 8, "head": 9}


def lyn_partlab():
    """R_audit part labels (lab harness, drawn on the reference) carried into the right neutral by the pair's warp."""
    def make():
        L = lyn("R")
        lab = np.load(config.data("lyn_R_partlab"))
        return cv2.remap(lab.astype(np.float32), L["mapx"], L["mapy"], cv2.INTER_NEAREST, borderValue=0).astype(np.uint8)
    return memo("partlab", make)


def audit_masks():
    """R_audit common.lyn_masks (+ 'all' and the all-skin mask used for the dark-skin recolour): wcloth (white cloth),
    dcloth (dark cloth), skin, fur (head / tail light), line, alpha."""
    def make():
        N = lyn("R")["neutral"]
        parts = lyn_partlab()
        lab = to_lab(N[..., :3])
        L, Ch = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
        a = N[..., 3] > 0.5
        mats = materials(N)
        torso = np.isin(parts, [PART["skirt"], PART["belly"], PART["chest"]])
        torso = cv2.dilate(torso.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        headtail = np.isin(parts, [PART["head"], PART["tail"]])
        line = cv2.dilate(mats["line"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ys, xs = np.nonzero(a)
        y0, Hf = ys.min(), ys.max() - ys.min() + 1
        v = (np.arange(N.shape[0])[:, None] - y0) / Hf * np.ones(a.shape, np.float32)
        band = (v > 0.17) & (v < 0.62)
        ht = cv2.dilate(headtail.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        xx = np.arange(N.shape[1])[None, :]
        M = {"wcloth": a & band & ~ht & (Ch < 12) & (L > 55) & ~line & (xx >= 480),
             "dcloth": a & band & ~ht & (mats["dark"] > 0.5) & ~line & (L > 25),
             "skin": a & (mats["skin"] > 0.5) & ~headtail & ~line,
             "fur": a & headtail & (mats["light"] > 0.5) & ~line,
             "line": a & mats["line"], "alpha": a}
        M["all"] = a
        M["skin_all"] = a & (mats["skin"] > 0.5) & (parts != 1)
        return M
    return memo("audit_masks", make)


def class_masks(F):
    """R_audit cases.class_masks: the tool's own colour classes (for frames without semantic masks)."""
    mats = materials(F)
    a = F[..., 3] > 0.5
    L = to_lab(F[..., :3])[..., 0]
    line = cv2.dilate(mats["line"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return {"skin": a & (mats["skin"] > 0.5) & ~line, "light": a & (mats["light"] > 0.5) & ~line,
            "dark": a & (mats["dark"] > 0.5) & ~line & (L > 25), "all": a}
