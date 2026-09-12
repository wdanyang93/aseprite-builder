"""Cell classifier, derived from one hand-labelled frame.

Frame 1 of the real sheet was labelled by hand, seven cells:

    cell  1   ear                 L* 205  texture 24.8  y 0.01-0.10
    cell  2   head                L* 221  texture 13.3  y 0.02-0.27
    cell 15   torso + near arm    L* 229  texture 10.9  cy 0.43
    cell 21   far arm             L* 204  texture 19.5  cy 0.50  x 0.74-0.95
    cell 29   tail                L* 226  texture  4.3  x0 0.02  thickness 12.3
    cell 30   near leg            L* 229  texture  1.3  cy 0.74  thickness 13.0
    cell 37   far leg             L* 204  texture  3.2  cy 0.80

Three separations fall straight out of those numbers and none of them needed a threshold search:

* **L\\* 229 versus 204** splits near from far, with no overlap. The art shades the far limb.
* **Vertical position** splits ear (top 10 %), head (top quarter), torso (middle), legs (bottom).
* **Reaching the left frame edge** plus low texture and high thickness is the tail: it starts at
  x 0.02 where no leg does.

So the classifier is written as those rules rather than fitted, and the labelled frame is kept as
a regression fixture: if a change stops reproducing all seven names, the change is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

PARTS = ("ear", "head", "torso", "arm_near", "arm_far", "tail", "leg_near", "leg_far")

# Boundaries read off the labelled frame, stated as the midpoint between the two classes.
EAR_MAX_Y0 = 0.12          # ear y0 0.01, head y0 0.02 - but the ear ENDS by 0.10
EAR_MAX_Y1 = 0.16
HEAD_MAX_CY = 0.22         # head cy 0.14, torso cy 0.43
LEG_MIN_CY = 0.60          # legs cy 0.74 / 0.80, torso 0.43, far arm 0.50
NEAR_FAR_L = 216.0         # near 229, far 204
TAIL_MAX_X0 = 0.12         # tail x0 0.02, far leg 0.21, near leg 0.50
TAIL_MAX_TEXTURE = 8.0     # tail 4.3; the far arm at the same height has 19.5
TAIL_MIN_THICKNESS = 9.0   # tail 12.3
ARM_MAX_SHARE = 0.08       # far arm 2.6 % of the figure; torso 18.9 %
OVERSIZED_SHARE = 0.26     # a cell this large holds more than one part (torso 18.9 %, merged 35 %)
GARMENT_MAX_SAT = 22       # the white bandeau and briefs
GARMENT_MIN_VAL = 230
GARMENT_MIN_PX = 400
GARMENT_BLOB_MIN_PX = 300
# The briefs measured y1 0.55-0.57 and width 0.19-0.23 on every frame; these bounds reject a
# stray white highlight claiming to be the garment.
BRIEFS_MIN_Y = 0.45
BRIEFS_MAX_Y = 0.66
BRIEFS_MIN_WIDTH = 0.14
BRIEFS_MAX_WIDTH = 0.32
SPLIT_MIN_PX = 1200
OPPOSITE_MIN_GAP = 0.12    # near and far limbs sit this far apart in x during a walk
MERGED_LEG_SHARE = 0.26    # one leg runs 19-21 % of the figure; more than this is two


@dataclass
class CellFeatures:
    cell: int
    px: int
    share: float
    x0: float
    x1: float
    y0: float
    y1: float
    cx: float
    cy: float
    lightness: float
    texture: float
    thickness: float
    elongation: float

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def measure_cells(rgba: np.ndarray, mask: np.ndarray, labels: np.ndarray,
                  min_px: int = 400) -> list[CellFeatures]:
    """The same seven numbers that were shown to the person, for every cell."""
    from .segment import _skeletonise

    height, width = mask.shape
    lightness_map = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB)[..., 0]
    grey = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(grey, (9, 9))
    texture_map = np.sqrt(np.maximum(0.0, cv2.blur(grey * grey, (9, 9)) - mean * mean))
    figure_px = max(1, int(mask.sum()))
    out: list[CellFeatures] = []
    for cell in range(1, int(labels.max()) + 1):
        sel = labels == cell
        px = int(sel.sum())
        if px < min_px:
            continue
        ys, xs = np.nonzero(sel)
        skeleton, distance = _skeletonise(sel)
        widths = distance[skeleton] if skeleton.any() else np.array([1.0])
        thickness = float(np.mean(widths))
        span = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
        out.append(CellFeatures(
            cell=cell, px=px, share=px / figure_px,
            x0=float(xs.min()) / width, x1=float(xs.max()) / width,
            y0=float(ys.min()) / height, y1=float(ys.max()) / height,
            cx=float(xs.mean()) / width, cy=float(ys.mean()) / height,
            lightness=float(np.median(lightness_map[sel])),
            texture=float(np.median(texture_map[sel])),
            thickness=thickness,
            elongation=span / max(1.0, thickness * 2.0)))
    out.sort(key=lambda c: -c.px)
    return out


def find_garment_figure(rgba: np.ndarray, mask: np.ndarray) -> dict[str, Any] | None:
    """The bandeau and briefs, found on the whole figure rather than inside one cell.

    Searching inside a single cell only worked where the merged cell happened to contain the
    clothing; frames 3 and 5 reported "no garment" simply because nobody looked. Run on the whole
    figure the same key finds both garments on all eight frames, and remarkably consistently:

        bandeau  y 0.33-0.40   briefs y 0.51-0.56   briefs width 0.19-0.23 of the frame

    The briefs' lower edge is therefore a reliable hip line for every frame, not just the ones
    whose cells merged.
    """
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    cloth = mask & (hsv[..., 1] < GARMENT_MAX_SAT) & (hsv[..., 2] > GARMENT_MIN_VAL)
    if int(cloth.sum()) < GARMENT_MIN_PX:
        return None
    count, labels, stats, _c = cv2.connectedComponentsWithStats(cloth.astype(np.uint8), 8)
    blobs = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < GARMENT_BLOB_MIN_PX:
            continue
        blobs.append({"y0": int(stats[i, cv2.CC_STAT_TOP]),
                      "y1": int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
                      "x0": int(stats[i, cv2.CC_STAT_LEFT]),
                      "x1": int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
                      "area": int(stats[i, cv2.CC_STAT_AREA]), "label": i})
    if len(blobs) < 2:
        return None
    blobs.sort(key=lambda b: b["y0"])
    height, width = mask.shape
    top, bottom = blobs[0], blobs[-1]
    briefs_width = (bottom["x1"] - bottom["x0"]) / width
    if not (BRIEFS_MIN_Y <= bottom["y1"] / height <= BRIEFS_MAX_Y):
        return None
    if not (BRIEFS_MIN_WIDTH <= briefs_width <= BRIEFS_MAX_WIDTH):
        return None
    return {"bandeau": {"y0": top["y0"] / height, "y1": top["y1"] / height},
            "briefs": {"y0": bottom["y0"] / height, "y1": bottom["y1"] / height,
                       "x0": bottom["x0"] / width, "x1": bottom["x1"] / width,
                       "width": briefs_width},
            "hip_row": bottom["y1"], "hip_y": bottom["y1"] / height,
            "briefs_mask": labels == bottom["label"],
            "blobs": len(blobs)}


def find_garment(rgba: np.ndarray, region: np.ndarray) -> dict[str, Any] | None:
    """The white top and bottom, and the hip line under the bottom.

    The owner pointed out that the clothing is visible and separable, and it is: inside the
    oversized cell of frames 6 and 7 a low-saturation bright key finds exactly two blobs, the
    bandeau starting at y 0.33 and the briefs ending at y 0.56, on every frame measured. The lower
    edge of the briefs is the hip, and that is where a torso stops and a leg starts - a leg cannot
    reach the neck.
    """
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    cloth = region & (hsv[..., 1] < GARMENT_MAX_SAT) & (hsv[..., 2] > GARMENT_MIN_VAL)
    if int(cloth.sum()) < GARMENT_MIN_PX:
        return None
    count, _labels, stats, _c = cv2.connectedComponentsWithStats(cloth.astype(np.uint8), 8)
    blobs = sorted(((int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
                     int(stats[i, cv2.CC_STAT_AREA]))
                    for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= GARMENT_BLOB_MIN_PX))
    if len(blobs) < 2:
        return None
    height = region.shape[0]
    return {"top_y": blobs[0][0] / height, "hip_y": blobs[-1][1] / height,
            "hip_row": blobs[-1][1], "blobs": len(blobs),
            "px": int(cloth.sum())}


def split_torso_and_leg(rgba: np.ndarray, region: np.ndarray,
                        reference_x: float | None = None,
                        hip_row: int | None = None) -> dict[str, Any] | None:
    """Cut an oversized cell into torso and the near limb at the hip line.

    Frames 6 and 7 had no line between the torso and the near leg, so one cell held 35 % of the
    figure and the near leg vanished from the output. Cutting at the garment's lower edge returns
    the torso to 17-18 %, matching the hand-labelled frame's 18.9 %, and yields a lower piece whose
    x centre (0.70) sits opposite the far leg's (0.46-0.51) exactly as a walk cycle requires.
    """
    if hip_row is None:
        garment = find_garment(rgba, region)
        if garment is None:
            return None
        hip_row = garment["hip_row"]
    else:
        garment = {"hip_row": hip_row, "hip_y": hip_row / region.shape[0], "source": "figure"}
    hip = hip_row
    upper = region.copy()
    upper[hip:] = False
    lower = region.copy()
    lower[:hip] = False
    if int(lower.sum()) < SPLIT_MIN_PX or int(upper.sum()) < SPLIT_MIN_PX:
        return None
    out = {"torso": upper, "limb": lower, "hip_y": garment["hip_y"], "garment": garment}
    if reference_x is not None:
        ys, xs = np.nonzero(lower)
        out["limb_cx"] = float(xs.mean()) / region.shape[1]
        out["opposite_reference"] = bool(abs(out["limb_cx"] - reference_x) > OPPOSITE_MIN_GAP)
    return out


def classify_cell(cell: CellFeatures) -> tuple[str, float, str]:
    """(part, confidence, reason). Rules only - nothing fitted, nothing searched."""
    depth = "near" if cell.lightness >= NEAR_FAR_L else "far"

    if cell.y1 <= EAR_MAX_Y1 and cell.y0 <= EAR_MAX_Y0:
        return "ear", 0.9, f"위쪽 끝 y1={cell.y1:.2f}"
    if cell.cy <= HEAD_MAX_CY:
        return "head", 0.85, f"중심 y={cell.cy:.2f}"
    if (cell.x0 <= TAIL_MAX_X0 and cell.texture <= TAIL_MAX_TEXTURE
            and cell.thickness >= TAIL_MIN_THICKNESS):
        return "tail", 0.9, f"왼쪽 끝 x0={cell.x0:.2f}, 질감 {cell.texture:.1f}, 두께 {cell.thickness:.1f}"
    if cell.cy >= LEG_MIN_CY:
        return f"leg_{depth}", 0.8, f"중심 y={cell.cy:.2f}, L*={cell.lightness:.0f}"
    if cell.share <= ARM_MAX_SHARE:
        return f"arm_{depth}", 0.75, f"비중 {cell.share*100:.1f}%, L*={cell.lightness:.0f}"
    return "torso", 0.7, f"중심 y={cell.cy:.2f}, 비중 {cell.share*100:.1f}%"


def classify_frame(rgba: np.ndarray, mask: np.ndarray,
                   labels: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Name every cell, splitting any cell that plainly holds more than one part.

    Three signals the owner identified, in the order they are applied:

    1. **The clothing is separable.** Two white blobs; the lower one's bottom edge is the hip.
    2. **Length from the neck and from the ground.** A leg does not reach the neck, so the piece
       below the hip is a limb and the piece above it is the torso.
    3. **The near limbs sit opposite the far ones.** With the far leg already named, the split
       piece's x centre confirms it: measured 0.70 against the far leg's 0.46-0.51.
    """
    features = measure_cells(rgba, mask, labels)
    garment = find_garment_figure(rgba, mask)
    # the far limbs are named first so their position can serve as the reference in step 3
    far_reference: float | None = None
    for cell in features:
        part, _c, _r = classify_cell(cell)
        if part == "leg_far":
            far_reference = cell.cx
            break

    regions: dict[str, np.ndarray] = {}
    report: list[dict[str, Any]] = []
    for cell in features:
        part, confidence, reason = classify_cell(cell)
        sel = labels == cell.cell
        if part == "torso" and cell.share >= OVERSIZED_SHARE:
            split = split_torso_and_leg(rgba, sel, reference_x=far_reference,
                                        hip_row=garment["hip_row"] if garment else None)
            if split is not None:
                regions["torso"] = regions.get("torso", np.zeros_like(mask)) | split["torso"]
                limb = "leg_near" if split.get("opposite_reference", True) else "leg_far"
                regions[limb] = regions.get(limb, np.zeros_like(mask)) | split["limb"]
                report.append(dict(cell.to_dict(), part=f"torso+{limb}", confidence=0.8,
                                   reason=(f"비중 {cell.share*100:.0f}% 과대 -> 하의 하단 "
                                           f"y={split['hip_y']:.2f} 에서 분할, "
                                           f"다리 중심 x={split.get('limb_cx', 0):.2f}")))
                continue
            reason += f" (과대하지만 옷 경계를 못 찾아 분할 보류)"
        regions[part] = regions.get(part, np.zeros_like(mask)) | sel
        report.append(dict(cell.to_dict(), part=part, confidence=confidence, reason=reason))
    # The hip line applies to the whole figure, not just merged cells. Any pixel a cell gave to a
    # leg but which sits above the briefs belongs to the torso, and any torso pixel below them
    # belongs to a leg - "a leg does not reach the neck", as the owner put it. Applying this to
    # every frame is what let frames 3 and 5 use the garment at all.
    if garment is not None:
        hip = garment["hip_row"]
        above = np.zeros_like(mask)
        above[:hip] = True
        for name in [n for n in regions if n.startswith("leg")]:
            stray = regions[name] & above
            if stray.any():
                regions[name] = regions[name] & ~above
                regions["torso"] = regions.get("torso", np.zeros_like(mask)) | stray
        torso = regions.get("torso")
        if torso is not None:
            below = torso & ~above
            if below.any():
                target = _nearest_leg_name(regions, below, mask) or "leg_near"
                regions["torso"] = torso & above
                regions[target] = regions.get(target, np.zeros_like(mask)) | below
        # the briefs themselves are clothing on the torso
        regions["torso"] = regions.get("torso", np.zeros_like(mask)) | garment["briefs_mask"]
        for name in list(regions):
            if name != "torso":
                regions[name] = regions[name] & ~garment["briefs_mask"]
        regions = {k: v for k, v in regions.items() if v.any()}

    # A leg region far larger than its siblings holds both legs: frame 7 reported 30 % where every
    # other frame sat at 19-21 %. Shading splits it, exactly as it does for a merged limb chain -
    # the far leg is the darker half. The owner's third point is the check: with the left arm
    # forward the left leg must trail, so the far leg's foot lands behind the near one.
    for name in [n for n in regions if n.startswith("leg")]:
        share = int(regions[name].sum()) / max(1, int(mask.sum()))
        if share < MERGED_LEG_SHARE:
            continue
        from .segment import split_merged_limb
        split = split_merged_limb(rgba, regions[name])
        if split is None:
            continue
        regions.pop(name)
        regions["leg_near"] = regions.get("leg_near", np.zeros_like(mask)) | split["near"]
        regions["leg_far"] = regions.get("leg_far", np.zeros_like(mask)) | split["far"]
        report.append({"cell": -1, "part": "leg_near+leg_far", "confidence": 0.7,
                       "reason": (f"{name} 이 {share*100:.0f}% 로 과대 -> 명암 L*<"
                                  f"{split['threshold']:.0f} 로 분리")})
        break

    # anything the cell pass missed (line pixels, specks) joins the region it borders most
    owned = np.logical_or.reduce(list(regions.values())) if regions else np.zeros_like(mask)
    leftover = mask & ~owned
    if leftover.any() and regions:
        names = list(regions)
        grown = np.stack([cv2.dilate(regions[n].astype(np.uint8),
                                     np.ones((5, 5), np.uint8)).astype(np.int16) for n in names])
        winner = np.argmax(grown, axis=0)
        touched = grown.max(axis=0) > 0
        for i, name in enumerate(names):
            regions[name] = regions[name] | (leftover & touched & (winner == i))
    return regions, report


def _nearest_leg_name(regions: dict[str, np.ndarray], piece: np.ndarray,
                      mask: np.ndarray) -> str | None:
    """Which existing leg region the stray piece touches most."""
    ring = (cv2.dilate(piece.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~piece & mask
    best, best_len = None, 0
    for name, region in regions.items():
        if not name.startswith("leg"):
            continue
        shared = int((ring & region).sum())
        if shared > best_len:
            best, best_len = name, shared
    return best


LABELLED_FRAME_1 = {
    "source": "codex_a frame 1, labelled by the project owner",
    "cells": {1: "ear", 2: "head", 15: "torso", 21: "arm_far", 29: "tail",
              30: "leg_near", 37: "leg_far"},
    "note": ("칸 15는 몸통과 near 팔이 한 칸이다 - 둘 사이에 선이 없다. "
             "분리하려면 near 팔을 따로 그려 받아야 한다."),
}
