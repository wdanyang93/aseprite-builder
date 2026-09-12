"""Character frame metrics.

Measured on the real Lyn dig sheets (v1.11): silhouette-based mirror tests are *not* enough to
find a reversed frame in a dig cycle, because the pose changes so much between raise and strike
that the mirrored neighbour is no more similar than the unmirrored one (IoU 0.26 vs 0.71 at the
actual flip). What does separate the frames cleanly is **where the eyes sit relative to the head**
and **which side the tool is on**:

    dig_A  eye_dx  -6.4  -5.3  +33.9  +14.6  +12.4  +3.1  +8.4  -4.6
                    L     L      R      R      R     R     R     L      <- flips at 2->3 and 7->8

So the facing signal is colour-class based (eyes / tool), and the silhouette signal is only a
supporting vote. Everything here is pose-invariant or explicitly labelled pose-dependent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.signal import find_peaks

WHITE_KEY_LEVEL = 235          # sheets on white: sum(rgb) < 3*level counts as foreground
DARK_KEY_LEVEL = 28            # sheets on black: sum(rgb) > 3*level counts as foreground
MIN_ALPHA = 16
ALPHA_KEEP = 8                 # matte alpha at or above this is foreground
HEAD_BAND = 0.30               # top 30 % of the figure is the head band
EYE_MIN_PIXELS = 15
TOOL_MIN_PIXELS = 50
CORE_ERODE_DIVISOR = 40
FACING_DEADZONE_PX = 4.0       # |eye_dx| below this = frontal / undecided (real sheets: true flips are >= 4.6)


@dataclass
class Frame:
    index: int                 # 1-based
    rgba: np.ndarray           # cropped to the figure's bbox
    mask: np.ndarray           # foreground, same size as rgba
    sheet_x0: int              # column where this frame starts in the sheet
    sheet_bbox: list[int]      # [x, y, w, h] in sheet pixels
    stray_components: list[list[int]] = field(default_factory=list)   # dropped badges/specks
    split_method: str = ""     # "gutter" or "equal" (set on frame 1)
    matte_info: dict[str, Any] = field(default_factory=dict)   # set on frame 1


@dataclass
class FrameMetrics:
    index: int
    width: int
    height: int
    area: int
    ground_y: int              # bottom of the foreground in the frame crop
    ground_x: float            # centre of the lowest foreground rows
    centroid: list[float]
    core_area: int             # eroded silhouette (limbs removed)
    eye_dx: float | None       # eye centroid minus head centroid; None = not found
    tool_dx: float | None      # tool centroid minus body centroid; None = not found
    mean_lab: list[float]
    skin_lab: list[float] | None   # skin class only: the stable palette invariant (dE 0-4 across a dig cycle)
    facing: str                # "L" | "R" | "-"
    tool_side: str             # "L" | "R" | "-"
    pose_dependent: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "width": self.width, "height": self.height, "area": self.area,
            "ground_y": self.ground_y, "ground_x": round(self.ground_x, 2),
            "centroid": [round(v, 2) for v in self.centroid], "core_area": self.core_area,
            "eye_dx": None if self.eye_dx is None else round(self.eye_dx, 2),
            "tool_dx": None if self.tool_dx is None else round(self.tool_dx, 2),
            "mean_lab": [round(v, 2) for v in self.mean_lab],
            "skin_lab": None if self.skin_lab is None else [round(v, 2) for v in self.skin_lab],
            "facing": self.facing, "tool_side": self.tool_side,
            "pose_dependent": self.pose_dependent,
        }


# ------------------------------------------------------------------ input --

def detect_background(rgb: np.ndarray) -> str:
    """"alpha" | "light" | "dark". Sheets arrive on white, on a baked checkerboard, and on black.

    The corners are the only pixels guaranteed not to be the character; their median decides.
    A sheet delivered on black keyed as if it were white gives a 0.909 foreground ratio - the
    whole canvas - and every frame comes out full height.
    """
    corners = np.array([rgb[4, 4], rgb[4, -5], rgb[-5, 4], rgb[-5, -5]], np.float32)
    return "dark" if float(np.median(corners.mean(axis=1))) < 128.0 else "light"


def foreground_mask(rgba: np.ndarray) -> np.ndarray:
    """Native alpha when present, otherwise a light- or dark-background key."""
    if rgba.shape[2] == 4 and int(rgba[..., 3].min()) < 255:
        return rgba[..., 3] > MIN_ALPHA
    rgb = rgba[..., :3]
    if detect_background(rgb) == "dark":
        return rgb.astype(int).sum(axis=2) > 3 * DARK_KEY_LEVEL
    return rgb.astype(int).sum(axis=2) < 3 * WHITE_KEY_LEVEL


COMPONENT_AREA_MIN_RATIO = 0.55   # components must be comparable in size to be frames
SEVERANCE_WEIGHT = 12.0        # cutting through a body dominates panel-evenness
STRAY_AREA_RATIO = 0.25        # a component below this share of the largest one is not the figure


def drop_stray_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Keep only components comparable in size to the largest one.

    Exporters stamp a badge in a corner ("AI", a logo, a watermark). It is opaque, so it survives
    keying, and because it sits in the last column it silently becomes part of frame 8's bounding
    box. Measured on the supplied walk sheets: frame 8 came out 1017-1536 px tall against ~570 px
    for frames 1-7, i.e. a 176 % height "deviation" that was really a 94x69 px badge.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask, []
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    keep = np.zeros_like(mask)
    dropped: list[list[int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= STRAY_AREA_RATIO * biggest:
            keep |= labels == i
        else:
            dropped.append([int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]), area])
    return keep, dropped


def _frames_from_components(rgba: np.ndarray, mask: np.ndarray, count: int) -> list[Frame] | None:
    """One frame per connected component, when the component count matches exactly.

    Returns None unless there are exactly ``count`` components of comparable size, so a sheet whose
    figures really do touch still falls through to cutting.
    """
    n, labels, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n - 1 < count:
        return None
    areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)]
    areas.sort(reverse=True)
    chosen = areas[:count]
    smallest, largest = chosen[-1][0], chosen[0][0]
    if smallest < largest * COMPONENT_AREA_MIN_RATIO:
        return None
    ordered = sorted(chosen, key=lambda a: int(stats[a[1], cv2.CC_STAT_LEFT]))
    frames: list[Frame] = []
    for index, (_area, label) in enumerate(ordered, start=1):
        region = labels == label
        ys, xs = np.nonzero(region)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = rgba[y0:y1, x0:x1].copy()
        local = region[y0:y1, x0:x1]
        if crop.shape[2] == 3:
            crop = np.dstack([crop, local.astype(np.uint8) * 255])
        else:
            crop[..., 3] = np.where(local, crop[..., 3], 0)
        frames.append(Frame(index=index, rgba=crop, mask=local, sheet_x0=x0,
                            sheet_bbox=[x0, y0, x1 - x0, y1 - y0]))
    return frames


def _cut_candidates(mask: np.ndarray, count: int) -> list[tuple[str, list[int]]]:
    """Two independent ways to place the count-1 cuts, so one can rescue the other."""
    height, width = mask.shape
    profile = mask.sum(axis=0).astype(float)
    out: list[tuple[str, list[int]]] = []

    smooth = np.convolve(profile, np.ones(15) / 15.0, mode="same")
    valleys = smooth.max() - smooth
    peaks, _ = find_peaks(valleys, distance=max(8, int(width / (count * 1.5))),
                          prominence=smooth.max() * 0.05)
    if len(peaks) >= count - 1:
        strongest = np.argsort(-valleys[peaks])[:count - 1]
        out.append(("gutter", sorted(int(peaks[i]) for i in strongest)))

    cols = np.where(mask.any(axis=0))[0]
    if cols.size:
        x0, x1 = int(cols.min()), int(cols.max())
        step = (x1 - x0 + 1) / float(count)
        out.append(("equal", [int(round(x0 + i * step)) for i in range(1, count)]))
    return out


def _score_cuts(mask: np.ndarray, cuts: list[int]) -> float:
    """Lower is better. Severing a body is the dominant term, not a tiebreak.

    The previous weighting normalised the severed pixel count by the *panel area*, which made it
    vanish: on a real sheet the equal-division cuts sliced 46-94 px of body each and still scored
    0.030 against the gutter option's 0.055, so every one of eight frames lost a tail or a hand.
    Severance is now measured against the figure's own mean column occupancy - a cut through a
    limb is then a large number, as it should be - and it is the leading term.
    """
    height, width = mask.shape
    edges = [0] + list(cuts) + [width]
    areas = []
    for i in range(len(edges) - 1):
        sub = mask[:, edges[i]:edges[i + 1]]
        if not sub.any():
            return float("inf")
        areas.append(float(sub.sum()))
    areas_arr = np.array(areas)
    spread = float(areas_arr.std() / max(1.0, areas_arr.mean()))
    occupied = mask.sum(axis=0)
    reference = float(occupied[occupied > 0].mean()) if (occupied > 0).any() else 1.0
    severed = float(np.mean([occupied[c] for c in cuts]) / max(1.0, reference))
    return spread + SEVERANCE_WEIGHT * severed


def _refine_cuts(mask: np.ndarray, cuts: list[int], window_ratio: float = 0.35) -> list[int]:
    """Nudge each cut to the emptiest column nearby, so a cut lands in a gap when one exists."""
    occupied = mask.sum(axis=0)
    width = mask.shape[1]
    if len(cuts) < 1:
        return cuts
    pitch = width / float(len(cuts) + 1)
    window = max(2, int(pitch * window_ratio))
    out = []
    for c in cuts:
        lo, hi = max(1, c - window), min(width - 1, c + window)
        segment = occupied[lo:hi]
        out.append(lo + int(np.argmin(segment)) if segment.size else c)
    return sorted(set(out))


def refine_with_matte(rgba: np.ndarray, coarse: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Rescue the character from a same-value background using region matting.

    Lyn's white bandeau, white hair and white tail sit at luminance ~234 while the exporter's
    background sits at 248-254. A per-pixel threshold cannot separate 20 levels reliably: at 235
    the coarse key punched 45,095 px of holes into the figure (8.9 % of its area) and broke it into
    56 components, so the feet became small blobs and were discarded as strays. Region matting
    keeps them because they are *connected* to the body even where their colour matches the
    background. Measured on one SE frame: holes 9.6 % -> 0.8 %, foreground 71,291 -> 77,365 px.

    Then any background still enclosed by the figure is filled - a hole inside a body is a keying
    artefact, not a see-through gap.
    """
    from ..matte import grabcut_matte

    height, width = coarse.shape
    kept, _stray = drop_stray_components(coarse)
    n, labels, stats, _c = cv2.connectedComponentsWithStats(kept.astype(np.uint8), 8)
    if n <= 1:
        return coarse, {"matte": "skipped_no_components"}
    refined = np.zeros_like(coarse)
    key_alpha = (coarse.astype(np.uint8) * 255)
    per_component = 0
    for i in range(1, n):
        x, y, w, h = [int(stats[i, k]) for k in
                      (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
        result = grabcut_matte(rgba[..., :3], [x, y, w, h], key_alpha)
        cx, cy, cw, ch = result.crop_box
        region = refined[cy:cy + ch, cx:cx + cw]
        np.logical_or(region, result.alpha >= ALPHA_KEEP, out=region)
        per_component += 1
    # fill enclosed background
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = refined.astype(np.uint8)
    inv = (1 - padded).astype(np.uint8)
    cv2.floodFill(inv, np.zeros((height + 4, width + 4), np.uint8), (0, 0), 2)
    filled = ((padded > 0) | (inv != 2))[1:-1, 1:-1]
    holes_filled = int(filled.sum() - refined.sum())
    return filled, {"matte": "grabcut_per_component", "components": per_component,
                    "holes_filled_px": holes_filled,
                    "coarse_foreground_px": int(coarse.sum()), "refined_foreground_px": int(filled.sum())}


def split_sheet(rgba: np.ndarray, count: int, *, matte: bool = True) -> list[Frame]:
    """Cut a horizontal strip into ``count`` frames.

    Two strategies are scored against each other and the better one wins. Gutter valleys are the
    natural choice, but panels overlap whenever a raised pickaxe or flying debris crosses into the
    neighbouring column - on the supplied sheets two frames merged into one blob and the valley
    search found only 6 of the 7 cuts, producing a 131 px "frame". Equal division of the content
    span rescues those, and the score (panel-area spread + how much content each cut severs)
    decides without the caller having to know which case they have.
    """
    mask = foreground_mask(rgba)
    matte_info: dict[str, Any] = {"matte": "native_alpha" if (rgba.shape[2] == 4 and int(rgba[..., 3].min()) < 255) else "key_only"}
    if matte and matte_info["matte"] == "key_only":
        mask, matte_info = refine_with_matte(rgba, mask)
    mask, stray = drop_stray_components(mask)
    height, width = mask.shape

    # If the figures are separate components, use them directly. This is the only lossless option
    # when they overlap horizontally: the real sheet had only 4 fully empty interior columns for
    # the 7 cuts an 8-frame split needs, so every cut severed 46-94 px of a tail or a hand. The
    # components, however, numbered exactly 8 - the figures overlap visually without touching.
    component_frames = _frames_from_components(rgba, mask, count)
    if component_frames is not None:
        component_frames[0].split_method = "components"
        component_frames[0].matte_info = matte_info
        if stray:
            component_frames[0].stray_components = stray
        return component_frames

    candidates = _cut_candidates(mask, count)
    if not candidates:
        raise ValueError("전경을 찾지 못해 프레임을 나눌 수 없습니다.")
    # Each candidate is also offered in a locally refined form: a cut pulled to the emptiest
    # nearby column. On overlapping figures that is the difference between losing a hand and not.
    expanded: list[tuple[str, list[int]]] = []
    for name, cut_set in candidates:
        expanded.append((name, cut_set))
        refined = _refine_cuts(mask, cut_set)
        if len(refined) == len(cut_set) and refined != cut_set:
            expanded.append((f"{name}_refined", refined))
    method, cuts = min(expanded, key=lambda mc: _score_cuts(mask, mc[1]))

    edges = [0] + list(cuts) + [width]
    frames: list[Frame] = []
    for i in range(count):
        x0, x1 = edges[i], edges[i + 1]
        sub_mask = mask[:, x0:x1]
        ys, xs = np.nonzero(sub_mask)
        if xs.size == 0:
            raise ValueError(f"프레임 {i + 1} 에 전경이 없습니다.")
        bx0, bx1 = int(xs.min()), int(xs.max()) + 1
        by0, by1 = int(ys.min()), int(ys.max()) + 1
        crop = rgba[by0:by1, x0 + bx0:x0 + bx1].copy()
        crop_mask = sub_mask[by0:by1, bx0:bx1]
        if crop.shape[2] == 3 or int(crop[..., 3].min()) == 255:
            crop = np.dstack([crop[..., :3], crop_mask.astype(np.uint8) * 255])
        frames.append(Frame(index=i + 1, rgba=crop, mask=crop_mask, sheet_x0=x0,
                            sheet_bbox=[x0 + bx0, by0, bx1 - bx0, by1 - by0]))
    frames[0].split_method = method
    frames[0].matte_info = matte_info
    if stray:
        frames[0].stray_components = stray
    return frames


# --------------------------------------------------------------- classes --

def colour_classes(rgb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Cheap HSV classes for this character design. Tuned on the real sheets; documented so the
    next character can be given its own ranges."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return {
        "tool":   mask & (s < 60) & (v < 140),                            # dark grey metal
        "handle": mask & (h >= 5) & (h <= 20) & (s > 80) & (v > 60) & (v < 200),
        "eyes":   mask & (h >= 8) & (h <= 25) & (s > 120) & (v > 120),     # orange irises
        "skin":   mask & (h >= 5) & (h <= 20) & (s > 20) & (s < 90) & (v > 170),
    }


def measure(frame: Frame) -> FrameMetrics:
    mask = frame.mask
    rgb = frame.rgba[..., :3].copy()
    rgb[~mask] = 255
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    area = int(mask.sum())
    cx, cy = float(xs.mean()), float(ys.mean())

    # ground: centre of the lowest 3 % of foreground rows
    ground_y = int(ys.max())
    low_rows = ys >= ground_y - max(1, int(height * 0.03))
    ground_x = float(xs[low_rows].mean())

    # Erosion radius. height/22 removed a walking figure's separated legs entirely, turning a
    # pose change into a 47 % "scale deviation"; height/40 keeps limbs and still strips fine
    # detail. Measured max |core deviation| across a cycle: walk 47%->21%, dig 12%->11%.
    core_k = max(3, int(height / CORE_ERODE_DIVISOR)) | 1
    core = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_k, core_k))) > 0

    classes = colour_classes(rgb, mask)
    head_band = np.zeros_like(mask)
    head_band[: int(height * HEAD_BAND)] = True
    head = mask & head_band
    head_cx = float(np.nonzero(head)[1].mean()) if head.any() else cx
    eyes = classes["eyes"] & head_band
    eye_dx = float(np.nonzero(eyes)[1].mean() - head_cx) if int(eyes.sum()) >= EYE_MIN_PIXELS else None
    tool = classes["tool"] | classes["handle"]
    tool_dx = float(np.nonzero(tool)[1].mean() - cx) if int(tool.sum()) >= TOOL_MIN_PIXELS else None

    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    mean_lab = [float(v) for v in lab_img[mask].astype(np.float32).mean(axis=0)]
    skin = classes["skin"]
    skin_lab = [float(v) for v in lab_img[skin].astype(np.float32).mean(axis=0)] if int(skin.sum()) > 50 else None

    if eye_dx is None or abs(eye_dx) < FACING_DEADZONE_PX:
        facing = "-"
    else:
        facing = "L" if eye_dx < 0 else "R"
    tool_side = "-" if tool_dx is None else ("L" if tool_dx < 0 else "R")

    return FrameMetrics(
        index=frame.index, width=width, height=height, area=area, ground_y=ground_y, ground_x=ground_x,
        centroid=[cx, cy], core_area=int(core.sum()), eye_dx=eye_dx, tool_dx=tool_dx, mean_lab=mean_lab,
        skin_lab=skin_lab, facing=facing, tool_side=tool_side,
        pose_dependent={"bbox_width": width, "aspect": round(width / max(1, height), 3)},
    )


def silhouette_iou(a: np.ndarray, b: np.ndarray, *, mirror_a: bool = False) -> float:
    """IoU of two masks aligned by bottom-centre (ground pivot)."""
    if mirror_a:
        a = a[:, ::-1]
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    canvas_a, canvas_b = np.zeros((h, w), bool), np.zeros((h, w), bool)
    for src, dst in ((a, canvas_a), (b, canvas_b)):
        oy, ox = h - src.shape[0], (w - src.shape[1]) // 2
        dst[oy:oy + src.shape[0], ox:ox + src.shape[1]] = src
    union = int((canvas_a | canvas_b).sum())
    return float((canvas_a & canvas_b).sum() / max(1, union))
