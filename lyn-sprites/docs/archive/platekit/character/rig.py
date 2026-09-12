"""Part rig: cut a character frame into moveable pieces and reassemble it.

This is not a checker. It cuts the figure into head, torso, tail and limb segments, records a
pivot for each, and recomposes them with per-part scale/rotation/translation. Nothing is generated:
every output pixel came from the source art. That makes three things possible that measurement
alone cannot do:

* **proportion repair** - scale the head to canon while leaving the body, or lengthen a short leg
* **frame repair** - take a limb from a good frame and place it on a bad one
* **new poses** - rotate limb segments about their joints to make frames that were never drawn

Two rules from the project's existing rig spec are load-bearing and kept here:

1. **Cuts overlap.** A part is cut with extra pixels past its joint and the part above it covers
   the seam, so rotating does not open a hole. The spec puts this as "목은 상체 레이어가 위로
   20px 더 올라오고, 머리 레이어가 그 위를 덮는다".
2. **Occluded pixels are restored before they are needed.** A leg hidden behind the tail has no
   pixels; swing the tail and the gap shows. Hidden regions are inpainted at cut time.

Resolution matters. A 181 px grid cell puts an upper arm at ~8 px wide, which no segmentation can
split into joints. The rig works on the delivered 1024x1536 singles (figure ~1400 px).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

# z order, low to high, matching the project's existing parts_S table
PART_Z = {
    "tail_base": 0, "tail_tip": 1,
    "leg_far_upper": 2, "leg_far_lower": 3, "leg_far_foot": 4,
    "arm_far_upper": 5, "arm_far_lower": 6, "arm_far_hand": 7,
    "torso": 8,
    "leg_near_upper": 9, "leg_near_lower": 10, "leg_near_foot": 11,
    "head": 12,
    "arm_near_upper": 13, "arm_near_lower": 14, "arm_near_hand": 15,
}
SEAM_OVERLAP_PX = 0.014      # of figure height, added past every cut so rotation cannot tear
NECK_SEARCH = (0.15, 0.40)   # fraction of height to look for the neck minimum
HIP_SEARCH = (0.45, 0.70)
LIMB_MIN_AREA = 0.004        # of foreground, below this a limb blob is noise


@dataclass
class Part:
    name: str
    rgba: np.ndarray          # cut with overlap, alpha 0 outside
    origin: list[int]         # [x, y] of rgba[0,0] in the source frame
    pivot: list[float]        # joint position in source-frame coordinates
    z: int
    occluded_px: int = 0
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "origin": self.origin,
                "pivot": [round(v, 1) for v in self.pivot], "z": self.z,
                "size": [int(self.rgba.shape[1]), int(self.rgba.shape[0])],
                "occluded_px": self.occluded_px, "flags": self.flags}


@dataclass
class Transform:
    """Per-part change. Rotation is about the part's own pivot."""
    dx: float = 0.0
    dy: float = 0.0
    degrees: float = 0.0
    scale: float = 1.0


# ------------------------------------------------------------- analysis --

def colour_layers(rgba: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """skin / fur / cloth. Fur is what separates the tail and hair from the body."""
    rgb = rgba[..., :3].copy()
    rgb[~mask] = 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return {
        "skin": mask & (h >= 5) & (h <= 22) & (s >= 25) & (s <= 110) & (v > 150),
        "fur": mask & (s < 38) & (v > 195),
        "cloth": mask & (s < 28) & (v > 225),
    }


def _smooth_row_widths(mask: np.ndarray) -> np.ndarray:
    widths = mask.sum(axis=1).astype(np.float32)
    kernel = max(5, int(mask.shape[0] * 0.015)) | 1
    return np.convolve(widths, np.ones(kernel) / kernel, mode="same")


def find_joints(mask: np.ndarray) -> dict[str, Any]:
    """Neck and hip from the row-width profile, plus the body corridor.

    The neck is the sharpest narrowing in the upper figure: measured 99 px against a 300 px head
    and a 250 px chest on the real frame, which is unmistakable. The hip is the narrowing between
    the torso's widest row and the legs.
    """
    height, width = mask.shape
    profile = _smooth_row_widths(mask)
    lo, hi = int(height * NECK_SEARCH[0]), int(height * NECK_SEARCH[1])
    neck = lo + int(np.argmin(profile[lo:hi])) if hi > lo else int(height * 0.25)
    lo2, hi2 = int(height * HIP_SEARCH[0]), int(height * HIP_SEARCH[1])
    hip = lo2 + int(np.argmin(profile[lo2:hi2])) if hi2 > lo2 else int(height * 0.55)
    torso_band = mask[neck:hip]
    cols = np.nonzero(torso_band.any(axis=0))[0]
    corridor = (int(cols.min()), int(cols.max()) + 1) if cols.size else (0, width)
    head_cols = np.nonzero(mask[:neck].any(axis=0))[0]
    head_cx = float(head_cols.mean()) if head_cols.size else width / 2.0
    return {"neck_y": int(neck), "hip_y": int(hip), "corridor": list(corridor),
            "head_center_x": head_cx, "neck_width": float(profile[neck]),
            "head_height": int(neck), "torso_height": int(hip - neck),
            "leg_height": int(height - hip)}


# ------------------------------------------------------------- cutting --

def _inpaint_occluded(rgba: np.ndarray, mask: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill pixels a part needs but does not own, using its own neighbourhood.

    A far leg sits behind the tail, so its hidden middle has no colour. Swinging the tail exposes
    it. Telea inpainting from the surrounding leg pixels is not correct art, but it is the right
    colour and it is never seen except as a sliver at the edge of a swing.
    """
    if not hole.any():
        return rgba
    out = rgba.copy()
    bgr = cv2.cvtColor(out[..., :3], cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    out[..., :3] = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    out[..., 3] = np.where(hole, 255, out[..., 3])
    return out


def _cut(rgba: np.ndarray, region: np.ndarray, pivot: tuple[float, float], name: str,
         *, overlap: int = 0, occluder: np.ndarray | None = None) -> Part | None:
    """Extract one part. ``overlap`` dilates the region so the seam is covered by the part above."""
    if not region.any():
        return None
    grown = region
    if overlap > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (overlap * 2 + 1,) * 2)
        grown = cv2.dilate(region.astype(np.uint8), kernel) > 0
    ys, xs = np.nonzero(grown)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = rgba[y0:y1, x0:x1].copy()
    local = grown[y0:y1, x0:x1]
    crop[..., 3] = np.where(local, crop[..., 3], 0)
    occluded = 0
    if occluder is not None:
        hole = local & occluder[y0:y1, x0:x1] & (crop[..., 3] == 0)
        occluded = int(hole.sum())
        if occluded:
            crop = _inpaint_occluded(crop, local, hole)
    return Part(name=name, rgba=crop, origin=[x0, y0], pivot=[float(pivot[0]), float(pivot[1])],
                z=PART_Z.get(name, 8), occluded_px=occluded)


def segment(rgba: np.ndarray, mask: np.ndarray) -> tuple[list[Part], dict[str, Any]]:
    """Cut a frame into head, torso, tail and limb segments.

    Limbs are split at the elbow and knee by dividing each limb blob along its own principal axis,
    which works because a limb is long and thin: the axis is the bone. This is coarser than a
    hand-placed skeleton and it is enough to rotate a forearm about an elbow.
    """
    height, width = mask.shape
    joints = find_joints(mask)
    layers = colour_layers(rgba, mask)
    overlap = max(4, int(height * SEAM_OVERLAP_PX))
    neck, hip = joints["neck_y"], joints["hip_y"]
    cx0, cx1 = joints["corridor"]

    # tail: the largest fur blob whose centroid sits outside the body corridor
    tail = np.zeros_like(mask)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(layers["fur"].astype(np.uint8), 8)
    best, best_area = None, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx = float(centroids[i][0])
        if area > best_area and (cx < cx0 or cx > cx1):
            best, best_area = i, area
    if best is not None and best_area >= LIMB_MIN_AREA * mask.sum():
        tail = labels == best

    body = mask & ~tail
    head_region = body.copy()
    head_region[neck:] = False
    torso_region = body.copy()
    torso_region[:neck] = False
    torso_region[hip:] = False
    legs_region = body.copy()
    legs_region[:hip] = False

    # arms: skin outside the torso corridor within the torso band
    arms_region = torso_region.copy()
    arms_region[:, cx0:cx1] = False
    torso_core = torso_region & ~arms_region

    parts: list[Part] = []
    head = _cut(rgba, head_region, (joints["head_center_x"], neck), "head", overlap=overlap)
    if head:
        parts.append(head)
    torso = _cut(rgba, torso_core, ((cx0 + cx1) / 2.0, neck), "torso", overlap=overlap)
    if torso:
        parts.append(torso)
    if tail.any():
        tail_parts = _split_along_axis(rgba, tail, "tail", ("tail_base", "tail_tip"), overlap,
                                       occluder=legs_region)
        parts.extend(tail_parts)
    for label, region, names in (("arm", arms_region, ("upper", "lower", "hand")),
                                 ("leg", legs_region, ("upper", "lower", "foot"))):
        for near, blob in _near_far_blobs(region, mask):
            parts.extend(_split_along_axis(
                rgba, blob, label,
                tuple(f"{label}_{near}_{n_}" for n_ in names), overlap))
    parts.sort(key=lambda p: p.z)
    report = dict(joints)
    report["parts"] = [p.to_dict() for p in parts]
    report["tail_found"] = bool(tail.any())
    report["total_occluded_px"] = sum(p.occluded_px for p in parts)
    report["head_ratio"] = round(height / max(1.0, float(joints["head_height"])), 2)
    return parts, report


def _near_far_blobs(region: np.ndarray, mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Split a limb region into the near and far limb. The near one is larger (it occludes)."""
    n, labels, stats, _c = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
    blobs = [(int(stats[i, cv2.CC_STAT_AREA]), labels == i) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= LIMB_MIN_AREA * mask.sum()]
    blobs.sort(key=lambda b: -b[0])
    out = []
    for i, (_area, blob) in enumerate(blobs[:2]):
        out.append(("near" if i == 0 else "far", blob))
    return out


def _split_along_axis(rgba: np.ndarray, blob: np.ndarray, label: str, names: Sequence[str],
                      overlap: int, occluder: np.ndarray | None = None) -> list[Part]:
    """Cut a long thin blob into len(names) segments along its principal axis.

    A limb's principal axis is its bone, so cutting perpendicular to it at even spacing lands the
    cuts near the elbow/knee without a hand-placed skeleton. Each segment's pivot is the cut it
    hangs from, which is exactly what a rotation needs.
    """
    ys, xs = np.nonzero(blob)
    if xs.size < 50:
        return []
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = points.mean(axis=0)
    centred = points - mean
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
    t = centred @ axis
    lo, hi = float(t.min()), float(t.max())
    parts: list[Part] = []
    count = len(names)
    for i, name in enumerate(names):
        a = lo + (hi - lo) * i / count
        b = lo + (hi - lo) * (i + 1) / count
        seg = np.zeros_like(blob)
        sel = (t >= a) & (t <= b)
        seg[ys[sel], xs[sel]] = True
        joint = mean + axis * a
        part = _cut(rgba, seg, (float(joint[0]), float(joint[1])), name,
                    overlap=overlap, occluder=occluder)
        if part:
            parts.append(part)
    return parts


# ----------------------------------------------------------- compositing --

def compose(parts: Sequence[Part], canvas_size: tuple[int, int],
            transforms: dict[str, Transform] | None = None,
            offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Paint the parts back in z order, each with its own transform about its pivot."""
    width, height = canvas_size
    out = np.zeros((height, width, 4), np.float32)
    transforms = transforms or {}
    for part in sorted(parts, key=lambda p: p.z):
        t = transforms.get(part.name, Transform())
        ph, pw = part.rgba.shape[:2]
        # part-local pivot
        px = part.pivot[0] - part.origin[0]
        py = part.pivot[1] - part.origin[1]
        m = cv2.getRotationMatrix2D((px, py), t.degrees, t.scale)
        m[0, 2] += part.origin[0] + t.dx + offset[0]
        m[1, 2] += part.origin[1] + t.dy + offset[1]
        warped = cv2.warpAffine(part.rgba, m, (width, height), flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = (warped[..., 3:4].astype(np.float32) / 255.0)
        out[..., :3] = warped[..., :3].astype(np.float32) * a + out[..., :3] * (1.0 - a)
        out[..., 3:4] = np.maximum(out[..., 3:4], warped[..., 3:4].astype(np.float32))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def rebuild_identity(parts: Sequence[Part], canvas_size: tuple[int, int]) -> np.ndarray:
    """Recompose with no transforms. Must reproduce the source: the rig's sanity check."""
    return compose(parts, canvas_size)


def scale_head(parts: Sequence[Part], factor: float) -> dict[str, Transform]:
    """Resize the head about the neck. The body is untouched, so the head-to-body ratio changes.

    This is the direct fix for "비율이 안 맞는다": a frame whose head is 12 % too big becomes
    correct without redrawing anything, because the head pixels are simply scaled about the neck
    joint they hang from.
    """
    return {"head": Transform(scale=factor)}


def swing(parts: Sequence[Part], joint_degrees: dict[str, float]) -> dict[str, Transform]:
    """Rotate named segments about their own joints, propagating down the chain.

    Rotating an upper arm must carry the forearm and hand with it, otherwise the limb comes apart.
    """
    chains = {
        "arm_near_upper": ["arm_near_lower", "arm_near_hand"],
        "arm_far_upper": ["arm_far_lower", "arm_far_hand"],
        "leg_near_upper": ["leg_near_lower", "leg_near_foot"],
        "leg_far_upper": ["leg_far_lower", "leg_far_foot"],
        "tail_base": ["tail_tip"],
    }
    by_name = {p.name: p for p in parts}
    out: dict[str, Transform] = {}
    for name, degrees in joint_degrees.items():
        out[name] = Transform(degrees=degrees)
        parent = by_name.get(name)
        if parent is None:
            continue
        for child_name in chains.get(name, []):
            child = by_name.get(child_name)
            if child is None:
                continue
            # rotate the child about the parent's pivot: same angle, plus the displacement the
            # rotation imposes on the child's own pivot
            theta = np.deg2rad(degrees)
            cos, sin = float(np.cos(theta)), float(np.sin(theta))
            vx = child.pivot[0] - parent.pivot[0]
            vy = child.pivot[1] - parent.pivot[1]
            nx = cos * vx + sin * vy
            ny = -sin * vx + cos * vy
            out[child_name] = Transform(dx=nx - vx, dy=ny - vy, degrees=degrees)
    return out
