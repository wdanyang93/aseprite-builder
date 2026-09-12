"""Ribbon deformation. Move existing pixels onto a new centerline.

This is the part that actually edits the drawing. Everything before it only decided which pixels
belong to which limb; this takes a limb and puts it somewhere else, by resampling the original
pixels rather than drawing anything.

A limb is described as a ribbon:

    s in [0, 1]   position along the centerline, 0 at the root (shoulder or hip)
    u in [-1, 1]  signed distance across it, normalised by the width on that side

Normalising ``u`` by the per-side width is what separates length from thickness: moving the
centerline changes the pose, scaling the widths changes the thickness, and neither disturbs the
other. Two-sided widths matter because a limb is not symmetric - measured on a real leg, the two
sides differ by 2.5 to 40 %.

Warping runs **backward**: for each destination pixel, find its (s, u), look up where that (s, u)
sat in the source, and sample there. Forward warping leaves holes; backward warping cannot.

Interpolating between two frames' ribbons with the root pinned gives exactly the behaviour the
owner described - "the top changes little, the fingertips change a lot" - because the root is
shared and the deviation grows along s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

SAMPLES = 64                  # centerline resampling resolution
RAY_LIMIT = 400               # px to search outward for the silhouette edge
MIN_WIDTH = 1.0
FOLD_EPS = 1e-3


@dataclass
class Ribbon:
    """A limb as a centerline plus per-side widths, in frame pixels."""
    name: str
    points: np.ndarray        # (SAMPLES, 2) x, y - root first
    width_left: np.ndarray    # (SAMPLES,)
    width_right: np.ndarray   # (SAMPLES,)
    root: np.ndarray          # (2,)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def tangents(self) -> np.ndarray:
        d = np.gradient(self.points, axis=0)
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.maximum(norm, 1e-6)

    def normals(self) -> np.ndarray:
        t = self.tangents()
        return np.stack([-t[:, 1], t[:, 0]], axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "length": round(self.length, 1),
                "root": [round(float(v), 1) for v in self.root],
                "tip": [round(float(v), 1) for v in self.points[-1]],
                "mean_width": [round(float(self.width_left.mean()), 1),
                               round(float(self.width_right.mean()), 1)]}


# ------------------------------------------------------------ extract --

def _resample(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < 2:
        return np.repeat(points, count, axis=0)[:count]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] <= 0:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, arc[-1], count)
    return np.stack([np.interp(targets, arc, points[:, 0]),
                     np.interp(targets, arc, points[:, 1])], axis=1)


def _ray_width(mask: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """Distance from a centerline point to the silhouette edge along one normal direction."""
    height, width = mask.shape
    for step in range(1, RAY_LIMIT):
        p = origin + direction * step
        x, y = int(round(p[0])), int(round(p[1]))
        if not (0 <= y < height and 0 <= x < width) or not mask[y, x]:
            return float(step - 1)
    return float(RAY_LIMIT)


def extract_ribbon(mask: np.ndarray, name: str = "part",
                   root_hint: Sequence[float] | None = None,
                   samples: int = SAMPLES) -> Ribbon | None:
    """Centerline plus two-sided widths for one part mask.

    ``root_hint`` picks which end is the root - for an arm that is the shoulder, which is the end
    nearest the torso. Without it the higher end is taken, since limbs hang downward.
    """
    from .form import _neighbour_count, _prune, _walk_from
    from .segment import _skeletonise

    if int(mask.sum()) < 200:
        return None
    skeleton, _distance = _skeletonise(mask)
    skeleton = _prune(skeleton, max(3, int(mask.shape[0] * 0.02)))
    if not skeleton.any():
        return None
    counts = _neighbour_count(skeleton)
    ends = list(zip(*np.nonzero(skeleton & (counts == 1))))
    if not ends:
        # A closed loop: the arm ends in a fist, so the medial axis circles the hand and has no
        # free end at all. Measured on frame 1: 108 skeleton pixels, 0 endpoints, and the walk
        # collapsed to a 2 px "ribbon". Cutting the loop at the point farthest from the root hint
        # turns it back into a chain.
        ys, xs = np.nonzero(skeleton)
        if xs.size == 0:
            return None
        if root_hint is not None:
            hint = np.asarray(root_hint, np.float32)
            far = int(np.argmax((xs - hint[0]) ** 2 + (ys - hint[1]) ** 2))
        else:
            far = int(np.argmax(ys))
        cut = skeleton.copy()
        cy, cx = int(ys[far]), int(xs[far])
        cut[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = False
        skeleton = cut
        counts = _neighbour_count(skeleton)
        ends = list(zip(*np.nonzero(skeleton & (counts == 1))))
        if not ends:
            ys, xs = np.nonzero(skeleton)
            if xs.size == 0:
                return None
            ends = [(int(ys[0]), int(xs[0]))]
    best: list[tuple[int, int]] = []
    for endpoint in ends:
        walk = _walk_from(skeleton, endpoint, stop_at_junction=False)
        if len(walk) > len(best):
            best = walk
    if len(best) < 2:
        return None
    points = np.array([[float(x), float(y)] for y, x in best], np.float32)
    if root_hint is not None:
        hint = np.asarray(root_hint, np.float32)
        if (np.linalg.norm(points[0] - hint) > np.linalg.norm(points[-1] - hint)):
            points = points[::-1]
    elif points[0][1] > points[-1][1]:
        points = points[::-1]

    resampled = _resample(points, samples).astype(np.float32)
    ribbon = Ribbon(name=name, points=resampled,
                    width_left=np.zeros(samples, np.float32),
                    width_right=np.zeros(samples, np.float32),
                    root=resampled[0].copy())
    normals = ribbon.normals()
    for i in range(samples):
        ribbon.width_left[i] = max(MIN_WIDTH, _ray_width(mask, resampled[i], normals[i]))
        ribbon.width_right[i] = max(MIN_WIDTH, _ray_width(mask, resampled[i], -normals[i]))
    return ribbon


# ---------------------------------------------------------- interpolate --

def interpolate(a: Ribbon, b: Ribbon, t: float, *, pin_root: bool = True,
                name: str | None = None) -> Ribbon:
    """Blend two ribbons. With ``pin_root`` the root stays at A's, so deviation grows along s.

    This produces the motion the owner asked for directly: at the shoulder the two poses already
    agree, and the difference accumulates toward the hand.
    """
    points = (1.0 - t) * a.points + t * b.points
    if pin_root:
        points = points + (a.root - points[0])
    return Ribbon(name=name or f"{a.name}@{t:.2f}", points=points.astype(np.float32),
                  width_left=((1.0 - t) * a.width_left + t * b.width_left).astype(np.float32),
                  width_right=((1.0 - t) * a.width_right + t * b.width_right).astype(np.float32),
                  root=points[0].astype(np.float32).copy())


def fold_detected(ribbon: Ribbon) -> tuple[bool, float]:
    """Does the ribbon self-intersect? Normals crossing means the warp folds over itself.

    Reported rather than silently accepted: a folded map produces a limb turned inside out, and
    the fix is more control points, not a smoothing pass.
    """
    normals = ribbon.normals()
    tangents = ribbon.tangents()
    # a fold shows up as the tangent reversing direction along the centerline
    dots = np.sum(tangents[1:] * tangents[:-1], axis=1)
    worst = float(dots.min()) if dots.size else 1.0
    # and as the offset curve crossing itself: left edge points must stay ordered along s
    left = ribbon.points + normals * ribbon.width_left[:, None]
    progress = np.sum(np.diff(left, axis=0) * tangents[:-1], axis=1)
    crossing = float(progress.min()) if progress.size else 1.0
    return (worst < 0.0 or crossing < -FOLD_EPS), min(worst, crossing)


# --------------------------------------------------------------- warp --

def warp(rgba: np.ndarray, mask: np.ndarray, source: Ribbon, target: Ribbon,
         canvas_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Move the part's pixels from the source ribbon onto the target ribbon.

    Backward mapping: every destination pixel asks which source pixel belongs at its (s, u).
    Sampling is done on premultiplied alpha so the edge does not pick up transparent black.
    """
    height, width = canvas_shape or mask.shape
    samples = len(target.points)
    t_normals = target.normals()
    s_normals = source.normals()

    # Build a dense (s, u) -> source-pixel table by walking the target ribbon and, for each
    # cross-section, mapping the same normalised offsets back to the source cross-section.
    offsets = np.linspace(-1.0, 1.0, RIBBON_CROSS_SAMPLES, dtype=np.float32)
    dst_x = np.empty((samples, RIBBON_CROSS_SAMPLES), np.float32)
    dst_y = np.empty_like(dst_x)
    src_x = np.empty_like(dst_x)
    src_y = np.empty_like(dst_x)
    for i in range(samples):
        t_w = np.where(offsets >= 0, target.width_left[i], target.width_right[i])
        s_w = np.where(offsets >= 0, source.width_left[i], source.width_right[i])
        dst_x[i] = target.points[i, 0] + t_normals[i, 0] * offsets * t_w
        dst_y[i] = target.points[i, 1] + t_normals[i, 1] * offsets * t_w
        src_x[i] = source.points[i, 0] + s_normals[i, 0] * offsets * s_w
        src_y[i] = source.points[i, 1] + s_normals[i, 1] * offsets * s_w

    # Rasterise the destination quad grid, carrying the source coordinates with it.
    map_x = np.full((height, width), -1.0, np.float32)
    map_y = np.full((height, width), -1.0, np.float32)
    for i in range(samples - 1):
        for j in range(RIBBON_CROSS_SAMPLES - 1):
            quad_dst = np.array([[dst_x[i, j], dst_y[i, j]], [dst_x[i, j + 1], dst_y[i, j + 1]],
                                 [dst_x[i + 1, j + 1], dst_y[i + 1, j + 1]],
                                 [dst_x[i + 1, j], dst_y[i + 1, j]]], np.float32)
            quad_src = np.array([[src_x[i, j], src_y[i, j]], [src_x[i, j + 1], src_y[i, j + 1]],
                                 [src_x[i + 1, j + 1], src_y[i + 1, j + 1]],
                                 [src_x[i + 1, j], src_y[i + 1, j]]], np.float32)
            transform = cv2.getPerspectiveTransform(quad_dst, quad_src)
            x0 = max(0, int(np.floor(quad_dst[:, 0].min())))
            x1 = min(width, int(np.ceil(quad_dst[:, 0].max())) + 1)
            y0 = max(0, int(np.floor(quad_dst[:, 1].min())))
            y1 = min(height, int(np.ceil(quad_dst[:, 1].max())) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            cell = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(cell, (quad_dst - [x0, y0]).astype(np.int32), 1)
            if not cell.any():
                continue
            ys, xs = np.nonzero(cell)
            pts = np.stack([xs + x0, ys + y0, np.ones(len(xs))], axis=0).astype(np.float32)
            mapped = transform @ pts
            mapped = mapped[:2] / np.maximum(mapped[2:], 1e-6)
            map_x[ys + y0, xs + x0] = mapped[0]
            map_y[ys + y0, xs + x0] = mapped[1]

    src = rgba.astype(np.float32)
    alpha = (src[..., 3:4] / 255.0) * mask[..., None]
    premultiplied = np.concatenate([src[..., :3] * alpha, alpha * 255.0], axis=2)
    valid = map_x >= 0
    sampled = cv2.remap(premultiplied, np.where(valid, map_x, 0).astype(np.float32),
                        np.where(valid, map_y, 0).astype(np.float32),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0))
    sampled[~valid] = 0
    out_alpha = np.clip(sampled[..., 3:4], 0, 255)
    safe = np.maximum(out_alpha / 255.0, 1e-6)
    out = np.dstack([np.clip(sampled[..., :3] / safe, 0, 255), out_alpha]).astype(np.uint8)
    folded, margin = fold_detected(target)
    return out, {"covered_px": int(valid.sum()), "folded": folded,
                 "fold_margin": round(margin, 4),
                 "source_px": int((mask & (rgba[..., 3] > 8)).sum()),
                 "result_px": int((out[..., 3] > 8).sum())}


RIBBON_CROSS_SAMPLES = 21
