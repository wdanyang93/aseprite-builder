"""Lineart segmentation. Let the drawing's own lines decide the boundaries.

Everything before this assigned pixels by distance to a centreline, optionally nudged by shading
or texture, and every version of that leaked across part boundaries: the tail covered the thigh,
the arm swallowed the gap to the torso, the far leg merged with the tail. The reason is that
distance has no idea where a part ends.

The drawing does. This is line art: parts are separated by dark strokes, both the outer silhouette
and interior lines. Measured on one real frame, the foreground's L* runs 47 at the 1st percentile
to 236 at the 95th, and a horizontal scan across the tail-and-leg boundary reads
``... 255 255 0 0 255 255 ...`` - the boundary is literally two black pixels wide. 8.7 % of the
figure is darker than L* 120.

So the pipeline becomes:

    lineart mask  ->  interior cells (connected regions between lines)
                  ->  each cell assigned to the part whose centreline crosses it
                  ->  lines themselves given to the part on their darker side

A cell cannot be split across parts, so a thigh cannot end up half tail. Where a cell has no
centreline through it, it is reported unassigned rather than silently absorbed.
"""
from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

def resize_rgba(rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize with premultiplied alpha, return straight alpha.

    Resampling straight alpha mixes transparent pixels' colour into the edge, which is where the
    dark or white fringe on a scaled sprite comes from. Premultiplying first, resampling, then
    un-premultiplying keeps the outline the colour it was drawn.
    """
    width, height = size
    src = rgba.astype(np.float32)
    alpha = src[..., 3:4] / 255.0
    premultiplied = np.concatenate([src[..., :3] * alpha, src[..., 3:4]], axis=2)
    interpolation = cv2.INTER_AREA if (width * height) < (rgba.shape[1] * rgba.shape[0]) else cv2.INTER_LANCZOS4
    scaled = cv2.resize(premultiplied, (width, height), interpolation=interpolation)
    out_alpha = np.clip(scaled[..., 3:4], 0.0, 255.0)
    safe = np.maximum(out_alpha / 255.0, 1e-6)
    rgb = np.clip(scaled[..., :3] / safe, 0.0, 255.0)
    return np.dstack([rgb, out_alpha]).astype(np.uint8)


LINE_L_MAX = 150.0             # L* at or below this is drawn line, not shading
LINE_DILATE = 1
MIN_CELL_PX = 60               # smaller islands are line noise, merged into their neighbour
# Lowering this to 4 px let a small cell inherit whichever region happened to brush it, and two
# frames lost their tail to the torso. A cell must share a real fraction of its own perimeter.
BORDER_MIN_PX = 12
BORDER_MIN_SHARE = 0.20        # of the cell's own perimeter
ORPHAN_NAME_MIN_PX = 800       # below this an orphan cell is noise, not a part
CELL_PER_CENTERLINE_PX = 260   # px of cell a single centreline pixel may plausibly own
UNASSIGNED = "unassigned"


def lineart_mask(rgba: np.ndarray, mask: np.ndarray, l_max: float = LINE_L_MAX) -> np.ndarray:
    """The drawn strokes: dark pixels inside the figure, plus the silhouette edge itself.

    A plain threshold on L* is enough because the art keeps its lines far darker than its shading:
    the shaded far leg sits at L* 203 while the lines sit below 150.
    """
    lightness = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    lines = mask & (lightness <= l_max)
    # the silhouette boundary is a line too, even where the outside is transparent
    border = mask & ~(cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    lines |= border
    if LINE_DILATE:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LINE_DILATE * 2 + 1,) * 2)
        lines = (cv2.dilate(lines.astype(np.uint8), kernel) > 0) & mask
    return lines


def interior_cells(mask: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the regions the lines enclose. These are the atoms of the segmentation."""
    interior = mask & ~lines
    count, labels = cv2.connectedComponents(interior.astype(np.uint8), connectivity=4)
    # absorb specks into the largest neighbour so they do not become their own parts
    sizes = np.bincount(labels.ravel(), minlength=count)
    small = [i for i in range(1, count) if sizes[i] < MIN_CELL_PX]
    if small:
        keep = np.isin(labels, small, invert=True) & (labels > 0)
        if keep.any():
            distance, nearest = cv2.distanceTransformWithLabels(
                (~keep).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            # map each absorbed pixel to the label of its nearest kept pixel
            ys, xs = np.nonzero(keep)
            lookup = np.zeros(nearest.max() + 1, np.int32)
            lookup[nearest[ys, xs]] = labels[ys, xs]
            for i in small:
                sel = labels == i
                labels[sel] = lookup[nearest[sel]]
    return labels, count


def assign_cells(labels: np.ndarray, chains: Sequence[Any], mask: np.ndarray,
                 lines: np.ndarray, rgba: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Give every cell to the part whose centreline spends the most length inside it.

    A cell is indivisible, which is the whole point: the thigh cell goes to a leg or to the tail,
    never half to each.
    """
    votes: dict[int, dict[str, float]] = {}
    for chain in chains:
        bounds = [0] + list(chain.joints) + [len(chain.points)]
        names = _segment_names_for(chain, len(bounds) - 1)
        for i, name in enumerate(names):
            for x, y in chain.points[bounds[i]:bounds[i + 1]]:
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= yi < labels.shape[0] and 0 <= xi < labels.shape[1]):
                    continue
                cell = int(labels[yi, xi])
                if cell <= 0:
                    # the centreline runs along a line pixel; look just around it
                    y0, y1 = max(0, yi - 2), min(labels.shape[0], yi + 3)
                    x0, x1 = max(0, xi - 2), min(labels.shape[1], xi + 3)
                    window = labels[y0:y1, x0:x1]
                    window = window[window > 0]
                    if window.size == 0:
                        continue
                    cell = int(np.bincount(window).argmax())
                votes.setdefault(cell, {}).setdefault(name, 0.0)
                votes[cell][name] += 1.0

    regions: dict[str, np.ndarray] = {}
    unassigned = np.zeros_like(mask)
    cell_report: list[dict[str, Any]] = []
    pending: list[int] = []
    for cell in range(1, int(labels.max()) + 1):
        sel = labels == cell
        if not sel.any():
            continue
        tally = votes.get(cell)
        if not tally:
            pending.append(cell)
            continue
        owner = max(tally, key=tally.get)
        # A cell takes the name of whichever centreline crosses it most, but a limb centreline
        # clipping the corner of the body cell must not rename the whole body. Measured: an
        # "arm_lower" claimed 27.7 % of the figure and an "ear" 19.4 %, i.e. the torso cell. If a
        # cell is far larger than the claiming part's own centreline can account for, it is held
        # back as torso instead.
        crossing = float(tally[owner])
        expected = crossing * CELL_PER_CENTERLINE_PX
        if int(sel.sum()) > expected and int(sel.sum()) >= ORPHAN_NAME_MIN_PX:
            pending.append(cell)
            cell_report.append({"cell": cell, "pixels": int(sel.sum()), "owner": "held_back",
                                "claimed_by": owner, "centerline_px": int(crossing)})
            continue
        regions[owner] = regions.get(owner, np.zeros_like(mask)) | sel
        cell_report.append({"cell": cell, "pixels": int(sel.sum()), "owner": owner,
                            "votes": {k: int(v) for k, v in tally.items()}})

    # Cells no centreline passes through - most often the clothing, which is its own cell walled
    # off by lines - go to the neighbour they share the longest border with. Iterated, so a cell
    # adjacent only to other pending cells resolves once its neighbour does.
    for _round in range(8):
        if not pending:
            break
        still: list[int] = []
        for cell in pending:
            sel = labels == cell
            ring = (cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~sel & mask
            perimeter = max(1, int(ring.sum()))
            best, best_len = None, 0
            for name, region in regions.items():
                if name == UNASSIGNED:
                    continue
                shared = int((ring & region).sum())
                if shared > best_len:
                    best, best_len = name, shared
            if (best is not None and best_len >= BORDER_MIN_PX
                    and best_len / perimeter >= BORDER_MIN_SHARE):
                regions[best] = regions[best] | sel
                cell_report.append({"cell": cell, "pixels": int(sel.sum()), "owner": best,
                                    "via": "shared_border", "border_px": best_len})
            else:
                still.append(cell)
        if len(still) == len(pending):
            break
        pending = still
    # Anything still without an owner is named from its own shape rather than abandoned.
    for cell in pending:
        sel = labels == cell
        if int(sel.sum()) >= ORPHAN_NAME_MIN_PX and rgba is not None:
            name = name_orphan_cell(sel, mask, rgba)
            if name != UNASSIGNED:
                regions[name] = regions.get(name, np.zeros_like(mask)) | sel
                cell_report.append({"cell": cell, "pixels": int(sel.sum()), "owner": name,
                                    "via": "orphan_shape"})
                continue
        unassigned |= sel
        cell_report.append({"cell": cell, "pixels": int(sel.sum()), "owner": UNASSIGNED})

    # Line pixels belong to the region they border. Only pixels that are *still unowned* may be
    # taken: the previous version dilated every region and let the winner claim the overlap, which
    # overwrote already-assigned parts - on one frame the torso disappeared entirely and half the
    # figure came back magenta.
    if lines.any() and regions:
        owned = np.logical_or.reduce(list(regions.values()))
        free = lines & ~owned & mask
        if free.any():
            names = [n for n in regions if n != UNASSIGNED]
            if names:
                grown = np.stack([cv2.dilate(regions[n].astype(np.uint8),
                                             np.ones((5, 5), np.uint8)).astype(np.int16)
                                  for n in names])
                winner = np.argmax(grown, axis=0)
                touched = grown.max(axis=0) > 0
                for i, name in enumerate(names):
                    regions[name] = regions[name] | (free & touched & (winner == i))

    report = {"cells": int(labels.max()), "unassigned_px": int(unassigned.sum()),
              "cell_detail": sorted(cell_report, key=lambda c: -c["pixels"])[:12]}
    if unassigned.any():
        regions[UNASSIGNED] = unassigned
    return regions, report


def name_orphan_cell(cell_mask: np.ndarray, figure: np.ndarray, rgba: np.ndarray) -> str:
    """Name a cell that no centreline reached, from its own shape.

    Cells are reliable; centrelines are not. Measured across the eight frames, every frame where
    100 % of the large cells had a centreline through them segmented perfectly, and only the frames
    at 40-75 % coverage produced unassigned pixels - two of them because the tail never became a
    branch at all. So instead of tuning the border thresholds (which merely traded one frame's tail
    for another's), an orphan cell is described directly: its own medial axis gives a length, a
    thickness and a taper, and its bounding box gives a position.
    """
    from .segment import (ARM_MAX_Y, ARM_MIN_Y, EAR_MAX_Y, EDGE_TOUCH, FOOT_MIN_Y, HEAD_MAX_Y,
                          TAIL_MIN_TAPER, TAIL_MIN_Y, _skeletonise)

    height, width = figure.shape
    ys, xs = np.nonzero(cell_mask)
    if xs.size == 0:
        return UNASSIGNED
    # position of the cell's extreme point, measured like a branch tip
    top_y = float(ys.min()) / height
    bottom_y = float(ys.max()) / height
    left_x = float(xs.min()) / width
    right_x = float(xs.max()) / width
    at_edge = left_x <= EDGE_TOUCH or right_x >= 1.0 - EDGE_TOUCH

    skeleton, distance = _skeletonise(cell_mask)
    widths = distance[skeleton] if skeleton.any() else np.array([1.0])
    mean_width = float(np.mean(widths))
    # taper: thickness at the extreme end versus the middle
    end_band = cell_mask.copy()
    if at_edge and left_x <= EDGE_TOUCH:
        end_band[:, int(width * (left_x + 0.08)):] = False
    elif at_edge:
        end_band[:, :int(width * (right_x - 0.08))] = False
    else:
        end_band[: int(height * (bottom_y - 0.08)), :] = False
    end_widths = distance[skeleton & end_band]
    taper = float(np.mean(end_widths) / max(1e-6, mean_width)) if end_widths.size else 0.5

    if bottom_y >= FOOT_MIN_Y:
        return "leg"
    if at_edge and bottom_y >= TAIL_MIN_Y and taper >= TAIL_MIN_TAPER:
        return "tail"
    if top_y <= HEAD_MAX_Y:
        return "head"
    if top_y <= EAR_MAX_Y:
        return "ear"
    if ARM_MIN_Y <= bottom_y <= ARM_MAX_Y and not at_edge:
        return "arm"
    return "torso"


def _segment_names_for(chain: Any, count: int) -> list[str]:
    from .segment import _segment_names
    return _segment_names(chain.name, count)


def segment_by_lineart(rgba: np.ndarray, mask: np.ndarray) -> tuple[dict[str, np.ndarray], list[Any], dict[str, Any]]:
    """Full lineart segmentation, replacing distance-based ownership."""
    from .segment import classify_depth, extract_chains, torso_region

    chains = extract_chains(mask)
    lines = lineart_mask(rgba, mask)
    labels, _count = interior_cells(mask, lines)
    regions, cell_report = assign_cells(labels, chains, mask, lines, rgba)
    depth = classify_depth(rgba, {k: v for k, v in regions.items() if k != UNASSIGNED})
    torso = torso_region(mask, regions)
    if torso.any():
        regions["torso"] = torso
    report = {
        "lineart_px": int(lines.sum()),
        "lineart_share": round(float(lines.sum() / max(1, mask.sum())), 4),
        "depth": depth,
        "chains": [c.to_dict() for c in chains],
        **cell_report,
        "coverage_px": int(np.logical_or.reduce(list(regions.values())).sum()) if regions else 0,
        "figure_px": int(mask.sum()),
    }
    return regions, chains, report
