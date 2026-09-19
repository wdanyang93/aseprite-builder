"""Deterministic post-processing of a generated frame into the request spec.

The generator keeps emitting 1374x1145-ish canvases with anti-aliased alpha and
a red fringe, and three regeneration attempts produced the same failures, so the
fix belongs here rather than in another prompt.

Step order is load-bearing: resampling has to happen *before* alpha is
binarised, otherwise LANCZOS puts the partial alpha straight back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .ccl import component_areas
from .measure import load_rgba, subject_bbox
from .spec import Spec

# Clockwise from north; fixed order keeps the red fill reproducible.
_NEIGHBOURS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


@dataclass(frozen=True)
class NormalizeResult:
    scale: float
    source_size: tuple[int, int]
    placed_size: tuple[int, int]
    binarised_px: int
    red_repaired_px: int
    red_dropped_px: int
    specks_removed: int
    notes: tuple[str, ...]


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift `arr` so that position (y, x) holds the old (y - dy, x - dx)."""
    out = np.zeros_like(arr)
    h, w = arr.shape[:2]
    out[
        max(0, dy) : h - max(0, -dy),
        max(0, dx) : w - max(0, -dx),
    ] = arr[
        max(0, -dy) : h - max(0, dy),
        max(0, -dx) : w - max(0, dx),
    ]
    return out


def _repair_pure_red(
    rgb: np.ndarray, opaque: np.ndarray, max_passes: int = 128
) -> tuple[np.ndarray, int, np.ndarray]:
    """Recolour pure-red pixels from their nearest non-red opaque neighbour.

    Returns the repaired RGB, how many pixels were recoloured, and the mask of
    red pixels that stayed unreachable (callers drop those).
    """
    rgb = rgb.copy()
    red = opaque & (rgb[:, :, 0] == 255) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 0)
    valid = opaque & ~red
    repaired = 0

    for _ in range(max_passes):
        if not red.any():
            break
        progressed = False
        for dy, dx in _NEIGHBOURS:
            if not red.any():
                break
            donor_rgb = _shift(rgb, -dy, -dx)
            donor_ok = _shift(valid, -dy, -dx)
            take = red & donor_ok
            count = int(np.count_nonzero(take))
            if count:
                rgb[take] = donor_rgb[take]
                valid |= take
                red &= ~take
                repaired += count
                progressed = True
        if not progressed:
            break

    return rgb, repaired, red


def normalize(
    source: str | Path,
    destination: str | Path,
    spec: Spec,
    *,
    margin: int = 16,
    alpha_threshold: int = 128,
    crop_threshold: int = 8,
) -> NormalizeResult:
    """Rewrite `source` into a spec-compliant PNG-32 at `destination`."""
    rgba = load_rgba(source)
    notes: list[str] = []

    # 1. Crop to the subject, ignoring the faintest fringe so it does not
    #    inflate the bounding box.
    box = subject_bbox(rgba[:, :, 3], threshold=crop_threshold)
    if box is None:
        raise ValueError(f"{source}: no subject above alpha {crop_threshold}")
    left, top, right, bottom = box
    cropped = rgba[top:bottom, left:right]
    src_h, src_w = cropped.shape[:2]

    # 2. Scale the subject as large as the canvas allows while keeping a margin
    #    on every side, so it can never touch the border.
    max_w = spec.width - 2 * margin
    max_h = spec.height - 2 * margin
    scale = min(max_h / src_h, max_w / src_w)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    if new_h < spec.subject_height_px_min:
        raise ValueError(
            f"{source}: subject fits to {new_h}px tall, below the required "
            f"{spec.subject_height_px_min}px — the source aspect ratio cannot "
            f"satisfy the spec on a {spec.width}x{spec.height} canvas"
        )

    with Image.fromarray(cropped, mode="RGBA") as img:
        resized = np.array(
            img.resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8
        )

    # 3. Centre it on a fully transparent canvas.
    canvas = np.zeros((spec.height, spec.width, 4), dtype=np.uint8)
    off_x = (spec.width - new_w) // 2
    off_y = (spec.height - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized

    alpha = canvas[:, :, 3]
    rgb = canvas[:, :, :3]

    # 4. Binarise alpha — this is what clears the partial-alpha failure.
    partial_before = int(np.count_nonzero((alpha > 0) & (alpha < 255)))
    opaque = alpha >= alpha_threshold
    alpha = np.where(opaque, 255, 0).astype(np.uint8)

    # 5. Drop specks before repairing red, so isolated red dust is not filled in
    #    and then deleted anyway.
    labels, areas = component_areas(alpha > 0)
    specks_removed = 0
    if areas.size:
        small = np.flatnonzero(areas <= spec.speck_max_area) + 1
        if small.size:
            speck_mask = np.isin(labels, small)
            alpha[speck_mask] = 0
            opaque = alpha > 0
            specks_removed = int(small.size)
            notes.append(f"{specks_removed}개 티끌 덩어리 제거")

    # 6. Repair the red fringe from surrounding colour.
    rgb, red_repaired, red_left = _repair_pure_red(rgb, opaque)
    red_dropped = int(np.count_nonzero(red_left))
    if red_dropped:
        alpha[red_left] = 0
        notes.append(f"주변 색을 찾지 못한 빨강 {red_dropped}px는 투명 처리")

    # 7. Zero RGB wherever the pixel is transparent.
    rgb[alpha == 0] = 0

    out = np.dstack([rgb, alpha])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.fromarray(out, mode="RGBA") as img:
        img.save(destination, format="PNG", optimize=True)

    return NormalizeResult(
        scale=scale,
        source_size=(src_w, src_h),
        placed_size=(new_w, new_h),
        binarised_px=partial_before,
        red_repaired_px=red_repaired,
        red_dropped_px=red_dropped,
        specks_removed=specks_removed,
        notes=tuple(notes),
    )
