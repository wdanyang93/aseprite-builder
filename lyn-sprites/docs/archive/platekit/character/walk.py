"""5-direction walk sheets -> validated 8-direction x 8-frame Godot character asset.

Direction mapping is **declared in config, never inferred**. Measurement can tell a front view
from a back view (eye-colour blobs: 4-5 = both eyes, 2 = one eye, 0 = back) but not E from SE:
the two side sheets in the real corpus score 0.489 silhouette IoU against each other, clearly
distinct yet not separable into named directions by any measurement available here. So the tool
reports what it sees and the person decides.

N and S are never mirror-derived. The three mirrored directions are NW<-NE, W<-E, SW<-SE.

Ground pivot excludes the tail. Lyn's tail is huge and sweeps sideways and low; using the
bottom-most pixel would put the pivot on the tail tip and the character would slide while walking.
The pivot is the lowest foreground **inside the core body's horizontal corridor**.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .. import __version__
from ..io import save_json, save_png, sha256_file
from .metrics import Frame, colour_classes, measure, split_sheet


def _head_width(mask: np.ndarray) -> float:
    """Widest row of the top quarter over figure height. Resolution independent proportion."""
    height = mask.shape[0]
    return float(mask[: max(1, int(height * 0.25))].sum(axis=1).max() / height)
from .qa import LYN_CANON, continuity_report, load_animation, orientation_report, style_signature

SCHEMA = "platekit.character_walk"
SCHEMA_VERSION = 1
DIRECTION_ORDER = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
MIRROR_RULES = {"NW": "NE", "W": "E", "SW": "SE"}
NEVER_MIRROR = {"N", "S"}

GROUND_DRIFT_WARN_PX = 4.0
SCALE_WARN = 0.05
SCALE_FAIL = 0.08
EDGE_GRADIENT_WARN = 3.0       # below this the silhouette edge is a smear, not a cut
CROSS_DIRECTION_SIZE_WARN = 0.10
CROSS_DIRECTION_SIZE_FAIL = 0.25
BOUNCE_IS_REAL_CORR = -0.30
SYMMETRY_THRESHOLD = 0.50      # self mirror IoU at or above this = a front or back view
SIDE_MARGIN = 0.015            # reference delta beyond this decides left vs right
ROW_OVERLAP_RATIO = 0.20       # grid rows overlap by this share of the pitch (a forward foot)    # height vs leg-spread correlation at or below this = real bounce


# ------------------------------------------------------------ geometry --

def core_corridor(mask: np.ndarray) -> tuple[int, int]:
    """Horizontal span of the body, tail excluded.

    The tail is a long low-frequency lobe off to one side; the body is the tall part. Rows in the
    upper 60 % contain head/torso/hips but never the tail tip, so their x-extent defines the
    corridor the feet must land in.
    """
    height = mask.shape[0]
    upper = mask[: int(height * 0.60)]
    cols = np.nonzero(upper.any(axis=0))[0]
    if cols.size == 0:
        cols = np.nonzero(mask.any(axis=0))[0]
    return int(cols.min()), int(cols.max()) + 1


def ground_pivot(mask: np.ndarray) -> tuple[float, int, float]:
    """(x, y, confidence) of the foot contact point, ignoring the tail."""
    x0, x1 = core_corridor(mask)
    corridor = np.zeros_like(mask)
    corridor[:, x0:x1] = True
    body = mask & corridor
    if not body.any():
        ys, xs = np.nonzero(mask)
        return float(xs.mean()), int(ys.max()), 0.0
    ys, xs = np.nonzero(body)
    ground_y = int(ys.max())
    band = ys >= ground_y - max(2, int(mask.shape[0] * 0.02))
    foot_x = xs[band]
    # confidence: a clean contact is a small number of foot-sized runs, not a smear
    row = body[ground_y - 1] if ground_y > 0 else body[ground_y]
    runs = int(np.count_nonzero(np.diff(row.astype(np.int8)) == 1))
    confidence = 1.0 if runs in (1, 2) else (0.7 if runs == 3 else 0.4)
    return float(foot_x.mean()), ground_y, confidence


def edge_softness(rgba: np.ndarray, mask: np.ndarray) -> float:
    """Mean |luminance gradient| in the 1 px band just inside the silhouette.

    This replaces a brightness-based "checker residue" metric that could not work for this
    character. Measured: counting opaque pixels brighter than 244 gave 4.8-7.4 %, but the interior
    scored as high as the boundary band (7.2 % vs 4.1 %), so it was counting Lyn's white bandeau,
    hair and tail - not background. A periodicity test settled it: the background checker carries
    1.5 units of high-frequency energy while the character's own detail carries 10.7-14.0, so any
    surviving checker is undetectable inside the figure.

    What *is* measurable is whether the cut edge is crisp. A soft, low-gradient edge means
    half-transparent background was left attached; a hard edge means a clean cut. Checker residue
    itself is therefore a human review item on the contact sheet, not an automated gate.
    """
    if not mask.any():
        return 0.0
    lum = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.abs(cv2.Laplacian(lum, cv2.CV_32F, ksize=3))
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    band = mask & ~inner
    return float(grad[band].mean()) if band.any() else 0.0


# --------------------------------------------------------------- build --

def _mirror_point(x: float, width: int) -> float:
    """Generic point transform, used for the pivot now and hand/weapon sockets later."""
    return float(width - 1 - x)


def body_height(mask: np.ndarray) -> int:
    """Head top to foot contact. Excludes the tail, which hangs low and would inflate the bbox."""
    gx, gy, _c = ground_pivot(mask)
    ys, _xs = np.nonzero(mask)
    return int(gy - int(ys.min()))


def walk_bounce_correlation(frames: Sequence[Frame]) -> float:
    """Is the height variation legitimate gait bounce, or is the character changing size?

    In a real walk the body is lowest when the legs are widest apart, so height and leg spread are
    *negatively* correlated. Measured on the delivered alpha set: +0.454, i.e. some frames are both
    tall and wide-legged. That is body-size drift, not bounce, so it must be normalised out.
    A strongly negative value means leave it alone - flattening real bounce makes the walk look
    like the character is gliding.
    """
    heights, spreads = [], []
    for f in frames:
        heights.append(float(body_height(f.mask)))
        low = f.mask[int(f.mask.shape[0] * 0.80):]
        cols = np.nonzero(low.any(axis=0))[0]
        spreads.append(float(cols.max() - cols.min() + 1) if cols.size else 0.0)
    # Either series can be flat (a back view whose leg spread barely changes), and corrcoef then
    # divides by zero and returns NaN. The JSON writer refuses NaN, which is how this surfaced.
    if len(heights) < 3 or float(np.std(spreads)) < 1e-6 or float(np.std(heights)) < 1e-6:
        return 0.0
    correlation = float(np.corrcoef(np.array(heights), np.array(spreads))[0, 1])
    return 0.0 if not np.isfinite(correlation) else correlation


def required_extents(frames: Sequence[Frame], canon_height: float) -> tuple[float, float, float]:
    """Pixels needed left of, right of, and above the ground pivot at the canon height.

    Returned per side rather than as a width, because a cell must satisfy the widest LEFT
    requirement and the widest RIGHT requirement independently. Summing per-row widths and taking
    the max gave 566 px, which still clipped E and SE: the west directions need 333 px on the left
    and the east directions need 236 px on the right, so the cell needs 333 + 236, not either row's
    own total.

    Lyn's tail sweeps far to one side, so a west-facing direction needs 549 px of width where the
    project's 512 cell only offers 512 - the three western directions were being clipped. Rather
    than silently cropping the tail or silently breaking canon, the needed size is reported.
    """
    left = right = top = 0.0
    for f in frames:
        gx, gy, _c = ground_pivot(f.mask)
        scale = canon_height / max(1.0, float(body_height(f.mask)))
        left = max(left, gx * scale)
        right = max(right, (f.mask.shape[1] - gx) * scale)
        top = max(top, gy * scale)
    return (float(left), float(right), float(top))


def _normalise(frames: Sequence[Frame], canon_height: float, cell: tuple[int, int],
               pivot: tuple[int, int], *, preserve_bounce: bool = False) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    cell_w, cell_h = cell
    px, py = pivot
    canvases, records = [], []
    median_body = float(np.median([body_height(f.mask) for f in frames]))
    for f in frames:
        mask = f.mask
        m = measure(f)
        gx, gy, conf = ground_pivot(mask)
        # Scale on head-to-ground height, not the bbox: the tail hangs below the feet and its sweep
        # would otherwise drive the scale. When the bounce is real, every frame gets the same
        # sheet-level scale so the rise and fall survives.
        body = float(body_height(mask))
        target = median_body if preserve_bounce else body
        scale = (canon_height / max(1.0, median_body)) if preserve_bounce else (canon_height / max(1.0, body))
        if abs(scale - 1.0) < 0.02:
            scale = 1.0
        rgba = f.rgba
        if scale != 1.0:
            nw = max(1, int(round(rgba.shape[1] * scale)))
            nh = max(1, int(round(rgba.shape[0] * scale)))
            rgba = cv2.resize(rgba, (nw, nh),
                              interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)
            gx, gy = gx * scale, gy * scale
        canvas = np.zeros((cell_h, cell_w, 4), np.uint8)
        ox, oy = int(round(px - gx)), int(round(py - gy))
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(cell_w, ox + rgba.shape[1]), min(cell_h, oy + rgba.shape[0])
        clipped = x0 > ox or y0 > oy or x1 < ox + rgba.shape[1] or y1 < oy + rgba.shape[0]
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = rgba[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        canvases.append(canvas)
        records.append({"index": f.index, "scale_applied": round(float(scale), 4),
                        "measured_height": int(m.height), "body_height": int(body),
                        "ground_confidence": round(conf, 2),
                        "translation": [ox, oy], "clipped": bool(clipped),
                        "edge_gradient": round(edge_softness(f.rgba, f.mask), 2)})
    return canvases, records


def _row_cuts(mask: np.ndarray, rows: int) -> list[int]:
    """Row boundaries placed where the sheet is emptiest, not at even multiples.

    Equal division sliced through every figure on the real grid: each of the eight nominal rows had
    130-270 foreground pixels sitting on its top or bottom edge, which is how the ears and feet were
    being cut off. The rows are still evenly pitched, so the search is a small window around each
    nominal cut for the row with the least content.
    """
    height = mask.shape[0]
    profile = mask.sum(axis=1).astype(float)
    pitch = height / float(rows)
    window = int(pitch * 0.22)
    cuts = []
    for r in range(1, rows):
        nominal = int(round(r * pitch))
        lo, hi = max(1, nominal - window), min(height - 1, nominal + window)
        segment = profile[lo:hi]
        cuts.append(lo + int(np.argmin(segment)) if segment.size else nominal)
    return cuts


def split_grid(rgba: np.ndarray, rows: int, cols: int) -> list[list[Frame]]:
    """Cut an N x M grid sheet into per-row frame lists.

    A full 8-direction set can arrive as one grid instead of five strips. Row boundaries are found
    by content (see :func:`_row_cuts`); columns reuse the strip logic's scored candidates.
    """
    from .metrics import ALPHA_KEEP, _cut_candidates, _score_cuts, foreground_mask

    mask = foreground_mask(rgba)
    height, width = mask.shape
    cuts = _row_cuts(mask, rows)
    # Rows genuinely overlap: a forward foot reaches past the emptiest line into the next row. The
    # bands are therefore widened downward and each figure is picked as the largest component inside
    # its band, so a neighbour's toe cannot be mistaken for part of this row.
    overlap = int(height / float(rows) * ROW_OVERLAP_RATIO)
    row_edges = [0] + cuts + [height]
    col_w = width / float(cols)
    out: list[list[Frame]] = []
    for r in range(rows):
        frames: list[Frame] = []
        y_lo = row_edges[r]
        y_hi = min(height, row_edges[r + 1] + overlap)
        band = mask[y_lo:y_hi]
        candidates = _cut_candidates(band, cols)
        col_cuts = (min(candidates, key=lambda mc: _score_cuts(band, mc[1]))[1]
                    if candidates else [int(round(i * col_w)) for i in range(1, cols)])
        col_edges = [0] + list(col_cuts) + [width]
        for c in range(cols):
            y0, y1 = y_lo, y_hi
            x0, x1 = col_edges[c], col_edges[c + 1]
            sub = mask[y0:y1, x0:x1]
            if not sub.any():
                raise ValueError(f"그리드 셀 ({r + 1},{c + 1}) 에 전경이 없습니다.")
            # Largest component only: the overlap band may also contain the next row's toes.
            n_comp, labels, stats, _cent = cv2.connectedComponentsWithStats(sub.astype(np.uint8), 8)
            if n_comp > 2:
                biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                keep = labels == biggest
                near = cv2.dilate(keep.astype(np.uint8),
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
                for i in range(1, n_comp):
                    if i != biggest and (labels == i) is not None and ((labels == i) & near).any():
                        keep |= labels == i
                sub = keep
            ys, xs = np.nonzero(sub)
            by0, by1 = int(ys.min()), int(ys.max()) + 1
            bx0, bx1 = int(xs.min()), int(xs.max()) + 1
            crop = rgba[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1].copy()
            crop_mask = sub[by0:by1, bx0:bx1]
            if crop.shape[2] == 4:
                crop[..., 3] = np.where(crop_mask, crop[..., 3], 0)
            if crop.shape[2] == 3 or int(crop[..., 3].min()) == 255:
                crop = np.dstack([crop[..., :3], crop_mask.astype(np.uint8) * 255])
            frames.append(Frame(index=c + 1, rgba=crop, mask=crop_mask, sheet_x0=x0 + bx0,
                                sheet_bbox=[x0 + bx0, y0 + by0, bx1 - bx0, by1 - by0]))
        frames[0].split_method = "grid_equal"
        out.append(frames)
    return out


def grid_direction_diagnostic(rgba: np.ndarray, rows: int = 8, cols: int = 8,
                              reference: Sequence[np.ndarray] | None = None) -> dict[str, Any]:
    """Which row is which direction, measured rather than assumed.

    Two signals, both resolution independent:

    * **self mirror symmetry** - a front or back view is left-right symmetric, a side or diagonal
      view is not. Measured on the real grid: rows 1 and 5 scored 0.69 and 0.72 while every other
      row sat at 0.26-0.34, so those two are S and N.
    * **side, against a known right-facing reference** - rows 2-4 preferred the mirrored reference
      (-0.047 to -0.085) and rows 6-8 preferred it unmirrored (+0.101 to +0.212).

    Together these place the rows without anybody having to eyeball the sheet. The eye-colour
    detector used for strips is useless here: a grid cell is ~150 px tall and the irises are a
    handful of pixels.
    """
    grid = split_grid(rgba, rows, cols)
    target = 256

    def resize(mask: np.ndarray) -> np.ndarray:
        factor = target / mask.shape[0]
        return cv2.resize(mask.astype(np.uint8),
                          (max(1, int(mask.shape[1] * factor)), target),
                          interpolation=cv2.INTER_NEAREST) > 0

    def overlap(a: np.ndarray, b: np.ndarray, mirror: bool = False) -> float:
        if mirror:
            a = a[:, ::-1]
        width = max(a.shape[1], b.shape[1])
        pa = np.zeros((target, width), bool)
        pb = np.zeros((target, width), bool)
        pa[:, (width - a.shape[1]) // 2:(width - a.shape[1]) // 2 + a.shape[1]] = a
        pb[:, (width - b.shape[1]) // 2:(width - b.shape[1]) // 2 + b.shape[1]] = b
        return float((pa & pb).sum() / max(1, (pa | pb).sum()))

    scaled = [[resize(f.mask) for f in row] for row in grid]
    reference_scaled = [resize(m) for m in reference] if reference else None
    report: list[dict[str, Any]] = []
    for r, row in enumerate(scaled):
        symmetry = float(np.mean([overlap(m, m, mirror=True) for m in row]))
        side, delta = None, None
        if reference_scaled:
            plain = float(np.mean([max(overlap(m, rm) for rm in reference_scaled) for m in row]))
            mirrored = float(np.mean([max(overlap(m, rm, mirror=True) for rm in reference_scaled) for m in row]))
            delta = plain - mirrored
            side = "right" if delta > SIDE_MARGIN else ("left" if delta < -SIDE_MARGIN else None)
        report.append({"row": r + 1, "self_symmetry": round(symmetry, 3),
                       "symmetric_view": symmetry >= SYMMETRY_THRESHOLD,
                       "reference_delta": None if delta is None else round(delta, 3), "side": side})
    return {"rows": report, "frames_per_row": cols}


def build_walk(config_path: str | Path, out: str | Path, *, fps: int = 8,
               canon: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    """One shot: declared sheets -> 64 normalised frames + manifest + Godot assets.

    Two input layouts. ``sources`` maps each direction to its own strip and the missing three are
    mirror derived. ``grid`` names one sheet holding all eight directions as rows, in which case
    nothing is derived - every direction is authored art.
    """
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    if cfg.get("grid"):
        return _build_walk_from_grid(cfg, root, out, fps=fps, canon=canon, force=force)
    character = str(cfg.get("character_id", "lyn"))
    action = str(cfg.get("action", "walk"))
    count = int(cfg.get("frame_count", 8))
    sources: dict[str, str] = cfg.get("sources", {})
    if not sources:
        raise ValueError("config에 sources가 없습니다.")
    mirror_rules = {str(k): str(v) for k, v in cfg.get("mirror_rules", MIRROR_RULES).items()}
    for derived, origin in mirror_rules.items():
        if derived in NEVER_MIRROR or origin in NEVER_MIRROR:
            raise ValueError(f"N/S는 미러 파생 대상이 아닙니다: {derived} <- {origin}")
        if origin not in sources:
            raise ValueError(f"미러 원본 '{origin}'이 sources에 없습니다.")
    missing = [d for d in DIRECTION_ORDER if d not in sources and d not in mirror_rules]
    if missing:
        raise ValueError(f"8방향을 채울 수 없습니다. 빠진 방향: {missing}")

    out_dir = Path(out)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)

    hashes_before = {d: sha256_file(root / p) for d, p in sources.items()}
    loaded: dict[str, list[Frame]] = {}
    warnings: list[str] = []
    hard: list[str] = []
    for direction, rel in sources.items():
        path = root / rel
        frames = load_animation(path, count)
        if len(frames) != count:
            hard.append(f"{direction}: 프레임 {len(frames)} != {count}")
        for f in frames:
            if f.sheet_bbox[0] <= 0:
                warnings.append(f"{direction} f{f.index}: 셀 왼쪽 경계에 닿았습니다")
        loaded[direction] = frames

    # Canon height priority: project spec > config > median of the sources. The existing Lyn spec
    # fixes 460 px inside a 512 cell with the pivot at (256, 478); taking the median of these
    # sheets instead gave 592 px, which does not fit the cell and clipped every direction.
    heights = [measure(f).height for frames in loaded.values() for f in frames]
    measured_median = float(np.median(heights))
    canon_height = cfg.get("canonical_height_px")
    canon_source = "config"
    if canon_height is None:
        canon_height = float(canon["height_px"]) if canon and "height_px" in canon else None
        canon_source = "project_spec"
    if canon_height is None:
        canon_height = measured_median
        canon_source = "source_median"
    canon_height = float(canon_height)
    cell = tuple(cfg.get("canvas", {}).get(k, v) for k, v in (("width", 512), ("height", 512)))
    pivot_cfg = cfg.get("pivot", {})
    pivot = (int(pivot_cfg.get("x", cell[0] // 2)), int(pivot_cfg.get("y", int(cell[1] * 0.934))))

    # ---- normalise the authored directions ----
    normalised: dict[str, list[np.ndarray]] = {}
    per_direction: dict[str, Any] = {}
    for direction, frames in loaded.items():
        bounce = walk_bounce_correlation(frames)
        preserve = bounce <= BOUNCE_IS_REAL_CORR
        canvases, records = _normalise(frames, canon_height, cell, pivot, preserve_bounce=preserve)
        normalised[direction] = canvases
        orient = orientation_report(frames, [measure(f) for f in frames], "auto")
        cont = continuity_report(frames, [measure(f) for f in frames], loop=True)
        # Raw per-crop foot x is meaningless (each crop is bbox-tight and the tail moves the left
        # edge). What matters is where the foot lands on the shared canvas after normalisation.
        placed = []
        for canvas in canvases:
            alpha = canvas[..., 3] > 8
            if alpha.any():
                gx, gy, _c = ground_pivot(alpha)
                placed.append((gx, gy))
        drift_x = float(max(p[0] for p in placed) - min(p[0] for p in placed)) if placed else 0.0
        drift_y = float(max(p[1] for p in placed) - min(p[1] for p in placed)) if placed else 0.0
        # Two different things were being conflated. The scale to canon is a deliberate uniform
        # resize of the whole sheet (0.66-0.83 here, because the sources are drawn ~590 px tall and
        # the canon is 460) and is not a defect. What matters is the *spread within a direction*:
        # do the eight frames of one animation agree with each other?
        scales = [r["scale_applied"] for r in records]
        spread = (max(scales) - min(scales)) / max(1e-6, float(np.mean(scales)))
        # The input drift is what the sheet arrived with; it is corrected per frame, so the number
        # that matters is what is LEFT on the normalised canvases. Reporting the input as a failure
        # flagged sheets the pipeline had already fixed.
        placed_heights = []
        for canvas in canvases:
            alpha = canvas[..., 3] > 8
            if alpha.any():
                ys = np.nonzero(alpha)[0]
                placed_heights.append(float(ys.max() - ys.min()))
        residual = ((max(placed_heights) - min(placed_heights)) / max(1e-6, float(np.mean(placed_heights)))
                    if placed_heights else 0.0)
        flags: list[str] = []
        if residual >= SCALE_FAIL:
            flags.append(f"SCALE_DRIFT_AFTER_NORMALISE {residual:.1%}")
        elif residual >= SCALE_WARN:
            flags.append(f"residual scale {residual:.1%}")
        if any(r["ground_confidence"] < 0.7 for r in records):
            flags.append("GROUND_POINT_LOW_CONFIDENCE")
        if drift_y > GROUND_DRIFT_WARN_PX:
            flags.append(f"FOOT_SLIDE_Y {drift_y:.0f}px")
        elif drift_x > GROUND_DRIFT_WARN_PX * 4:
            flags.append(f"foot_slide_x {drift_x:.0f}px")
        if any(r["edge_gradient"] < EDGE_GRADIENT_WARN for r in records):
            flags.append("SOFT_EDGE_possible_background_bleed")
        if any(r["clipped"] for r in records):
            hard.append(f"{direction}: 정규화 캔버스에서 잘렸습니다")
        outliers = [p["pair"] for p in cont["pairs"] if p["outlier"]]
        per_direction[direction] = {
            "source": sources[direction], "derived": False,
            "bounce_correlation": round(bounce, 3),
            "height_policy": "preserved_gait_bounce" if preserve else "normalised_per_frame",
            "frames": records, "canon_scale": [round(min(scales), 3), round(max(scales), 3)],
            "input_scale_spread": round(spread, 4), "residual_scale_spread": round(residual, 4),
            "ground_x_drift_px": round(drift_x, 1), "ground_y_drift_px": round(drift_y, 1),
            "orientation": {"intended": orient["intended_facing"],
                            "flip_candidates": orient["flip_candidates"],
                            "transitions": orient["facing_transitions"]},
            "continuity_outliers": outliers, "flags": flags,
        }
        if orient["flip_candidates"]:
            warnings.append(f"{direction}: 방향 뒤집힘 후보 {orient['flip_candidates']} "
                            "(자동 수정하지 않습니다. character fix로 확인 후 처리하세요)")
        if outliers:
            warnings.append(f"{direction}: 연속성 이상치 {outliers}")
        warnings.extend(f"{direction}: {f}" for f in flags)

    # Cross-direction size agreement: if one sheet is drawn much bigger than another, the
    # character changes size when the player turns. Measured on the real corpus: S needs 0.66 and
    # SE needs 0.77, a 16 % difference that normalisation hides but the eye notices.
    # Must be resolution independent. Comparing canon scale is wrong: the E sheet arrives as
    # 1024x1536 singles (~1400 px figure) while the others are ~590 px, so the scale factor differs
    # 72 % purely from source resolution and says nothing about the drawing. head_width_ratio
    # (widest row of the top quarter / figure height) is a pure proportion and comparable.
    authored_scales = {d: float(np.mean([f["scale_applied"] for f in per_direction[d]["frames"]]))
                       for d in sources}
    head_ratios = {d: float(np.median([_head_width(f.mask) for f in loaded[d]])) for d in sources}
    cross = (max(head_ratios.values()) - min(head_ratios.values())) / max(1e-6, float(np.mean(list(head_ratios.values()))))
    biggest = max(head_ratios, key=head_ratios.get)
    smallest = min(head_ratios, key=head_ratios.get)
    if cross >= CROSS_DIRECTION_SIZE_FAIL:
        hard.append(f"방향 간 머리 비율 차이 {cross:.1%} (가장 큼 {biggest} {head_ratios[biggest]:.3f}, "
                    f"가장 작음 {smallest} {head_ratios[smallest]:.3f}) - 원화 재발주 대상")
    elif cross >= CROSS_DIRECTION_SIZE_WARN:
        warnings.append(f"방향 간 머리 비율 차이 {cross:.1%} "
                        f"({biggest} {head_ratios[biggest]:.3f} vs {smallest} {head_ratios[smallest]:.3f}): "
                        "돌아설 때 머리 크기가 변해 보일 수 있습니다")

    # ---- mirror derivation ----
    for derived, origin in mirror_rules.items():
        src = normalised[origin]
        normalised[derived] = [c[:, ::-1].copy() for c in src]
        per_direction[derived] = {
            "source": sources[origin], "derived": True,
            "derived_from": {"direction": origin, "transform": "mirror_x"},
            "pivot": [_mirror_point(pivot[0], cell[0]), pivot[1]],
            "frames": [{"index": i + 1, "derived_from_frame": i + 1, "transform": "mirror_x"}
                       for i in range(count)],
            "flags": [],
        }

    # ---- write final frames ----
    total = 0
    for direction in DIRECTION_ORDER:
        d = out_dir / "final" / direction
        d.mkdir(parents=True, exist_ok=True)
        for i, canvas in enumerate(normalised[direction]):
            save_png(d / f"frame_{i:02d}.png", canvas, force=True)
            total += 1
    if total != len(DIRECTION_ORDER) * count:
        hard.append(f"최종 프레임 {total} != {len(DIRECTION_ORDER) * count}")

    # ---- mirror round trip must be exact ----
    for derived, origin in mirror_rules.items():
        for a, b in zip(normalised[derived], normalised[origin]):
            if not np.array_equal(a[:, ::-1], b):
                hard.append(f"{derived}: mirror round-trip 불일치")
                break

    for direction, rel in sources.items():
        if sha256_file(root / rel) != hashes_before[direction]:
            hard.append(f"{direction}: 원본이 변경되었습니다 (SOURCE_MODIFIED)")

    save_png(out_dir / "reports" / "eight_direction_contact.png",
             _contact_grid(normalised, mirror_rules, count), force=True)

    status = "fail" if hard else ("warn" if warnings else "pass")
    manifest = {
        "schema": SCHEMA, "version": SCHEMA_VERSION, "platekit_version": __version__,
        "character_id": character, "action": action,
        "direction_order": DIRECTION_ORDER, "fps": fps, "loop": True,
        "canvas": {"width": cell[0], "height": cell[1]},
        "pivot": {"kind": "ground", "x": pivot[0], "y": pivot[1],
                  "note": "꼬리를 제외한 몸통 통로 안의 최하단 = 발 접지점"},
        "canonical_height_px": round(canon_height, 1),
        "canonical_height_source": canon_source,
        "measured_source_median_px": round(measured_median, 1),
        "mapping_source": str(cfg.get("mapping_source", "declared")),
        "source_directions": sorted(sources), "mirror_rules": mirror_rules,
        "source_sha256": hashes_before,
        "directions": per_direction,
        "cross_direction_head_ratio_spread": round(cross, 4),
        "head_width_ratio": {k: round(v, 3) for k, v in head_ratios.items()},
        "authored_canon_scale": {k: round(v, 3) for k, v in authored_scales.items()},
        "review_notes": ["체커 잔존은 자동 판정 불가(캐릭터 흰 부분과 배경이 구분되지 않음). "
                         "reports/eight_direction_contact.png 를 사람이 확인해야 합니다."],
        "qa": {"source_frames": len(sources) * count,
               "derived_frames": len(mirror_rules) * count,
               "total_frames": total, "status": status,
               "hard_failures": hard, "warnings": warnings},
    }
    save_json(out_dir / "walk.json", manifest)
    return manifest


def _contact_grid(normalised: dict[str, list[np.ndarray]], mirror_rules: dict[str, str],
                  count: int, tile: int = 128) -> np.ndarray:
    rows = []
    for direction in DIRECTION_ORDER:
        tiles = []
        for canvas in normalised[direction]:
            rgb = canvas[..., :3].copy()
            alpha = canvas[..., 3] > 8
            rgb[~alpha] = 60
            t = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_AREA)
            tiles.append(t)
        label = np.full((tile, 96, 3), 30, np.uint8)
        cv2.putText(label, direction, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        if direction in mirror_rules:
            cv2.putText(label, "MIRROR", (6, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(label, f"<-{mirror_rules[direction]}", (6, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (120, 200, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate([label] + tiles, axis=1))
    return np.concatenate(rows, axis=0)


def direction_diagnostic(sheets: dict[str, str | Path], count: int = 8) -> dict[str, Any]:
    """What the images say about each sheet's view. Diagnostic only - never a mapping decision.

    Eye-coloured blobs in the head band: 4-5 = both eyes (front), 2 = one eye (side or 3/4),
    0 = no eyes (back). Distinguishing E from SE is beyond this and is left to the person.
    """
    out: dict[str, Any] = {}
    for label, path in sheets.items():
        frames = load_animation(path, count)
        blobs, sides = [], []
        for f in frames:
            rgb = f.rgba[..., :3].copy()
            rgb[~f.mask] = 0
            classes = colour_classes(rgb, f.mask)
            band = np.zeros_like(f.mask)
            band[: int(f.mask.shape[0] * 0.35)] = True
            eyes = classes["eyes"] & band
            n, _lab, stats, _c = cv2.connectedComponentsWithStats(eyes.astype(np.uint8), 8)
            keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 40]
            blobs.append(len(keep))
            if keep:
                cx = float(np.mean([stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2 for i in keep]))
                sides.append(cx / f.mask.shape[1])
        median_blobs = float(np.median(blobs))
        view = "front" if median_blobs >= 4 else ("back" if median_blobs < 1 else "side_or_three_quarter")
        out[label] = {"eye_blobs_per_frame": blobs, "view_guess": view,
                      "eye_x_fraction": round(float(np.median(sides)), 3) if sides else None,
                      "facing_guess": None if not sides else ("right" if np.median(sides) > 0.5 else "left")}
    return out


# --------------------------------------------------------- godot export --

def export_godot(manifest: dict[str, Any], walk_out: str | Path, godot_out: str | Path,
                 res_base: str, *, force: bool = False) -> dict[str, Any]:
    """SpriteFrames + a playable demo scene.

    Textures are declared as ExtResource, never loaded from a runtime-built string: an earlier
    milestone shipped a demo whose textures were only reachable by `load("res://...")` and they
    were unreachable in an exported build.
    """
    walk_dir = Path(walk_out)
    root = Path(godot_out)
    base = res_base.rstrip("/")
    count = len(manifest["directions"][manifest["direction_order"][0]]["frames"])
    fps = int(manifest["fps"])
    action = manifest["action"]

    # copy the frames into the Godot project
    art = root / "art" / manifest["character_id"] / action
    for direction in manifest["direction_order"]:
        dst = art / direction
        dst.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            src = walk_dir / "final" / direction / f"frame_{i:02d}.png"
            (dst / f"frame_{i:02d}.png").write_bytes(src.read_bytes())

    # ---- SpriteFrames .tres ----
    ext_lines, anim_lines = [], []
    rid = 1
    for direction in manifest["direction_order"]:
        ids = []
        for i in range(count):
            ids.append(f"{rid}_{direction.lower()}{i}")
            ext_lines.append(f'[ext_resource type="Texture2D" '
                             f'path="{base}/{action}/{direction}/frame_{i:02d}.png" id="{ids[-1]}"]')
            rid += 1
        frames = ", ".join('{\n"duration": 1.0,\n"texture": ExtResource("%s")\n}' % i for i in ids)
        anim_lines.append('{\n"frames": [%s],\n"loop": true,\n"name": &"%s_%s",\n"speed": %d.0\n}'
                          % (frames, action, direction.lower(), fps))
    tres = ['[gd_resource type="SpriteFrames" load_steps=%d format=3]' % (len(ext_lines) + 1), '']
    tres += ext_lines
    tres += ['', '[resource]', 'animations = [%s]' % ", ".join(anim_lines)]
    frames_path = root / "art" / manifest["character_id"] / f"{action}_frames.tres"
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frames_path.write_text("\n".join(tres) + "\n", encoding="utf-8", newline="\n")

    # ---- controller ----
    controller = '''class_name LynWalkController
extends CharacterBody2D

## 8-way walk driven by the SpriteFrames platekit generated.
##
## The direction table is the project's canonical enum order (0:S 1:SW 2:W 3:NW 4:N 5:NE 6:E 7:SE)
## and must not be reordered. Vertical input is scaled because the room plates are 3/4 view: a
## full-speed vertical step covers less ground on screen than a horizontal one.

const DIRECTIONS := ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
const DIAG := 0.7071068
const VECTORS := [
	Vector2(0, 1), Vector2(-DIAG, DIAG), Vector2(-1, 0), Vector2(-DIAG, -DIAG),
	Vector2(0, -1), Vector2(DIAG, -DIAG), Vector2(1, 0), Vector2(DIAG, DIAG),
]

@export var speed: float = 220.0
@export var vertical_factor: float = 0.75
@export var animation_prefix: String = "walk_"

var facing_index: int = 0

@onready var sprite: AnimatedSprite2D = $AnimatedSprite2D


func _ready() -> void:
	_play(facing_index)


func _physics_process(_delta: float) -> void:
	var dir := Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")
	if dir != Vector2.ZERO:
		facing_index = _nearest_direction(dir)
		_play(facing_index)
		if not sprite.is_playing():
			sprite.play()
	else:
		sprite.pause()
	velocity = Vector2(dir.x, dir.y * vertical_factor).normalized() * speed * dir.length()
	move_and_slide()


func _nearest_direction(dir: Vector2) -> int:
	var n := dir.normalized()
	var best := 0
	var best_dot := -2.0
	for i in VECTORS.size():
		var d: float = n.dot(VECTORS[i])
		if d > best_dot:
			best_dot = d
			best = i
	return best


func _play(index: int) -> void:
	var name := animation_prefix + String(DIRECTIONS[index]).to_lower()
	if sprite.sprite_frames != null and sprite.sprite_frames.has_animation(name):
		if sprite.animation != name:
			sprite.animation = name
	else:
		push_warning("platekit: 애니메이션이 없습니다: %s" % name)


func current_direction() -> String:
	return DIRECTIONS[facing_index]
'''
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "lyn_walk_controller.gd").write_text(controller, encoding="utf-8", newline="\n")

    demo = '''extends Node2D

## Neutral grid so foot sliding, vertical bob and pivot wobble are visible.
## F1 grid, F2 pivot marker, F3 fps cycle, F4 auto-rotate through all eight directions.

@onready var actor: CharacterBody2D = $Lyn
@onready var sprite: AnimatedSprite2D = $Lyn/AnimatedSprite2D
@onready var hud: Label = $UI/Info
@onready var grid: Node2D = $Grid
@onready var pivot_marker: Node2D = $Lyn/PivotMarker

var _fps_steps := [4, 6, 8, 10, 12]
var _fps_index := 2
var _auto := false
var _auto_t := 0.0
var _auto_index := 0


func _ready() -> void:
	pivot_marker.visible = false
	_apply_fps()


func _process(delta: float) -> void:
	if _auto:
		_auto_t += delta
		if _auto_t >= 1.0:
			_auto_t = 0.0
			_auto_index = (_auto_index + 1) % 8
			actor.facing_index = _auto_index
			actor._play(_auto_index)
			sprite.play()
	var derived := ["SW", "W", "NW"].has(actor.current_direction())
	hud.text = "Direction: %s\\nAnimation: %s\\nFrame: %d/%d\\nDerived: %s\\nFPS: %d\\n\\nArrows/WASD move · F1 grid · F2 pivot · F3 fps · F4 auto" % [
		actor.current_direction(), sprite.animation, sprite.frame + 1,
		sprite.sprite_frames.get_frame_count(sprite.animation), "mirror_x" if derived else "no",
		_fps_steps[_fps_index]]


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey) or not event.pressed or event.echo:
		return
	match int((event as InputEventKey).keycode):
		KEY_F1:
			grid.visible = not grid.visible
		KEY_F2:
			pivot_marker.visible = not pivot_marker.visible
		KEY_F3:
			_fps_index = (_fps_index + 1) % _fps_steps.size()
			_apply_fps()
		KEY_F4:
			_auto = not _auto


func _apply_fps() -> void:
	var frames: SpriteFrames = sprite.sprite_frames
	for name in frames.get_animation_names():
		frames.set_animation_speed(name, float(_fps_steps[_fps_index]))
'''
    (root / "scripts" / "walk_demo.gd").write_text(demo, encoding="utf-8", newline="\n")

    scene = f'''[gd_scene load_steps=5 format=3]

[ext_resource type="Script" path="res://scripts/walk_demo.gd" id="1_demo"]
[ext_resource type="Script" path="res://scripts/lyn_walk_controller.gd" id="2_ctrl"]
[ext_resource type="SpriteFrames" path="{base}/{action}_frames.tres" id="3_frames"]

[sub_resource type="CircleShape2D" id="CircleShape2D_body"]
radius = 12.0

[node name="WalkDemo" type="Node2D"]
script = ExtResource("1_demo")

[node name="Grid" type="Node2D" parent="."]

[node name="Ground" type="Line2D" parent="Grid"]
width = 2.0
default_color = Color(0.35, 0.38, 0.42, 1)
points = PackedVector2Array(-2000, 0, 2000, 0)

[node name="Lyn" type="CharacterBody2D" parent="." groups=["player"]]
script = ExtResource("2_ctrl")
collision_layer = 1
collision_mask = 1

[node name="Shape" type="CollisionShape2D" parent="Lyn"]
shape = SubResource("CircleShape2D_body")

[node name="AnimatedSprite2D" type="AnimatedSprite2D" parent="Lyn"]
sprite_frames = ExtResource("3_frames")
animation = &"{action}_s"
autoplay = "{action}_s"
offset = Vector2(0, -222)
texture_filter = 2

[node name="PivotMarker" type="Line2D" parent="Lyn"]
width = 2.0
default_color = Color(1, 0.35, 0.2, 1)
points = PackedVector2Array(-14, 0, 14, 0, 0, 0, 0, -14)

[node name="Camera" type="Camera2D" parent="Lyn"]
position_smoothing_enabled = true
position_smoothing_speed = 8.0
zoom = Vector2(1.6, 1.6)

[node name="UI" type="CanvasLayer" parent="."]
layer = 10

[node name="Info" type="Label" parent="UI"]
offset_left = 16.0
offset_top = 12.0
offset_right = 520.0
offset_bottom = 190.0
'''
    (root / "scenes").mkdir(exist_ok=True)
    (root / "scenes" / "walk_demo.tscn").write_text(scene, encoding="utf-8", newline="\n")

    project = f'''; Generated by platekit {__version__}. Walk verification project.
config_version=5

[application]
config/name="Lyn Walk Demo"
run/main_scene="res://scenes/walk_demo.tscn"
config/features=PackedStringArray("4.3", "GL Compatibility")

[display]
window/size/viewport_width=1280
window/size/viewport_height=720
window/stretch/mode="canvas_items"

[rendering]
renderer/rendering_method="gl_compatibility"
textures/canvas_textures/default_texture_filter=1
'''
    (root / "project.godot").write_text(project, encoding="utf-8", newline="\n")

    animations = [f"{action}_{d.lower()}" for d in manifest["direction_order"]]
    return {"project": str(root), "sprite_frames": str(frames_path),
            "animations": animations, "frames_per_animation": count,
            "texture_ext_resources": len(ext_lines), "fps": fps}

# --------------------------------------------------- Godot export --

def export_godot(manifest, out, *, res_base="res://art/lyn/walk", scene_name="LynWalk"):
    """SpriteFrames + a player scene, using ExtResource references.

    Every texture is declared as an ext_resource rather than loaded from a built string. A
    string-only path can be stripped from an export build, which is the failure this project
    already hit once with slot variants.
    """
    out_dir = Path(out)
    (out_dir / "godot").mkdir(parents=True, exist_ok=True)
    directions = manifest["direction_order"]
    count = int(manifest["qa"]["total_frames"] / len(directions))
    fps = int(manifest.get("fps", 8))
    base = res_base.rstrip("/")

    ext = []
    for direction in directions:
        for i in range(count):
            ext.append(("%d_%s%d" % (len(ext) + 1, direction.lower(), i),
                        "%s/%s/frame_%02d.png" % (base, direction, i)))
    lines = ['[gd_resource type="SpriteFrames" load_steps=%d format=3]' % (len(ext) + 1), ""]
    for rid, path in ext:
        lines.append('[ext_resource type="Texture2D" path="%s" id="%s"]' % (path, rid))
    lines += ["", "[resource]", "animations = ["]
    idx = 0
    for direction in directions:
        frames = []
        for _i in range(count):
            frames.append('{\n"duration": 1.0,\n"texture": ExtResource("%s")\n}' % ext[idx][0])
            idx += 1
        lines.append('{\n"frames": [%s],\n"loop": true,\n"name": &"walk_%s",\n"speed": %.1f\n},'
                     % (", ".join(frames), direction.lower(), float(fps)))
    lines += ["]", ""]
    tres = out_dir / "godot" / "lyn_walk_frames.tres"
    tres.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    pivot_y = int(manifest["pivot"]["y"])
    cell_h = int(manifest["canvas"]["height"])
    # AnimatedSprite2D is centred, so shift it so the ground pivot sits on the node origin.
    offset_y = -(pivot_y - cell_h / 2.0)
    scene = "\n".join([
        '[gd_scene load_steps=4 format=3]', '',
        '[ext_resource type="SpriteFrames" path="%s/lyn_walk_frames.tres" id="1_frames"]' % base,
        '[ext_resource type="Script" path="res://addons/platekit/platekit_walk_actor.gd" id="2_actor"]',
        '', '[sub_resource type="CircleShape2D" id="CircleShape2D_lyn"]', 'radius = 9.0', '',
        '[node name="%s" type="CharacterBody2D" groups=["player"]]' % scene_name,
        'collision_layer = 1', 'collision_mask = 1', '',
        '[node name="Shape" type="CollisionShape2D" parent="."]',
        'shape = SubResource("CircleShape2D_lyn")', '',
        '[node name="Visual" type="AnimatedSprite2D" parent="."]',
        'sprite_frames = ExtResource("1_frames")', 'animation = &"walk_s"',
        'offset = Vector2(0, %.1f)' % offset_y, 'texture_filter = 2', '',
        '[node name="WalkActor" type="Node" parent="."]',
        'script = ExtResource("2_actor")', 'sprite_path = NodePath("../Visual")', '',
    ])
    (out_dir / "godot" / ("%s.tscn" % scene_name)).write_text(scene, encoding="utf-8", newline="\n")
    return {"spriteframes": str(tres), "scene": str(out_dir / "godot" / ("%s.tscn" % scene_name)),
            "animations": ["walk_%s" % d.lower() for d in directions],
            "texture_refs": len(ext), "res_base": base, "sprite_offset_y": round(offset_y, 1)}


def _build_walk_from_grid(cfg, root, out, *, fps, canon, force):
    """All eight directions authored as rows of one grid sheet. Nothing is mirror derived."""
    grid_cfg = cfg["grid"]
    sheet_path = root / str(grid_cfg["file"])
    rows = [str(d) for d in grid_cfg.get("row_order", DIRECTION_ORDER)]
    count = int(grid_cfg.get("frames_per_row", cfg.get("frame_count", 8)))
    if sorted(rows) != sorted(DIRECTION_ORDER):
        raise ValueError(f"row_order가 8방향을 모두 담고 있지 않습니다: {rows}")

    from PIL import Image as _Image
    rgba = np.asarray(_Image.open(sheet_path).convert("RGBA")).copy()
    sha_before = sha256_file(sheet_path)
    grid = split_grid(rgba, len(rows), count)

    warnings: list[str] = []
    hard: list[str] = []
    heights = [body_height(f.mask) for row in grid for f in row]
    canon_height = cfg.get("canonical_height_px")
    canon_source = "config"
    if canon_height is None:
        canon_height = float(canon["height_px"]) if canon and "height_px" in canon else None
        canon_source = "project_spec"
    if canon_height is None:
        canon_height = float(np.median(heights))
        canon_source = "source_median"
    canon_height = float(canon_height)
    cell = (int(cfg.get("canvas", {}).get("width", 512)), int(cfg.get("canvas", {}).get("height", 512)))
    pivot_cfg = cfg.get("pivot", {})
    pivot = (int(pivot_cfg.get("x", cell[0] // 2)), int(pivot_cfg.get("y", int(cell[1] * 0.934))))
    # One cell must hold every direction, so take the widest requirement across all rows.
    margin = 8
    extents = [required_extents(row, canon_height) for row in grid]
    max_left = max(e[0] for e in extents)
    max_right = max(e[1] for e in extents)
    max_top = max(e[2] for e in extents)
    need_w = int(np.ceil(max_left + max_right)) + 2 * margin
    need_h = int(np.ceil(max_top)) + 2 * margin
    need_px = int(np.ceil(max_left)) + margin
    if need_w > cell[0] or need_h > cell[1]:
        grown = (max(cell[0], need_w), max(cell[1], need_h))
        warnings.append(
            f"캔버스를 {cell[0]}x{cell[1]} -> {grown[0]}x{grown[1]} 로 넓혔습니다. "
            f"꼬리가 서쪽 방향에서 {need_w}px 를 요구합니다 (프로젝트 규격은 {cell[0]}px). "
            "규격을 유지해야 하면 원화의 꼬리 폭을 줄이거나 규격을 갱신하세요.")
        pivot = (max(pivot[0], need_px), grown[1] - margin)
        cell = grown

    out_dir = Path(out)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)
    normalised: dict[str, list[np.ndarray]] = {}
    per_direction: dict[str, Any] = {}
    for direction, frames in zip(rows, grid):
        bounce = walk_bounce_correlation(frames)
        preserve = bounce <= BOUNCE_IS_REAL_CORR
        canvases, records = _normalise(frames, canon_height, cell, pivot, preserve_bounce=preserve)
        normalised[direction] = canvases
        placed = []
        for canvas in canvases:
            alpha = canvas[..., 3] > 8
            if alpha.any():
                gx, gy, _c = ground_pivot(alpha)
                placed.append((gx, gy))
        drift_x = float(max(p[0] for p in placed) - min(p[0] for p in placed)) if placed else 0.0
        drift_y = float(max(p[1] for p in placed) - min(p[1] for p in placed)) if placed else 0.0
        heights_placed = []
        for canvas in canvases:
            alpha = canvas[..., 3] > 8
            if alpha.any():
                ys = np.nonzero(alpha)[0]
                heights_placed.append(float(ys.max() - ys.min()))
        residual = ((max(heights_placed) - min(heights_placed)) / max(1e-6, float(np.mean(heights_placed)))
                    if heights_placed else 0.0)
        flags: list[str] = []
        if residual >= SCALE_FAIL:
            flags.append(f"SCALE_DRIFT_AFTER_NORMALISE {residual:.1%}")
        if drift_y > GROUND_DRIFT_WARN_PX:
            flags.append(f"FOOT_SLIDE_Y {drift_y:.0f}px")
        if any(r["clipped"] for r in records):
            hard.append(f"{direction}: 정규화 캔버스에서 잘렸습니다")
        cont = continuity_report(frames, [measure(f) for f in frames], loop=True)
        outliers = [p["pair"] for p in cont["pairs"] if p["outlier"]]
        per_direction[direction] = {
            "source": str(grid_cfg["file"]), "derived": False, "from_grid_row": rows.index(direction) + 1,
            "bounce_correlation": round(bounce, 3),
            "height_policy": "preserved_gait_bounce" if preserve else "normalised_per_frame",
            "frames": records, "residual_scale_spread": round(residual, 4),
            "ground_x_drift_px": round(drift_x, 1), "ground_y_drift_px": round(drift_y, 1),
            "continuity_outliers": outliers, "flags": flags,
        }
        warnings.extend(f"{direction}: {f}" for f in flags)
        if outliers:
            warnings.append(f"{direction}: 연속성 이상치 {outliers}")

    head_ratios = {d: float(np.median([_head_width(f.mask) for f in row])) for d, row in zip(rows, grid)}
    cross = (max(head_ratios.values()) - min(head_ratios.values())) / max(1e-6, float(np.mean(list(head_ratios.values()))))
    if cross >= CROSS_DIRECTION_SIZE_FAIL:
        hard.append(f"방향 간 머리 비율 차이 {cross:.1%} - 원화 재발주 대상")
    elif cross >= CROSS_DIRECTION_SIZE_WARN:
        warnings.append(f"방향 간 머리 비율 차이 {cross:.1%}")

    total = 0
    for direction in DIRECTION_ORDER:
        d = out_dir / "final" / direction
        d.mkdir(parents=True, exist_ok=True)
        for i, canvas in enumerate(normalised[direction]):
            save_png(d / f"frame_{i:02d}.png", canvas, force=True)
            total += 1
    if total != len(DIRECTION_ORDER) * count:
        hard.append(f"최종 프레임 {total} != {len(DIRECTION_ORDER) * count}")
    if sha256_file(sheet_path) != sha_before:
        hard.append("원본이 변경되었습니다 (SOURCE_MODIFIED)")

    save_png(out_dir / "reports" / "eight_direction_contact.png",
             _contact_grid(normalised, {}, count), force=True)
    status = "fail" if hard else ("warn" if warnings else "pass")
    manifest = {
        "schema": SCHEMA, "version": SCHEMA_VERSION, "platekit_version": __version__,
        "character_id": str(cfg.get("character_id", "lyn")), "action": str(cfg.get("action", "walk")),
        "direction_order": DIRECTION_ORDER, "fps": fps, "loop": True,
        "canvas": {"width": cell[0], "height": cell[1]},
        "pivot": {"kind": "ground", "x": pivot[0], "y": pivot[1],
                  "note": "꼬리를 제외한 몸통 통로 안의 최하단 = 발 접지점"},
        "canonical_height_px": round(canon_height, 1), "canonical_height_source": canon_source,
        "layout": "grid", "grid_row_order": rows,
        "canvas_required_px": [need_w, need_h], "canvas_grown": bool(cell != (
            int(cfg.get("canvas", {}).get("width", 512)), int(cfg.get("canvas", {}).get("height", 512)))),
        "mapping_source": str(cfg.get("mapping_source", "declared")),
        "source_directions": rows, "mirror_rules": {},
        "source_sha256": {"grid": sha_before},
        "cross_direction_head_ratio_spread": round(cross, 4),
        "head_width_ratio": {k: round(v, 3) for k, v in head_ratios.items()},
        "directions": per_direction,
        "qa": {"source_frames": total, "derived_frames": 0, "total_frames": total,
               "status": status, "hard_failures": hard, "warnings": warnings},
    }
    save_json(out_dir / "walk.json", manifest)
    return manifest
