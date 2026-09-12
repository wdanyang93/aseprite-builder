"""Accumulating character corpus. Nothing is ever discarded or replaced.

Every sheet delivered so far was previously judged against whatever was current and the loser was
thrown away. That was wrong twice over: a sheet that loses on proportion may still hold the best
frame for one direction, and a reference that is not art at all - a plain gait silhouette - is
worth keeping precisely because it is not art.

So this module only ever *adds*. Each ingest records the source hash, the detected layout, and one
row per extracted frame with its measurements. The canon is then derived from the aggregate of the
whole library rather than chosen by picking a file, which means it stops moving every time a new
sheet arrives.

    ingest(file)  ->  layout detected  ->  frames split  ->  rows appended to index.json
    derive_canon(index) -> one standard from the medians of the whole corpus
    conform(frame, canon) -> scale + colour + mirror + crop, so anything can join the set
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from .. import __version__
from ..io import save_json, save_png
from .metrics import colour_classes, detect_background, foreground_mask, measure

Image.MAX_IMAGE_PIXELS = None

INDEX_NAME = "index.json"
SCHEMA = "platekit.character_corpus"
SCHEMA_VERSION = 1

# layout detection
MIN_FRAME_SPAN = 0.04         # a frame must span at least this fraction of the sheet width
GRID_ROW_MIN = 2
MAX_GRID_ROWS = 12
ROW_PERIOD_MIN_STRENGTH = 0.35   # autocorrelation peak below this is not a real grid
COL_PERIOD_MIN_STRENGTH = 0.30
MAX_STRIP_COLS = 16
MERGED_SPAN_FACTOR = 1.6      # a span this much wider than one cell holds more than one figure
CELL_ASPECT_MIN = 0.30       # a cell holding one standing figure
CELL_ASPECT_MAX = 1.30
CELL_ASPECT_IDEAL = 0.65
SILHOUETTE_SAT_MAX = 30       # a pure silhouette has almost no colour
SILHOUETTE_COVER = 0.80       # ... and nearly all of its *interior* is one flat value


@dataclass
class FrameRow:
    frame_id: str
    source_hash: str
    source_name: str
    layout: str
    row: int
    col: int
    file: str
    width: int
    height: int
    body_height: int
    head_width_ratio: float
    skin_lab: list[float] | None
    is_silhouette: bool
    alpha_native: bool
    is_character: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["head_width_ratio"] = round(self.head_width_ratio, 4)
        if self.skin_lab:
            d["skin_lab"] = [round(v, 2) for v in self.skin_lab]
        return d


# ----------------------------------------------------------- detection --

def _spans(binary: np.ndarray) -> list[tuple[int, int]]:
    """Runs of non-empty positions in a 1-D occupancy vector."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, filled in enumerate(binary):
        if filled and start is None:
            start = i
        elif not filled and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(binary) - 1))
    return out


def is_silhouette(rgba: np.ndarray, mask: np.ndarray) -> bool:
    """A flat single-value figure: a form reference, not art.

    Worth detecting rather than rejecting. A silhouette carries the cleanest possible outline and
    centreline, so it is the best input the form extractor can get - it just must never be used as
    a colour or texture reference.
    """
    if not mask.any():
        return False
    # Measure the interior, not the edge. Antialiasing spreads a flat black figure across many
    # grey bins: the human gait reference scored 0.829 flatness on the full mask and would have
    # been rejected, while its eroded interior is a single value.
    interior = cv2.erode(mask.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    if int(interior.sum()) < 200:
        interior = mask
    grey = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)[interior]
    hist, _edges = np.histogram(grey, bins=16, range=(0, 256))
    flatness = float(hist.max() / max(1, hist.sum()))
    if flatness < SILHOUETTE_COVER:
        return False
    # Saturation is a supporting signal only, never a veto: a pure black figure has undefined hue
    # and the Lyn silhouette measured a median saturation of 43 despite being 100 % one grey.
    sat = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)[..., 1][interior]
    return flatness >= 0.98 or float(np.median(sat)) <= SILHOUETTE_SAT_MAX


def _axis_period(profile: np.ndarray, extent: int, max_count: int) -> tuple[int, float, int]:
    """Count of repeats along one axis, from the autocorrelation of its occupancy profile."""
    centred = profile.astype(np.float64) - profile.mean()
    correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1:]
    correlation /= (correlation[0] + 1e-9)
    lo, hi = max(4, int(extent / max_count)), int(extent / 1.5)
    if hi - lo < 3:
        return 1, 0.0, extent
    window = correlation[lo:hi]
    period = lo + int(np.argmax(window))
    return max(1, int(round(extent / period))), float(correlation[period]), period


def _row_period(mask: np.ndarray) -> tuple[int, float, int]:
    """Row count from the periodicity of the vertical occupancy profile.

    Counting row spans is off by one whenever a tail crosses into the next row (an 8-row grid
    reported 7), and counting column spans per candidate band is insensitive to the row count at
    all, because a grid's columns merge vertically regardless of where the bands are drawn - a
    5-row guess scored as well as the correct 8. Autocorrelation measures the thing that actually
    differs: a grid's profile repeats with a period of one cell, a strip's does not.
    Measured: both 8-row grids peak at period 156 of 1254 = 8 rows, correlation 0.72.
    """
    height = mask.shape[0]
    profile = mask.sum(axis=1).astype(np.float64)
    profile -= profile.mean()
    correlation = np.correlate(profile, profile, mode="full")[len(profile) - 1:]
    correlation /= (correlation[0] + 1e-9)
    lo, hi = max(4, int(height / MAX_GRID_ROWS)), int(height / 1.5)
    if hi - lo < 3:
        return 1, 0.0, height
    window = correlation[lo:hi]
    period = lo + int(np.argmax(window))
    return max(1, int(round(height / period))), float(correlation[period]), period


def detect_layout(rgba: np.ndarray) -> dict[str, Any]:
    """single | strip | grid, decided by column spans plus row periodicity and cell aspect."""
    mask = foreground_mask(rgba)
    height, width = mask.shape
    if not mask.any():
        return {"layout": "empty", "rows": 0, "cols": 0}
    col_spans = [s for s in _spans(mask.any(axis=0)) if s[1] - s[0] >= width * MIN_FRAME_SPAN]
    cols = max(1, len(col_spans))
    # Span counting under-counts whenever figures overlap horizontally: two new strips of eight
    # reported 5 and 3 because a swinging tail bridges the gap to the next figure. The column
    # profile is still periodic, so the autocorrelation count is used when it is both stronger and
    # larger than the span count.
    period_cols, col_strength, col_period = _axis_period(mask.sum(axis=0), width, MAX_STRIP_COLS)
    # Only override the span count when the spans are plainly merged - that is, when the widest
    # span is far wider than the periodic cell. A single figure has one span and a periodic-looking
    # profile (its own body), and trusting the period there turned one drawing into a 12x16 grid.
    widest = max((b - a + 1 for a, b in col_spans), default=0)
    merged = len(col_spans) >= 2 and widest > col_period * MERGED_SPAN_FACTOR
    if merged and period_cols > cols and col_strength >= COL_PERIOD_MIN_STRENGTH:
        cols = period_cols
    if cols < 2:
        return {"layout": "single", "rows": 1, "cols": 1}

    rows, strength, period = _row_period(mask)
    cell_aspect = (width / float(cols)) / (height / float(max(1, rows)))
    # A cell holds one standing figure, so its aspect is between 0.30 and 1.30. A strip read as
    # 12 rows would put the cell aspect at 2.7-4.5, which no figure has.
    grid = (rows >= GRID_ROW_MIN and strength >= ROW_PERIOD_MIN_STRENGTH
            and CELL_ASPECT_MIN <= cell_aspect <= CELL_ASPECT_MAX)
    detail = {"cols": cols, "row_period_px": period, "row_period_strength": round(strength, 3),
              "col_span_count": len(col_spans), "col_period_px": col_period,
              "col_period_strength": round(col_strength, 3), "cell_aspect": round(cell_aspect, 2)}
    if grid:
        return {"layout": "grid", "rows": rows, **detail}
    return {"layout": "strip", "rows": 1, **detail}


# ------------------------------------------------------------- ingest --

def _frame_rows(rgba: np.ndarray, layout: dict[str, Any]) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """(row, col, rgba_crop, mask_crop) for every frame the layout implies."""
    from .walk import split_grid
    from .metrics import split_sheet

    kind = layout["layout"]
    out: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    if kind == "grid":
        for r, frames in enumerate(split_grid(rgba, layout["rows"], layout["cols"])):
            for c, f in enumerate(frames):
                out.append((r, c, f.rgba, f.mask))
        return out
    if kind == "strip":
        try:
            frames = split_sheet(rgba, layout["cols"], matte=False)
        except ValueError:
            frames = []
        for c, f in enumerate(frames):
            out.append((0, c, f.rgba, f.mask))
        if out:
            return out
    mask = foreground_mask(rgba)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if crop.shape[2] == 3 or int(crop[..., 3].min()) == 255:
        crop = np.dstack([crop[..., :3], sub.astype(np.uint8) * 255])
    return [(0, 0, crop, sub)]


# A character frame is tall and narrow with a distinct head. A room plate is a wide slab of
# content. Measured on the corpus: room plates reported head-width ratios of 1.1-2.9 - physically
# impossible for a figure - and dragged the canon's head ratio to 0.327 with a 0.88 spread.
CHARACTER_MAX_ASPECT = 1.05     # width / height of the figure's own bbox
CHARACTER_MAX_HEAD_RATIO = 0.55
CHARACTER_MAX_FILL = 0.80       # a figure does not fill its bbox; a plate nearly does


def is_character(mask: np.ndarray) -> tuple[bool, str]:
    """Is this frame a figure, or a background plate that happened to be delivered here?"""
    height, width = mask.shape
    if height < 32 or width < 16:
        return False, "too_small"
    aspect = width / float(height)
    if aspect > CHARACTER_MAX_ASPECT:
        return False, f"aspect {aspect:.2f} > {CHARACTER_MAX_ASPECT}"
    if _head_width_ratio(mask) > CHARACTER_MAX_HEAD_RATIO:
        return False, "head_ratio implausible for a figure"
    if float(mask.mean()) > CHARACTER_MAX_FILL:
        return False, f"fill {float(mask.mean()):.2f} - solid slab"
    return True, ""


def _head_width_ratio(mask: np.ndarray) -> float:
    top = mask[: max(1, int(mask.shape[0] * 0.25))]
    return float(top.sum(axis=1).max() / mask.shape[0])


def load_index(root: str | Path) -> dict[str, Any]:
    path = Path(root) / INDEX_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": SCHEMA, "version": SCHEMA_VERSION, "platekit_version": __version__,
            "sources": {}, "frames": []}


def ingest(source: str | Path, root: str | Path, *, tags: Sequence[str] = (),
           force_layout: str | None = None) -> dict[str, Any]:
    """Add one sheet to the corpus. Re-ingesting the same bytes is a no-op, never a duplicate."""
    source = Path(source)
    root = Path(root)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    index = load_index(root)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    short = digest[:12]
    if short in index["sources"]:
        return {"status": "already_present", "source_hash": short,
                "frames": index["sources"][short]["frame_count"]}

    rgba = np.asarray(Image.open(source).convert("RGBA")).copy()
    mask_all = foreground_mask(rgba)
    layout = detect_layout(rgba) if force_layout is None else {
        "layout": force_layout, "rows": 1, "cols": max(1, len(
            [s for s in _spans(mask_all.any(axis=0))
             if s[1] - s[0] >= rgba.shape[1] * MIN_FRAME_SPAN]))}
    silhouette = is_silhouette(rgba, mask_all)
    alpha_native = rgba.shape[2] == 4 and int(rgba[..., 3].min()) < 255

    rows: list[FrameRow] = []
    for r, c, crop, sub in _frame_rows(rgba, layout):
        if not sub.any():
            continue
        frame_id = f"{short}_r{r}c{c}"
        rel = f"frames/{frame_id}.png"
        save_png(root / rel, crop, force=True)
        m = measure(type("F", (), {"index": c + 1, "rgba": crop, "mask": sub})())
        skin = None
        if not silhouette:
            rgb = crop[..., :3].copy()
            rgb[~sub] = 0
            classes = colour_classes(rgb, sub)
            if int(classes["skin"].sum()) > 50:
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[classes["skin"]].astype(np.float32)
                skin = [float(v) for v in lab.mean(axis=0)]
        from .walk import body_height
        character, reason = is_character(sub)
        frame_tags = list(tags) + ([] if character else ["not_character", reason])
        rows.append(FrameRow(
            frame_id=frame_id, source_hash=short, source_name=source.name,
            layout=layout["layout"], row=r, col=c, file=rel,
            width=int(sub.shape[1]), height=int(sub.shape[0]),
            body_height=int(body_height(sub)), head_width_ratio=_head_width_ratio(sub),
            skin_lab=skin, is_silhouette=silhouette, alpha_native=alpha_native,
            is_character=character, tags=frame_tags))

    index["sources"][short] = {
        "name": source.name, "sha256": digest, "size_px": [int(rgba.shape[1]), int(rgba.shape[0])],
        "layout": layout, "is_silhouette": silhouette, "alpha_native": alpha_native,
        "frame_count": len(rows), "tags": list(tags),
    }
    index["frames"].extend(r.to_dict() for r in rows)
    index["platekit_version"] = __version__
    save_json(root / INDEX_NAME, index)
    return {"status": "ingested", "source_hash": short, "layout": layout,
            "frames": len(rows), "is_silhouette": silhouette}


def ingest_all(sources: Sequence[str | Path], root: str | Path) -> dict[str, Any]:
    results = [ingest(s, root) for s in sources]
    index = load_index(root)
    return {"ingested": sum(1 for r in results if r["status"] == "ingested"),
            "already_present": sum(1 for r in results if r["status"] == "already_present"),
            "total_sources": len(index["sources"]), "total_frames": len(index["frames"]),
            "by_layout": {k: sum(1 for f in index["frames"] if f["layout"] == k)
                          for k in {f["layout"] for f in index["frames"]}},
            "silhouette_frames": sum(1 for f in index["frames"] if f["is_silhouette"]),
            "character_frames": sum(1 for f in index["frames"] if f.get("is_character", True)),
            "non_character_frames": sum(1 for f in index["frames"] if not f.get("is_character", True)),
            "results": results}


# -------------------------------------------------------------- canon --

def derive_canon(root: str | Path, *, min_body_px: int = 200) -> dict[str, Any]:
    """One standard from the medians of the whole corpus.

    Frames smaller than ``min_body_px`` are excluded from the colour and proportion medians: a
    150 px grid cell cannot measure a head ratio reliably, and including it drags the canon.
    Silhouette frames are excluded from the colour median only - their outline still counts.
    """
    index = load_index(root)
    frames = index["frames"]
    usable = [f for f in frames
              if f["body_height"] >= min_body_px and f.get("is_character", True)]
    heads = [f["head_width_ratio"] for f in usable]
    skins = [f["skin_lab"] for f in usable if f.get("skin_lab") and not f["is_silhouette"]]
    if not heads:
        raise ValueError(f"body_height >= {min_body_px}px 인 프레임이 없습니다.")
    head_median = float(np.median(heads))
    canon = {
        "schema": "platekit.character_canon", "version": 2, "platekit_version": __version__,
        "derived_from": {"sources": len(index["sources"]), "frames": len(frames),
                         "frames_used": len(usable), "min_body_px": min_body_px},
        "head_width_ratio": round(head_median, 4),
        "head_width_ratio_spread": round(float(np.percentile(heads, 90) - np.percentile(heads, 10)), 4),
        "skin_lab": [round(float(v), 2) for v in np.median(np.stack(skins), axis=0)] if skins else None,
        "skin_frames": len(skins),
        "height_px": 460,
        "note": ("전 코퍼스의 중앙값에서 유도했다. 파일을 골라 정하지 않으므로 새 자료가 들어와도 "
                 "기준이 크게 흔들리지 않는다."),
    }
    save_json(Path(root) / "canon.json", canon)
    return canon


def frame_deviation(row: dict[str, Any], canon: dict[str, Any]) -> dict[str, Any]:
    """How far one corpus frame sits from the canon, and what would fix it."""
    head_delta = row["head_width_ratio"] - canon["head_width_ratio"]
    skin_de = None
    if row.get("skin_lab") and canon.get("skin_lab"):
        skin_de = float(np.linalg.norm(np.array(row["skin_lab"]) - np.array(canon["skin_lab"])))
    return {
        "frame_id": row["frame_id"],
        "head_delta": round(head_delta, 4),
        "head_scale_fix": round(canon["head_width_ratio"] / max(1e-6, row["head_width_ratio"]), 4),
        "skin_dE": None if skin_de is None else round(skin_de, 2),
        "body_scale_fix": round(canon["height_px"] / max(1.0, float(row["body_height"])), 4),
        "usable_for_colour": not row["is_silhouette"],
    }


def conform_frame(rgba: np.ndarray, mask: np.ndarray, canon: dict[str, Any], *,
                  mirror: bool = False, cell: tuple[int, int] | None = None,
                  pivot: tuple[int, int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Bring one frame to the canon by moving and recolouring pixels only.

    Four operations, in this order, all reversible and all recorded:
      mirror -> uniform body scale -> conservative skin colour shift -> crop/place on the cell
    Nothing is redrawn, so a frame that differs in *proportion between its own parts* (a head too
    large for its body) is not fixed here; that needs the part rig.
    """
    from .walk import body_height, ground_pivot

    log: dict[str, Any] = {"mirror": bool(mirror)}
    if mirror:
        rgba = rgba[:, ::-1].copy()
        mask = mask[:, ::-1].copy()

    scale = float(canon["height_px"]) / max(1.0, float(body_height(mask)))
    log["scale"] = round(scale, 4)
    gx, gy, _c = ground_pivot(mask)
    if abs(scale - 1.0) >= 0.02:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        new_size = (max(1, int(round(rgba.shape[1] * scale))), max(1, int(round(rgba.shape[0] * scale))))
        rgba = cv2.resize(rgba, new_size, interpolation=interp)
        mask = cv2.resize(mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST) > 0
        gx, gy = gx * scale, gy * scale

    if canon.get("skin_lab"):
        rgb = rgba[..., :3].copy()
        rgb[~mask] = 0
        classes = colour_classes(rgb, mask)
        if int(classes["skin"].sum()) > 50:
            lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
            current = lab[classes["skin"]].mean(axis=0)
            delta = np.clip(np.array(canon["skin_lab"], np.float32) - current, -10.0, 10.0)
            log["skin_shift_lab"] = [round(float(v), 2) for v in delta]
            lab[mask] = np.clip(lab[mask] + delta, 0, 255)
            rgba[..., :3] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    cell_w, cell_h = cell if cell else (int(rgba.shape[1] + 32), int(rgba.shape[0] + 32))
    px, py = pivot if pivot else (cell_w // 2, cell_h - 16)
    canvas = np.zeros((cell_h, cell_w, 4), np.uint8)
    ox, oy = int(round(px - gx)), int(round(py - (mask.shape[0] - 1)))
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(cell_w, ox + rgba.shape[1]), min(cell_h, oy + rgba.shape[0])
    log["clipped"] = bool(x0 > ox or y0 > oy or x1 < ox + rgba.shape[1] or y1 < oy + rgba.shape[0])
    if x1 > x0 and y1 > y0:
        piece = rgba[y0 - oy:y1 - oy, x0 - ox:x1 - ox].copy()
        sub = mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        piece[..., 3] = np.where(sub, piece[..., 3], 0)
        canvas[y0:y1, x0:x1] = piece
    log["cell"] = [cell_w, cell_h]
    log["pivot"] = [px, py]
    return canvas, log
