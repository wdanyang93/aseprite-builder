"""Corpus labelling. Turns 356 anonymous frames into (direction, action) addressable frames.

Without labels the corpus is exactly what the review called out: a parts list whose search results
cannot be used. "Best frame for direction E, action walk" needs someone to have said which frames
are E and which are walking.

Three signals, all measured, none guessed:

* **action** from colour classes. A pickaxe is dark low-saturation metal plus a brown wooden
  handle; either it is in the frame or it is not. Walk versus idle then comes from how much the
  silhouette changes across the sheet.
* **direction (symmetric views)** from self-mirror silhouette overlap. A front or back view is
  left-right symmetric and a side or diagonal view is not. Measured on the real 8-row grid: the S
  and N rows scored 0.68 and 0.71 while every other row sat at 0.26-0.33.
* **direction (side)** by comparing against a reference of known facing. Rows that prefer the
  mirrored reference face the other way.

What measurement cannot do is separate E from SE, or N from NE: the two side sheets in the corpus
score 0.489 silhouette IoU against each other - clearly different poses, no basis for naming which
is the pure profile. Those come back as ``needs_human`` with the candidates listed, and a single
line in an overrides file settles a whole sheet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from .. import __version__
from ..io import save_json
from .corpus import INDEX_NAME, load_index
from .metrics import colour_classes

DIRECTION_ORDER = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
SYMMETRIC_MIN = 0.50          # self-mirror overlap at or above this = a front or back view
SIDE_MARGIN = 0.015           # reference delta beyond this decides east vs west
# Only the metal head discriminates. The wooden-handle class (brown, mid value) also matches
# shadowed skin and hair: measured 0.068-0.072 of the figure on walk sheets that contain no tool at
# all, against 0.060 on a real dig sheet - useless. The grey metal class separates cleanly:
# walk 0.0018-0.0069 versus dig 0.104.
TOOL_MIN_RATIO = 0.02         # grey-metal pixels as a share of the figure
IDLE_CHANGE_MAX = 0.06        # mean silhouette change across a sheet below this = idle
COMPARE_HEIGHT = 256


def _load(root: Path, rel: str) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(Image.open(root / rel).convert("RGBA"))
    return rgba, rgba[..., 3] > 8


def _resize(mask: np.ndarray) -> np.ndarray:
    factor = COMPARE_HEIGHT / max(1, mask.shape[0])
    return cv2.resize(mask.astype(np.uint8),
                      (max(1, int(mask.shape[1] * factor)), COMPARE_HEIGHT),
                      interpolation=cv2.INTER_NEAREST) > 0


def _overlap(a: np.ndarray, b: np.ndarray, mirror: bool = False) -> float:
    if mirror:
        a = a[:, ::-1]
    width = max(a.shape[1], b.shape[1])
    pa = np.zeros((COMPARE_HEIGHT, width), bool)
    pb = np.zeros((COMPARE_HEIGHT, width), bool)
    pa[:, (width - a.shape[1]) // 2:(width - a.shape[1]) // 2 + a.shape[1]] = a
    pb[:, (width - b.shape[1]) // 2:(width - b.shape[1]) // 2 + b.shape[1]] = b
    union = int((pa | pb).sum())
    return float((pa & pb).sum() / max(1, union))


# -------------------------------------------------------------- action --

def detect_action(rgba: np.ndarray, mask: np.ndarray) -> tuple[str, dict[str, Any]]:
    """dig when a tool is present, otherwise walk/idle (decided per sheet, not per frame)."""
    rgb = rgba[..., :3].copy()
    rgb[~mask] = 0
    classes = colour_classes(rgb, mask)
    metal = float(classes["tool"].sum() / max(1, mask.sum()))
    handle = float(classes["handle"].sum() / max(1, mask.sum()))
    return ("dig" if metal >= TOOL_MIN_RATIO else "locomotion"), {
        "metal_ratio": round(metal, 5), "handle_ratio": round(handle, 5),
        "note": "handle은 그림자 진 피부와 겹쳐 판정에 쓰지 않는다"}


# ----------------------------------------------------------- direction --

def label_rows(masks_by_row: dict[int, list[np.ndarray]],
               reference: Sequence[np.ndarray] | None = None,
               reference_facing: str = "right") -> dict[int, dict[str, Any]]:
    """Per-row view and side, with an explicit ``needs_human`` when the two cannot be told apart."""
    scaled = {r: [_resize(m) for m in ms] for r, ms in masks_by_row.items()}
    reference_scaled = [_resize(m) for m in reference] if reference else None
    out: dict[int, dict[str, Any]] = {}
    for row, masks in scaled.items():
        symmetry = float(np.mean([_overlap(m, m, mirror=True) for m in masks]))
        view = "symmetric" if symmetry >= SYMMETRIC_MIN else "oblique"
        side, delta = None, None
        if reference_scaled:
            plain = float(np.mean([max(_overlap(m, rm) for rm in reference_scaled) for m in masks]))
            mirrored = float(np.mean([max(_overlap(m, rm, mirror=True) for rm in reference_scaled)
                                      for m in masks]))
            delta = plain - mirrored
            if abs(delta) > SIDE_MARGIN:
                toward = "east" if delta > 0 else "west"
                side = toward if reference_facing == "right" else (
                    "west" if toward == "east" else "east")
        out[row] = {"self_symmetry": round(symmetry, 3), "view": view,
                    "reference_delta": None if delta is None else round(delta, 3), "side": side}
    return out


def assign_directions(row_info: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Turn per-row view/side into named directions where that is defensible.

    A full eight-row sheet in canon order is fully determined: two symmetric rows, three west rows
    between them and three east rows after. Anything else gets a candidate list instead of a name,
    because a two-row sheet of side views carries no information about which is E and which is SE.
    """
    rows = sorted(row_info)
    symmetric = [r for r in rows if row_info[r]["view"] == "symmetric"]
    out = {r: dict(row_info[r]) for r in rows}
    if len(rows) == 8 and len(symmetric) == 2:
        first, second = symmetric
        # S is the symmetric row whose neighbours run west; N is the other one.
        west_between = sum(1 for r in rows if first < r < second and row_info[r]["side"] == "west")
        if west_between >= 2:
            order = DIRECTION_ORDER
        else:
            order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        rotated = order[-first:] + order[:-first] if first else order
        for r, name in zip(rows, rotated):
            out[r]["direction"] = name
            out[r]["direction_source"] = "measured_eight_row"
        return out
    for r in rows:
        info = out[r]
        if info["view"] == "symmetric":
            info["direction_candidates"] = ["S", "N"]
        elif info["side"] == "east":
            info["direction_candidates"] = ["E", "SE", "NE"]
        elif info["side"] == "west":
            info["direction_candidates"] = ["W", "SW", "NW"]
        else:
            info["direction_candidates"] = list(DIRECTION_ORDER)
        info["needs_human"] = True
    return out


# -------------------------------------------------------------- driver --

def label_corpus(root: str | Path, *, reference_frames: Sequence[str] | None = None,
                 overrides: str | Path | None = None) -> dict[str, Any]:
    """Label every source in the corpus and write the result back into the index.

    ``overrides`` is a JSON file mapping a source hash to the labels a human settled, e.g.::

        {"5407028d77e1": {"rows": {"0": "E", "1": "SE"}, "action": "locomotion"}}

    Overrides always win and are recorded as ``direction_source: "human"``.
    """
    root = Path(root)
    index = load_index(root)
    manual: dict[str, Any] = json.loads(Path(overrides).read_text(encoding="utf-8")) if overrides else {}

    reference: list[np.ndarray] = []
    for rel in (reference_frames or []):
        reference.append(_load(root, rel)[1])

    by_source: dict[str, list[dict[str, Any]]] = {}
    for frame in index["frames"]:
        if frame.get("is_character", True):
            by_source.setdefault(frame["source_hash"], []).append(frame)

    report: list[dict[str, Any]] = []
    for source_hash, frames in by_source.items():
        masks_by_row: dict[int, list[np.ndarray]] = {}
        rgba_first: np.ndarray | None = None
        mask_first: np.ndarray | None = None
        for frame in frames:
            rgba, mask = _load(root, frame["file"])
            masks_by_row.setdefault(frame["row"], []).append(mask)
            if rgba_first is None:
                rgba_first, mask_first = rgba, mask

        action, action_info = detect_action(rgba_first, mask_first)
        # locomotion splits into walk and idle by how much the silhouette actually moves
        change = 0.0
        flat = [m for ms in masks_by_row.values() for m in ms]
        if action == "locomotion" and len(flat) >= 2:
            scaled = [_resize(m) for m in flat]
            change = float(np.mean([1.0 - _overlap(scaled[i], scaled[i - 1])
                                    for i in range(1, len(scaled))]))
            action = "idle" if change <= IDLE_CHANGE_MAX else "walk"

        row_info = assign_directions(label_rows(masks_by_row, reference or None))
        override = manual.get(source_hash, {})
        if "action" in override:
            action = str(override["action"])
        needs_human = False
        for row, info in row_info.items():
            name = (override.get("rows") or {}).get(str(row))
            if name:
                info["direction"] = str(name)
                info["direction_source"] = "human"
                info.pop("direction_candidates", None)
                info.pop("needs_human", None)
            elif "direction" not in info:
                needs_human = True

        for frame in frames:
            info = row_info.get(frame["row"], {})
            frame["action"] = action
            frame["silhouette_change"] = round(change, 4)
            if "direction" in info:
                frame["direction"] = info["direction"]
                frame["direction_source"] = info["direction_source"]
            else:
                frame["direction_candidates"] = info.get("direction_candidates", [])
                frame["needs_human"] = True

        index["sources"][source_hash]["labels"] = {
            "action": action, "action_signals": action_info,
            "silhouette_change": round(change, 4),
            "rows": {str(k): v for k, v in row_info.items()},
            "needs_human": needs_human,
        }
        report.append({"source_hash": source_hash, "name": index["sources"][source_hash]["name"],
                       "frames": len(frames), "action": action, "needs_human": needs_human,
                       "rows": {str(k): v.get("direction") or v.get("direction_candidates")
                                for k, v in row_info.items()}})

    index["platekit_version"] = __version__
    save_json(root / INDEX_NAME, index)
    labelled = sum(1 for f in index["frames"] if f.get("direction"))
    summary = {
        "schema": "platekit.corpus_labels", "version": 1, "platekit_version": __version__,
        "sources_labelled": len(by_source),
        "frames_with_direction": labelled,
        "frames_needing_human": sum(1 for f in index["frames"] if f.get("needs_human")),
        "by_action": {a: sum(1 for f in index["frames"] if f.get("action") == a)
                      for a in {f.get("action") for f in index["frames"] if f.get("action")}},
        "by_direction": {d: sum(1 for f in index["frames"] if f.get("direction") == d)
                         for d in DIRECTION_ORDER},
        "sources": sorted(report, key=lambda r: (-r["frames"], r["name"])),
    }
    save_json(root / "labels.json", summary)
    return summary


def select_best(root: str | Path, canon: dict[str, Any]) -> dict[str, Any]:
    """Rank, never exclude: every (direction, action) cell keeps its full ranked list.

    Ranking is deviation from canon, with native alpha breaking ties because a keyed frame has
    already lost the white clothing to the background once.
    """
    root = Path(root)
    index = load_index(root)
    cells: dict[str, list[dict[str, Any]]] = {}
    for frame in index["frames"]:
        if not frame.get("is_character", True) or not frame.get("direction"):
            continue
        key = f"{frame.get('action', 'unknown')}/{frame['direction']}"
        head_delta = abs(frame["head_width_ratio"] - canon["head_width_ratio"])
        skin_de = None
        if frame.get("skin_lab") and canon.get("skin_lab"):
            skin_de = float(np.linalg.norm(np.array(frame["skin_lab"]) - np.array(canon["skin_lab"])))
        score = head_delta / 0.03 + (skin_de or 0.0) / 6.0 + (0.0 if frame["alpha_native"] else 0.25)
        cells.setdefault(key, []).append({
            "frame_id": frame["frame_id"], "source_name": frame["source_name"],
            "col": frame["col"], "body_height": frame["body_height"],
            "head_delta": round(frame["head_width_ratio"] - canon["head_width_ratio"], 4),
            "skin_dE": None if skin_de is None else round(skin_de, 2),
            "alpha_native": frame["alpha_native"], "score": round(score, 3),
        })
    for key in cells:
        cells[key].sort(key=lambda r: r["score"])
    summary = {
        "schema": "platekit.corpus_selection", "version": 1, "platekit_version": __version__,
        "canon": {k: canon[k] for k in ("head_width_ratio", "skin_lab", "height_px") if k in canon},
        "cells": {k: {"count": len(v), "best": v[0], "ranked": v} for k, v in sorted(cells.items())},
        "note": "탈락이 아니라 순위다. 모든 프레임이 ranked 에 남아 나중에 교체할 수 있다.",
    }
    save_json(root / "selection.json", summary)
    return summary
