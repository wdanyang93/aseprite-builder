"""Leg identity from connectivity, not shading alone.

Shading is the strongest single cue for which limb is nearer - the art shades the far leg about 25
L* darker - and it was enough on seven of eight frames of the reference sheet. On frame 8 it is
wrong, and wrong in a specific way the owner identified: the thigh is shaded correctly and the
calf and foot are shaded as if they belonged to the other leg. Measured on that frame:

    thigh  y 230-301   near centre x 0.48   far centre x 0.61
    calf   y 301-373   near centre x 0.67   far centre x 0.45    <- sides swapped
    foot   y 373-419   near centre x 0.82   far centre x 0.25    <- swapped

A leg cannot cross to the other side of the body halfway down. So the shading label is treated as
evidence rather than as the answer, and the answer comes from connectivity: a leg is a chain of
pixels continuous from hip to toe, and each connected piece inherits the identity of the piece it
touches above it. Where connectivity and shading disagree, connectivity wins and the disagreement
is reported - it is an error in the drawing, not in the measurement.
"""
from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

SHADE_BINS = 16
SHADE_RANGE = (150.0, 255.0)
SHADE_MIN_PEAK_SHARE = 0.04
SLICE_COUNT = 12               # horizontal bands from hip to toe
MIN_PIECE_PX = 60
CROSS_TOLERANCE = 0.06         # a leg's x centre may wander this much between bands


def shade_split(lightness: np.ndarray, region: np.ndarray) -> dict[str, Any] | None:
    """Bimodal lightness split of a leg region. Returns the two masks and the threshold."""
    values = lightness[region]
    if values.size < 400:
        return None
    hist, edges = np.histogram(values, bins=SHADE_BINS, range=SHADE_RANGE)
    peaks = [i for i in range(1, SHADE_BINS - 1)
             if hist[i] > hist[i - 1] and hist[i] >= hist[i + 1]
             and hist[i] > values.size * SHADE_MIN_PEAK_SHARE]
    if len(peaks) < 2:
        return None
    peaks.sort(key=lambda i: -hist[i])
    low, high = sorted(peaks[:2])
    valley = low + int(np.argmin(hist[low:high + 1]))
    threshold = float(edges[valley])
    return {"threshold": threshold,
            "bright": region & (lightness >= threshold),
            "dark": region & (lightness < threshold)}


def _band_edges(region: np.ndarray, count: int) -> list[int]:
    ys = np.nonzero(region.any(axis=1))[0]
    if ys.size == 0:
        return []
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return [int(round(top + (bottom - top) * i / count)) for i in range(count + 1)]


def trace_legs(region: np.ndarray, lightness: np.ndarray) -> dict[str, Any]:
    """Two leg masks, assigned band by band from the hip down.

    Within each horizontal band the region splits into left and right pieces by x. A piece belongs
    to whichever leg occupied the nearest x in the band above, so identity propagates down the limb
    and cannot jump sides. The shading label is recorded per band for comparison.
    """
    height, width = region.shape
    edges = _band_edges(region, SLICE_COUNT)
    if len(edges) < 3:
        return {"legs": None, "bands": [], "flags": ["region too small"]}

    shade = shade_split(lightness, region)
    leg_a = np.zeros_like(region)          # the leg that starts on the left at the hip
    leg_b = np.zeros_like(region)
    last: dict[str, float | None] = {"a": None, "b": None}
    bands: list[dict[str, Any]] = []

    for i in range(len(edges) - 1):
        y0, y1 = edges[i], edges[i + 1]
        band = np.zeros_like(region)
        band[y0:y1] = region[y0:y1]
        if not band.any():
            continue
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
        pieces = [{"label": j, "area": int(stats[j, cv2.CC_STAT_AREA]),
                   "cx": float(centroids[j][0])}
                  for j in range(1, count) if stats[j, cv2.CC_STAT_AREA] >= MIN_PIECE_PX]
        if not pieces:
            continue
        pieces.sort(key=lambda p: p["cx"])

        assignment: dict[str, list[int]] = {"a": [], "b": []}
        if last["a"] is None and last["b"] is None:
            # first band: leftmost piece seeds leg A, rightmost seeds leg B
            if len(pieces) == 1:
                assignment["a"].append(pieces[0]["label"])
                assignment["b"].append(pieces[0]["label"])
            else:
                assignment["a"].append(pieces[0]["label"])
                assignment["b"].append(pieces[-1]["label"])
                for p in pieces[1:-1]:
                    key = "a" if abs(p["cx"] - pieces[0]["cx"]) < abs(p["cx"] - pieces[-1]["cx"]) else "b"
                    assignment[key].append(p["label"])
        else:
            for p in pieces:
                da = abs(p["cx"] - last["a"]) if last["a"] is not None else 1e9
                db = abs(p["cx"] - last["b"]) if last["b"] is not None else 1e9
                assignment["a" if da <= db else "b"].append(p["label"])

        for key, target in (("a", leg_a), ("b", leg_b)):
            for label in assignment[key]:
                target[y0:y1] |= (labels == label)[y0:y1]
            sel = np.zeros_like(region)
            for label in assignment[key]:
                sel |= labels == label
            if sel.any():
                last[key] = float(np.nonzero(sel)[1].mean())

        record = {"band": i, "y": [y0, y1], "pieces": len(pieces),
                  "cx_a": None if last["a"] is None else round(last["a"] / width, 3),
                  "cx_b": None if last["b"] is None else round(last["b"] / width, 3)}
        if shade is not None:
            for key, target in (("a", leg_a), ("b", leg_b)):
                piece = target.copy()
                piece[:y0] = False
                piece[y1:] = False
                if piece.any():
                    bright = int((piece & shade["bright"]).sum())
                    dark = int((piece & shade["dark"]).sum())
                    record[f"shade_{key}"] = "near" if bright >= dark else "far"
        bands.append(record)

    return {"leg_a": leg_a, "leg_b": leg_b, "bands": bands,
            "shade_threshold": None if shade is None else round(shade["threshold"], 1)}


def resolve_identity(trace: dict[str, Any]) -> dict[str, Any]:
    """Decide which traced leg is near, by majority vote of the shading across bands.

    A single band's shading can be wrong; the whole limb's cannot all be. Frame 8's thigh bands
    said one thing and its calf bands the opposite, which is exactly the disagreement this reports.
    """
    bands = trace.get("bands") or []
    votes = {"a": {"near": 0, "far": 0}, "b": {"near": 0, "far": 0}}
    for record in bands:
        for key in ("a", "b"):
            label = record.get(f"shade_{key}")
            if label:
                votes[key][label] += 1
    total_a = votes["a"]["near"] + votes["a"]["far"]
    total_b = votes["b"]["near"] + votes["b"]["far"]
    consistency_a = max(votes["a"].values()) / total_a if total_a else 0.0
    consistency_b = max(votes["b"].values()) / total_b if total_b else 0.0
    a_is_near = votes["a"]["near"] >= votes["a"]["far"]
    if votes["a"]["near"] == votes["a"]["far"] and total_b:
        a_is_near = votes["b"]["far"] >= votes["b"]["near"]
    flags: list[str] = []
    if min(consistency_a, consistency_b) < 0.75:
        flags.append(f"SHADING_INCONSISTENT (a {consistency_a:.0%}, b {consistency_b:.0%})")
    if a_is_near == (votes["b"]["near"] > votes["b"]["far"]):
        flags.append("BOTH_LEGS_SAME_LABEL")
    return {"near": "a" if a_is_near else "b", "far": "b" if a_is_near else "a",
            "votes": votes, "consistency": [round(consistency_a, 3), round(consistency_b, 3)],
            "flags": flags}


def leg_identity(region: np.ndarray, rgba: np.ndarray) -> dict[str, Any]:
    """Full pass: trace by connectivity, label by shading majority, report disagreement."""
    lightness = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    trace = trace_legs(region, lightness)
    if trace.get("leg_a") is None:
        return {"near_mask": None, "far_mask": None, "flags": trace.get("flags", [])}
    identity = resolve_identity(trace)
    near = trace["leg_a"] if identity["near"] == "a" else trace["leg_b"]
    far = trace["leg_b"] if identity["near"] == "a" else trace["leg_a"]
    height, width = region.shape

    def toe(mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        sel = ys >= ys.max() - 10
        return float(xs[sel].mean()) / width if sel.any() else None

    return {"near_mask": near, "far_mask": far,
            "near_toe": toe(near), "far_toe": toe(far),
            "lead": "near" if (toe(near) or 0) > (toe(far) or 0) else "far",
            "shade_threshold": trace.get("shade_threshold"),
            "bands": trace["bands"], **identity}
