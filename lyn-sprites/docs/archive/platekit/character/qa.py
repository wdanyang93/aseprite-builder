"""Character QA and the (deliberately narrow) fix.

Detects:
  ORIENTATION_FLIP        a frame faces the other way from the rest of the animation
  HEIGHT_DEVIATION        frame height off the canon by more than the tolerance
  AREA_DEVIATION          foreground area off the canon
  GROUND_DRIFT            ground pivot x wanders (feet slide)
  PALETTE_DRIFT           mean Lab off the canon
  LOOP_DISCONTINUITY      8->1 (or N->1) jumps more than the in-cycle median

Fixes (only these, only when asked, always logged):
  mirror_x on named frames  -> orientation
  uniform scale + translate -> height / ground   (normalise)

Never: non-uniform warp, reshaping, redraw, dropping frames, reordering frames.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .. import __version__
from ..io import save_json, save_png, sha256_file
from .metrics import Frame, FrameMetrics, foreground_mask, measure, silhouette_iou, split_sheet

# Thresholds. Calibrated on the three real v1.11 dig sheets; recorded in VALIDATION_V1_11.md.
# On a dig cycle bbox height swings +33%/-28% legitimately (raised pick, crouch) and full-area palette
# shifts by dE 14-22 when dirt enters the frame, so neither is a scale/colour invariant for ACTION
# animations. Core (eroded) area stays within about +-12% and skin-only Lab within dE 0-4.4.
HEIGHT_WARN = 0.06                  # locomotion animations only
HEIGHT_FAIL = 0.15
# Action animations (dig/swing): the figure crouches and rears, so bbox height is meaningless and
# core area is the scale invariant (measured spread 6-11 % across the three real dig cycles).
# Locomotion (walk/idle): height is the invariant (measured 0.9-2.6 % across five real walk
# cycles) and core area legitimately swings ~20 % as the legs separate, so it is warn-only there.
CORE_AREA_WARN = 0.15
CORE_AREA_FAIL = 0.30
CORE_AREA_WARN_LOCOMOTION = 0.25
CORE_AREA_FAIL_LOCOMOTION = 0.45
PALETTE_WARN_DE = 6.0               # skin class only
LOOP_OUTLIER_FACTOR = 2.0
MIRROR_MARGIN = 0.10                # silhouette signal only counts if this much stronger
FACING_SUPPORT_MIN = 0.5            # intended facing needs >= 50 % of frames voting, else frontal


# ------------------------------------------------------------- loading --

def load_animation(sheet_path: str | Path, count: int) -> list[Frame]:
    from PIL import Image
    rgba = np.asarray(Image.open(sheet_path).convert("RGBA")).copy()
    return split_sheet(rgba, count)


# ------------------------------------------------------- orientation --

def orientation_report(frames: Sequence[Frame], metrics: Sequence[FrameMetrics],
                       declared_facing: str = "auto") -> dict[str, Any]:
    """Which frames face the other way?

    Signal 1 (primary): colour-class facing (eyes vs head centre), with the tool side as a
    tie-breaker. Signal 2 (supporting): mirrored-vs-neighbour silhouette IoU.
    ``declared_facing``: "L" / "R" = the animation is supposed to face that way;
    "auto" = the majority facing is taken as intended.
    """
    facings = [m.facing for m in metrics]
    tool_sides = [m.tool_side for m in metrics]
    votes = {"L": 0, "R": 0}
    for f in facings:
        if f in votes:
            votes[f] += 1
    if declared_facing in ("L", "R"):
        intended = declared_facing
        intended_source = "declared"
    elif max(votes.values()) < FACING_SUPPORT_MIN * len(metrics):
        # Too few frames show a clear side: a frontal/back animation. Nothing to flip.
        intended = "-"
        intended_source = "undetermined_frontal"
    else:
        intended = "L" if votes["L"] >= votes["R"] else "R"
        intended_source = "majority"

    per_frame: list[dict[str, Any]] = []
    flips: list[int] = []
    for i, (frame, m) in enumerate(zip(frames, metrics)):
        prev = frames[i - 1]
        s_plain = silhouette_iou(frame.mask, prev.mask)
        s_mirror = silhouette_iou(frame.mask, prev.mask, mirror_a=True)
        silhouette_vote = "mirror" if s_mirror > s_plain + MIRROR_MARGIN else ("plain" if s_plain > s_mirror + MIRROR_MARGIN else "-")
        # Eyes decide. The tool side is reported but never used to flag: a pickaxe legitimately
        # crosses from one side to the other during a swing.
        colour_vote = m.facing
        opposite = intended != "-" and colour_vote in ("L", "R") and colour_vote != intended
        signals = []
        if opposite:
            signals.append("eyes")
        if silhouette_vote == "mirror":
            signals.append("silhouette_mirror")
        tool_agrees = m.tool_side in ("L", "R") and m.tool_side == colour_vote
        if opposite and tool_agrees:
            signals.append("tool_side_agrees")
        confidence = ("high" if opposite and (len(signals) >= 2) else
                      "medium" if opposite else ("low" if signals else "none"))
        flagged = opposite
        if flagged:
            flips.append(frame.index)
        per_frame.append({
            "index": frame.index, "facing": m.facing, "tool_side": m.tool_side,
            "eye_dx": m.eye_dx, "tool_dx": m.tool_dx,
            "iou_prev": round(s_plain, 3), "iou_mirror_prev": round(s_mirror, 3),
            "orientation_guess": "opposite" if opposite else ("same" if colour_vote == intended else "undetermined"),
            "signals": signals, "confidence": confidence,
            "qa_flag": "ORIENTATION_FLIP_CANDIDATE" if flagged else None,
        })

    # transitions where the facing sign changes: this is what a human sees as "the animation jumps"
    transitions = []
    for i in range(len(facings)):
        a, b = facings[i - 1], facings[i]
        if a in ("L", "R") and b in ("L", "R") and a != b:
            transitions.append(f"{frames[i - 1].index}->{frames[i].index}")

    return {
        "intended_facing": intended, "intended_source": intended_source,
        "votes": votes, "flip_candidates": flips, "facing_transitions": transitions,
        "frames": per_frame,
        "fix_options": ({"mirror_frames": flips, "result_facing": intended,
                         "alternative": {"mirror_frames": [f.index for f in frames if f.index not in flips],
                                         "result_facing": ("L" if intended == "R" else "R")}}
                        if flips else None),
    }


# ------------------------------------------------------ consistency --

def consistency_report(metrics: Sequence[FrameMetrics], canon_index: int,
                       animation_type: str = "action") -> dict[str, Any]:
    """animation_type: "locomotion" (walk/idle: bbox height is meaningful) or "action" (dig/swing:
    only core area and skin palette are trusted; height/area are reported as pose-dependent)."""
    canon = next(m for m in metrics if m.index == canon_index)
    out = []
    worst = "pass"
    for m in metrics:
        flags = []
        h_dev = (m.height - canon.height) / canon.height
        a_dev = (m.area - canon.area) / canon.area
        c_dev = (m.core_area - canon.core_area) / max(1, canon.core_area)
        g_drift = abs(m.ground_x / m.width - canon.ground_x / canon.width)
        skin_missing = canon.skin_lab is not None and m.skin_lab is None
        if m.skin_lab is not None and canon.skin_lab is not None:
            de = float(np.linalg.norm(np.array(m.skin_lab) - np.array(canon.skin_lab)))
        elif skin_missing:
            # The canon has readable skin and this frame does not: the colour moved out of the
            # skin range entirely, which is a stronger drift signal than any dE.
            de = float(np.linalg.norm(np.array(m.mean_lab) - np.array(canon.mean_lab)))
        else:
            de = 0.0
        fail = False
        if animation_type == "locomotion":
            if abs(h_dev) >= HEIGHT_FAIL:
                flags.append(f"HEIGHT_DEVIATION {h_dev:+.1%}"); fail = True
            elif abs(h_dev) >= HEIGHT_WARN:
                flags.append(f"height {h_dev:+.1%}")
        core_fail = CORE_AREA_FAIL_LOCOMOTION if animation_type == "locomotion" else CORE_AREA_FAIL
        core_warn = CORE_AREA_WARN_LOCOMOTION if animation_type == "locomotion" else CORE_AREA_WARN
        if abs(c_dev) >= core_fail:
            flags.append(f"SCALE_DEVIATION core {c_dev:+.1%}"); fail = True
        elif abs(c_dev) >= core_warn:
            flags.append(f"core_area {c_dev:+.1%}")
        if skin_missing:
            flags.append("PALETTE_DRIFT skin class not found (colour left the skin range)")
        elif de >= PALETTE_WARN_DE:
            flags.append(f"PALETTE_DRIFT skin dE {de:.1f}")
        status = "fail" if fail else ("warn" if flags else "pass")
        if status == "fail" or (status == "warn" and worst == "pass"):
            worst = status
        out.append({"index": m.index, "height_dev": round(h_dev, 4), "area_dev": round(a_dev, 4),
                    "core_area_dev": round(c_dev, 4), "ground_x_norm_drift": round(g_drift, 4),
                    "skin_palette_dE": round(de, 2), "status": status, "flags": flags,
                    "pose_dependent_note": "height/area vary with pose in action animations" if animation_type == "action" else None})
    return {"canon_index": canon_index, "animation_type": animation_type, "status": worst, "frames": out}


def continuity_report(frames: Sequence[Frame], metrics: Sequence[FrameMetrics], loop: bool = True) -> dict[str, Any]:
    pairs = []
    n = len(frames)
    rng = range(n) if loop else range(1, n)
    for i in rng:
        a, b = frames[i - 1], frames[i]
        ma, mb = metrics[i - 1], metrics[i]
        pairs.append({
            "pair": f"{a.index}->{b.index}",
            "silhouette_change": round(1.0 - silhouette_iou(b.mask, a.mask), 3),
            "height_change": round(abs(mb.height - ma.height) / max(1, ma.height), 4),
            "ground_x_change": round(abs(mb.ground_x / mb.width - ma.ground_x / ma.width), 4),
            "is_loop_closure": bool(loop and i == 0),
        })
    changes = np.array([p["silhouette_change"] for p in pairs])
    median = float(np.median(changes)) if changes.size else 0.0
    for p in pairs:
        p["outlier"] = bool(p["silhouette_change"] > LOOP_OUTLIER_FACTOR * median + 0.05)
        p["flag"] = ("LOOP_DISCONTINUITY" if p["outlier"] and p["is_loop_closure"]
                     else ("CONTINUITY_OUTLIER" if p["outlier"] else None))
    return {"median_silhouette_change": round(median, 3), "pairs": pairs}


# ---------------------------------------------------------------- fix --

def apply_mirror_fix(frames: Sequence[Frame], mirror_indices: Sequence[int]) -> list[Frame]:
    """Horizontally mirror the named frames. Whole-frame flip only; recorded as a derived transform."""
    out = []
    for f in frames:
        if f.index in mirror_indices:
            out.append(Frame(index=f.index, rgba=f.rgba[:, ::-1].copy(), mask=f.mask[:, ::-1].copy(),
                             sheet_x0=f.sheet_x0, sheet_bbox=list(f.sheet_bbox)))
        else:
            out.append(f)
    return out


def normalise_frames(frames: Sequence[Frame], metrics: Sequence[FrameMetrics], canon_index: int,
                     animation_type: str = "action", cell: tuple[int, int] | None = None,
                     pivot: tuple[int, int] | None = None) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Place every frame on one canvas with the ground pivot at the bottom centre.

    Scale policy:
      locomotion  bbox height -> canon height (the figure stays upright, height is meaningful)
      action      core-area scale, clamped to +-15 %, and no scaling at all inside +-5 %.
                  Bbox-height normalisation is WRONG here: a crouched strike frame was being
                  enlarged 1.37x on the real dig_E sheet, as if the character had shrunk.
    The canvas is sized from the union of the placed frames, so nothing is ever clipped.
    """
    canon = next(m for m in metrics if m.index == canon_index)
    scales = []
    for m in metrics:
        if animation_type == "locomotion":
            s = canon.height / m.height
        else:
            s = float(np.sqrt(canon.core_area / max(1, m.core_area)))
            s = float(np.clip(s, 0.85, 1.15))
            if abs(s - 1.0) < 0.05:
                s = 1.0
        scales.append(s)
    resized = []
    for f, m, s in zip(frames, metrics, scales):
        if s == 1.0:
            rgba = f.rgba
        else:
            nw, nh = max(1, int(round(m.width * s))), max(1, int(round(m.height * s)))
            rgba = cv2.resize(f.rgba, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4)
        gx = int(round(m.ground_x * s))
        gy = rgba.shape[0] - 1
        resized.append((rgba, gx, gy))
    # canvas from the union of extents relative to the pivot
    left = max(gx for _, gx, _ in resized)
    right = max(r.shape[1] - gx for r, gx, _ in resized)
    top = max(gy for _, _, gy in resized)
    margin = 8
    cell_w, cell_h = cell if cell else (left + right + 2 * margin, top + 1 + 2 * margin)
    px, py = pivot if pivot else (left + margin, cell_h - 1 - margin)
    canvases, records = [], []
    for (rgba, gx, gy), f, s in zip(resized, frames, scales):
        nh, nw = rgba.shape[:2]
        canvas = np.zeros((cell_h, cell_w, 4), np.uint8)
        ox, oy = px - gx, py - gy
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(cell_w, ox + nw), min(cell_h, oy + nh)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = rgba[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        canvases.append(canvas)
        records.append({"index": f.index, "scale": round(s, 4), "offset": [ox, oy],
                        "clipped": bool(x0 > ox or y0 > oy or x1 < ox + nw or y1 < oy + nh)})
    return canvases, records


# ------------------------------------------------------------- report --

def contact_sheet(frames: Sequence[Frame], flagged: Sequence[int], height: int = 220) -> np.ndarray:
    tiles = []
    for f in frames:
        rgb = f.rgba[..., :3].copy()
        rgb[~f.mask] = 90
        h, w = rgb.shape[:2]
        tile = cv2.resize(rgb, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_AREA)
        colour = (255, 60, 60) if f.index in flagged else (60, 200, 90)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), colour, 4)
        cv2.putText(tile, str(f.index), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        tiles.append(tile)
    gap = np.zeros((height, 6, 3), np.uint8)
    parts = []
    for t in tiles:
        parts += [t, gap]
    return np.concatenate(parts[:-1], axis=1)


def inspect_animation(sheet_path: str | Path, out: str | Path, *, count: int = 8, animation: str = "anim",
                      declared_facing: str = "auto", canon_frame: int = 1, loop: bool = True,
                      animation_type: str = "action", force: bool = False) -> dict[str, Any]:
    frames = load_animation(sheet_path, count)
    metrics = [measure(f) for f in frames]
    orient = orientation_report(frames, metrics, declared_facing)
    consist = consistency_report(metrics, canon_frame, animation_type)
    cont = continuity_report(frames, metrics, loop)
    flagged = sorted(set(orient["flip_candidates"]) | {r["index"] for r in consist["frames"] if r["status"] == "fail"})
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png(out_dir / "contact.png", contact_sheet(frames, flagged), force=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    for f in frames:
        save_png(out_dir / "frames" / f"{animation}_{f.index:02d}.png", f.rgba, force=True)
    status = "fail" if consist["status"] == "fail" else ("warn" if (flagged or consist["status"] == "warn"
                                                                    or any(p["flag"] for p in cont["pairs"])) else "pass")
    report = {
        "schema": "platekit.character_inspect", "version": 1, "platekit_version": __version__,
        "animation": animation, "source": str(sheet_path), "source_sha256": sha256_file(sheet_path),
        "frame_count": count, "canon_frame": canon_frame, "animation_type": animation_type, "status": status,
        "flagged_frames": flagged,
        "metrics": [m.to_dict() for m in metrics],
        "orientation": orient, "consistency": consist, "continuity": cont,
        "summary": _summary_lines(orient, consist, cont),
    }
    save_json(out_dir / "inspect.json", report)
    return report


def _summary_lines(orient: dict, consist: dict, cont: dict) -> list[str]:
    lines = []
    if orient["flip_candidates"]:
        lines.append(f"방향 뒤집힘 후보: 프레임 {orient['flip_candidates']} "
                     f"(의도 방향 {orient['intended_facing']} / 근거 {orient['intended_source']}; "
                     f"전환 {orient['facing_transitions']})")
    else:
        lines.append(f"방향 일관 ({orient['intended_facing'] or '정면'}).")
    bad = [f"{r['index']}:{','.join(r['flags'])}" for r in consist["frames"] if r["flags"]]
    lines.append("통일성: " + ("이상 없음" if not bad else "; ".join(bad)))
    outl = [p["pair"] for p in cont["pairs"] if p["outlier"]]
    lines.append("연속성 이상치: " + (", ".join(outl) if outl else "없음"))
    return lines


def fix_animation(sheet_path: str | Path, out: str | Path, *, count: int = 8, animation: str = "anim",
                  mirror_frames: Sequence[int] | None = None, declared_facing: str = "auto",
                  canon_frame: int = 1, normalise: bool = True, animation_type: str = "action",
                  force: bool = False) -> dict[str, Any]:
    """Apply the explicit mirror fix (and optional normalisation), then re-inspect.

    ``mirror_frames`` None = use the inspector's flip candidates. The fix is only a whole-frame
    horizontal mirror; nothing else is redrawn.
    """
    frames = load_animation(sheet_path, count)
    metrics = [measure(f) for f in frames]
    before = orientation_report(frames, metrics, declared_facing)
    targets = list(mirror_frames) if mirror_frames is not None else list(before["flip_candidates"])
    # Sandwich rule: a frame whose eyes could not be read (e.g. eye_dx below the deadzone at the
    # bottom of a swing) but whose cyclic neighbours are both being mirrored is mirrored too.
    # On the real dig_A sheet frame 6 (eye_dx +3.1) sits between mirrored 5 and 7.
    inferred: list[int] = []
    if mirror_frames is None:
        n = len(frames)
        facing_by_index = {m.index: m.facing for m in metrics}
        for f in frames:
            if f.index in targets or facing_by_index[f.index] != "-":
                continue
            left = frames[(f.index - 2) % n].index
            right = frames[f.index % n].index
            if left in targets and right in targets:
                inferred.append(f.index)
        targets = sorted(set(targets) | set(inferred))
    fixed = apply_mirror_fix(frames, targets)
    fixed_metrics = [measure(f) for f in fixed]
    after = orientation_report(fixed, fixed_metrics, declared_facing if declared_facing != "auto" else before["intended_facing"])

    out_dir = Path(out)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    records = []
    if normalise:
        canvases, norm = normalise_frames(fixed, fixed_metrics, canon_frame, animation_type)
        for f, canvas, n in zip(fixed, canvases, norm):
            save_png(out_dir / "frames" / f"{animation}_{f.index:02d}.png", canvas, force=True)
            records.append({"index": f.index, "file": f"frames/{animation}_{f.index:02d}.png",
                            "derived": f.index in targets, "transform": "mirror_x" if f.index in targets else None,
                            "normalise": n})
        cell = [int(canvases[0].shape[1]), int(canvases[0].shape[0])]
    else:
        for f in fixed:
            save_png(out_dir / "frames" / f"{animation}_{f.index:02d}.png", f.rgba, force=True)
            records.append({"index": f.index, "file": f"frames/{animation}_{f.index:02d}.png",
                            "derived": f.index in targets, "transform": "mirror_x" if f.index in targets else None})
        cell = None
    save_png(out_dir / "contact_after.png", contact_sheet(fixed, []), force=True)
    remaining = after["flip_candidates"]
    report = {
        "schema": "platekit.character_fix", "version": 1, "platekit_version": __version__,
        "animation": animation, "source": str(sheet_path), "source_sha256": sha256_file(sheet_path),
        "mirrored_frames": targets, "mirrored_by_neighbour_rule": inferred,
        "result_facing": after["intended_facing"],
        "remaining_flip_candidates": remaining,
        "status": "pass" if not remaining else "warn",
        "cell_px": cell, "frames": records,
        "before": {"flip_candidates": before["flip_candidates"], "transitions": before["facing_transitions"]},
        "after": {"flip_candidates": remaining, "transitions": after["facing_transitions"]},
    }
    save_json(out_dir / "fix.json", report)
    return report


# ------------------------------------------------------- cross-sheet --

# Style metrics are chosen for *stability under pose and view*, measured on 18 real sheets.
#
#   head_width_ratio  widest row in the top 25 % / figure height. THE proportion signal.
#                     Within the walk family it varies 8.5 %; the chibi dig sheet sits 48 % away,
#                     a 5.7:1 separation-to-noise ratio.
#   skin_lab          skin class mean Lab. Palette drift between deliveries.
#   line_darkness     mean luminance of the darkest 5 % of the figure = outline weight.
#
# Two metrics were tried and rejected against this corpus:
#   eye_height_ratio  invalid on back views - there are no eyes, and the detector locks onto the
#                     orange ear interiors instead, reporting a plausible 0.20-0.23. It is also
#                     normalised by total height, which changes with stance, so a side view reads
#                     0.117 and a front view 0.163 for the same character.
#   hair-blob head    this character's hair and tail are the same white, so the blob merges with
#                     the tail and head-count swings 1.0-8.8 on identical art.
STYLE_HEAD_WARN = 0.030      # ~ the walk family's own spread
STYLE_HEAD_FAIL = 0.055
STYLE_SKIN_DE_WARN = 6.0
STYLE_SKIN_DE_FAIL = 12.0
STYLE_LINE_WARN = 18.0


def _head_width_ratio(frame: Frame) -> float:
    """Widest row of the top quarter, over figure height. View-invariant: the head silhouette
    (hair + ears) is present from the front, the side and the back."""
    height = frame.mask.shape[0]
    top = frame.mask[: max(1, int(height * 0.25))]
    return float(top.sum(axis=1).max() / height)


def style_signature(sheet_path: str | Path, count: int = 8) -> dict[str, Any]:
    """Pose-stable style metrics for one sheet, taken as the median over its frames."""
    from .metrics import colour_classes

    frames = load_animation(sheet_path, count)
    heads, skins, lines, fills = [], [], [], []
    for f in frames:
        heads.append(_head_width_ratio(f))
        rgb = f.rgba[..., :3].copy()
        rgb[~f.mask] = 0
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        classes = colour_classes(rgb, f.mask)
        if int(classes["skin"].sum()) > 50:
            skins.append(lab[classes["skin"]].astype(np.float32).mean(axis=0))
        lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[f.mask].astype(np.float32)
        if lum.size > 100:
            lines.append(float(np.mean(np.sort(lum)[: max(1, int(lum.size * 0.05))])))
        fills.append(float(f.mask.mean()))
    return {
        "sheet": str(sheet_path),
        "frames": len(frames),
        "split_method": frames[0].split_method,
        "head_width_ratio": float(np.median(heads)),
        "skin_lab": [float(v) for v in np.median(np.stack(skins), axis=0)] if skins else None,
        "line_darkness": float(np.median(lines)) if lines else None,
        "fill_ratio": float(np.median(fills)),
        "median_height_px": float(np.median([f.mask.shape[0] for f in frames])),
    }


# Declared canon for Lyn, fixed at v1.12 from the five walk sheets' median. Locomotion was chosen
# as the baseline because it is the irreplaceable set: five directions already exist, while a dig
# cycle needs only one direction (the rest come from mirror derivation and the anchor's facing).
# Measured: walk eye ratio 0.117-0.219 (median 0.163), skin Lab (218.9, 134.0, 138.9), line 69.
LYN_CANON = {
    "id": "lyn_walk_median_v1_12",
    "head_width_ratio": 0.232,
    "skin_lab": [218.9, 134.0, 138.9],
    "line_darkness": 69.0,
}


def compare_sheets(sheets: Sequence[str | Path], out: str | Path, *, count: int = 8,
                   canon: int | dict[str, Any] = 0, names: Sequence[str] | None = None,
                   force: bool = False) -> dict[str, Any]:
    """Do these sheets look like the same character drawn the same way?

    Within-sheet QA cannot see this: every sheet can be internally perfect and still belong to a
    different art style. This is the check that catches "the walk cycle is a different character
    from the dig cycle".
    """
    paths = [Path(s) for s in sheets]
    ids = list(names) if names else [p.stem for p in paths]
    if len(ids) != len(paths):
        raise ValueError("names 개수가 시트 수와 다릅니다.")
    sigs = [style_signature(p, count) for p in paths]
    if isinstance(canon, dict):
        base = canon
        canon_id = str(canon.get("id", "declared"))
    else:
        base = sigs[canon % len(sigs)]
        canon_id = ids[canon % len(ids)]
    rows: list[dict[str, Any]] = []
    worst = "pass"
    for sid, sig in zip(ids, sigs):
        flags: list[str] = []
        head_delta = sig["head_width_ratio"] - base["head_width_ratio"]
        if abs(head_delta) >= STYLE_HEAD_FAIL:
            flags.append(f"PROPORTION_MISMATCH head {head_delta:+.3f}")
        elif abs(head_delta) >= STYLE_HEAD_WARN:
            flags.append(f"proportion head {head_delta:+.3f}")
        skin_de = None
        if sig["skin_lab"] and base["skin_lab"]:
            skin_de = float(np.linalg.norm(np.array(sig["skin_lab"]) - np.array(base["skin_lab"])))
            if skin_de >= STYLE_SKIN_DE_FAIL:
                flags.append(f"PALETTE_MISMATCH skin dE {skin_de:.1f}")
            elif skin_de >= STYLE_SKIN_DE_WARN:
                flags.append(f"palette skin dE {skin_de:.1f}")
        line_delta = None
        if sig["line_darkness"] is not None and base["line_darkness"] is not None:
            line_delta = sig["line_darkness"] - base["line_darkness"]
            if abs(line_delta) >= STYLE_LINE_WARN:
                flags.append(f"line_weight {line_delta:+.0f}")
        status = "fail" if any(f.split()[0].isupper() for f in flags) else ("warn" if flags else "pass")
        if status == "fail" or (status == "warn" and worst == "pass"):
            worst = status
        rows.append({"id": sid, "status": status, "flags": flags,
                     "head_width_ratio": sig["head_width_ratio"], "head_delta": head_delta,
                     "skin_dE": None if skin_de is None else round(skin_de, 2),
                     "line_darkness": sig["line_darkness"], "line_delta": line_delta,
                     "split_method": sig["split_method"], "median_height_px": sig["median_height_px"]})
    report = {"schema": "platekit.character_compare", "version": 2, "platekit_version": __version__,
              "canon": canon_id, "canon_values": {k: v for k, v in base.items()
                                                  if k in ("head_width_ratio", "skin_lab", "line_darkness")},
              "status": worst, "sheets": rows,
              "note": ("머리 폭 비율(상단 25% 최대 폭 / 키)이 주 비율 지표다. 눈 높이는 뒷모습에서 "
                       "귀 안쪽을 눈으로 오인하고 자세에 따라 흔들려 폐기했다.")}
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "compare.json", report)
    return report
