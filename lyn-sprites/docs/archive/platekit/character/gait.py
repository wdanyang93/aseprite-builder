"""Gait model measured from a 3D walk reference, used as the standard the art is corrected to.

The art is canon for design, colour and build. It is not canon for pose. The reference sheet's own
frames disagree with each other by more than a walk cycle allows, so the cycle itself has to come
from somewhere else - here, eight frames of a 3D figure walking, measured directly.

Everything below is a measurement, normalised by figure height so it transfers to any character:

**Fixed landmarks.** Across the eight reference frames, in units of figure height,

    crown      y 0.000   sd 0.0000   <- exactly fixed
    neck       y 0.143   sd 0.0026
    shoulder   y 0.170   sd 0.0076
    shoulder width 0.089 sd 0.0039
    hip        y 0.557   sd 0.1118   <- moves 15x more than the shoulder

So the shoulder is a pin, not a variable. An arm may be re-posed about it; it may not be moved.

**Arm swing.** Measured from the centreline drawn along the arm in the reference:

    102  110  104   92   76   67   78   83  degrees from the shoulder
    frame-to-frame change:  +8  -7 -12 -16  -9 +12  +4

Amplitude 43 degrees, and no single step exceeds 16. The delivered art jumps 45 degrees between
frames 1 and 2 - three times the reference's largest step - which is the defect being corrected.

**Stride.** Toe separation over figure height:

    0.453 0.382 0.171 0.174 0.113 0.411 0.376 0.219

Wide to narrow twice per cycle, as a walk must.
"""
from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

# --- measured from regression_assets/v1_13_gait_ref/gait_8f.png ---
LANDMARKS = {
    "crown": {"y": 0.000, "sd": 0.0000},
    "neck": {"y": 0.143, "sd": 0.0026},
    "shoulder": {"y": 0.170, "sd": 0.0076, "width": 0.089, "width_sd": 0.0039},
    "hip": {"y": 0.557, "sd": 0.1118},
}
FIXED_LANDMARKS = ("crown", "neck", "shoulder")     # sd below 0.01 of height
ARM_SWING_DEG = (101.6, 110.0, 103.5, 91.7, 76.0, 66.7, 78.4, 82.8)
STRIDE_OVER_HEIGHT = (0.453, 0.382, 0.171, 0.174, 0.113, 0.411, 0.376, 0.219)
MAX_STEP_DEG = 16.0            # largest frame-to-frame arm change in the reference
SWING_AMPLITUDE_DEG = 43.0


# The reference is a realistic figure at 6.9 heads; Lyn is 3.8 heads. Measured: the reference's
# neck sits at 0.145 of its height, Lyn's at 0.263-0.266 - a difference of 47 reference sigmas,
# which is proportion, not error. So a landmark's height fraction does NOT transfer between them.
# What transfers is stated in TRANSFERABLE below, and it is what the correction actually needs.
REFERENCE_HEADS = 6.9
TRANSFERABLE = {
    "shoulder_is_pinned": "어깨는 프레임간 신장의 0.8% 이내로 고정 (레퍼런스 sd 0.0076)",
    "hip_moves": "골반은 어깨보다 15배 움직인다 (sd 0.1118 대 0.0076)",
    "arm_swing_deg": "팔 스윙 각도는 비율과 무관하게 그대로 쓴다",
    "max_step_deg": "프레임간 팔 각도 변화 상한 16도",
    "stride_over_height": "보폭/신장 수열은 근사적으로 전이된다",
}
# Landmarks are therefore compared against the character's OWN median across its frames, using
# the reference only for how much they are allowed to vary.
SHOULDER_STABILITY = 0.0076    # of figure height
HIP_MOBILITY = 0.1118


def self_consistency(per_frame: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Are the character's own fixed landmarks stable across its own frames?

    This is the test that survives the proportion gap: whatever height fraction this character's
    shoulder sits at, it must sit at the SAME fraction in every frame, within the reference's
    tolerance. A frame outside that is a drawing error, not a different body.
    """
    out: dict[str, Any] = {}
    for name, tolerance in (("neck", SHOULDER_STABILITY), ("shoulder", SHOULDER_STABILITY),
                            ("shoulder_width", LANDMARKS["shoulder"]["width_sd"] * 3)):
        values = [f.get(name) for f in per_frame if f.get(name) is not None]
        if len(values) < 2:
            continue
        arr = np.asarray(values, np.float64)
        median = float(np.median(arr))
        offenders = [i + 1 for i, v in enumerate(values) if abs(v - median) > tolerance]
        out[name] = {"median": round(median, 4), "sd": round(float(arr.std()), 4),
                     "range": round(float(arr.max() - arr.min()), 4),
                     "tolerance": tolerance, "offending_frames": offenders,
                     "stable": not offenders}
    return out


def measure_landmarks(mask: np.ndarray) -> dict[str, Any]:
    """Crown, neck, shoulder and hip of one figure, in units of its own height.

    The same row-width profile the reference was measured with, so the numbers are comparable:
    the neck is the narrowest row above the shoulder, the shoulder is where the profile widens
    fastest, and the hip is the narrowest row in the middle of the body.
    """
    ys = np.nonzero(mask.any(axis=1))[0]
    if ys.size == 0:
        return {}
    top, bottom = int(ys.min()), int(ys.max())
    height = max(1, bottom - top + 1)
    profile = mask.sum(axis=1).astype(np.float32)
    window = max(5, int(height * 0.02)) | 1
    smooth = np.convolve(profile, np.ones(window) / window, mode="same")

    # Search windows are widened to cover both proportions: a 6.9-head figure puts the neck at
    # 0.145 of its height and a 3.8-head one at 0.265, so a window tuned to either misses the
    # other entirely - which is why the first pass reported Lyn's neck at 0.048 on three frames.
    neck_lo, neck_hi = top + int(height * 0.10), top + int(height * 0.34)
    neck = neck_lo + int(np.argmin(smooth[neck_lo:neck_hi])) if neck_hi > neck_lo + 2 else neck_lo
    sh_lo, sh_hi = neck, min(bottom, neck + int(height * 0.14))
    shoulder = sh_lo + int(np.argmax(np.diff(smooth[sh_lo:sh_hi]))) if sh_hi > sh_lo + 2 else sh_lo
    hip_lo, hip_hi = top + int(height * 0.46), top + int(height * 0.72)
    hip = hip_lo + int(np.argmin(smooth[hip_lo:hip_hi])) if hip_hi > hip_lo + 2 else hip_lo

    shoulder_cols = np.nonzero(mask[shoulder])[0]
    out = {
        "height_px": height, "top": top,
        "crown": 0.0,
        "neck": (neck - top) / height,
        "shoulder": (shoulder - top) / height,
        "hip": (hip - top) / height,
        "shoulder_row": int(shoulder), "neck_row": int(neck), "hip_row": int(hip),
    }
    if shoulder_cols.size:
        out["shoulder_cx"] = float(shoulder_cols.mean())
        out["shoulder_width"] = float(shoulder_cols.max() - shoulder_cols.min() + 1) / height
    return out


def landmark_deviation(measured: dict[str, Any]) -> dict[str, Any]:
    """How far a frame's fixed landmarks sit from the reference, in standard deviations.

    A landmark more than three reference sd away is not natural variation; it is a drawing that
    put the shoulder somewhere a walking body does not put it.
    """
    out: dict[str, Any] = {}
    for name in FIXED_LANDMARKS:
        if name not in measured:
            continue
        spec = LANDMARKS[name]
        delta = measured[name] - spec["y"]
        sigma = max(spec["sd"], 1e-4)
        out[name] = {"measured": round(measured[name], 4), "reference": spec["y"],
                     "delta": round(delta, 4), "sigmas": round(delta / sigma, 1),
                     "out_of_range": abs(delta) > 3 * sigma}
    if "shoulder_width" in measured:
        spec = LANDMARKS["shoulder"]
        delta = measured["shoulder_width"] - spec["width"]
        out["shoulder_width"] = {"measured": round(measured["shoulder_width"], 4),
                                 "reference": spec["width"],
                                 "delta": round(delta, 4),
                                 "sigmas": round(delta / max(spec["width_sd"], 1e-4), 1),
                                 "out_of_range": abs(delta) > 3 * spec["width_sd"]}
    return out


def arm_angle(arm_mask: np.ndarray, shoulder: Sequence[float]) -> float | None:
    """Angle of the shoulder-to-hand line, degrees, measured the same way as the reference."""
    if not arm_mask.any():
        return None
    ys, xs = np.nonzero(arm_mask)
    far = ys >= ys.max() - 8
    if not far.any():
        return None
    hand = (float(xs[far].mean()), float(ys.max()))
    return float(np.degrees(np.arctan2(hand[1] - shoulder[1], hand[0] - shoulder[0])))


def swing_plan(current: Sequence[float | None], *,
               phase_offset: int = 0) -> list[dict[str, Any]]:
    """Target angle per frame and the rotation needed, aligned to the reference cycle.

    ``phase_offset`` rotates which reference frame frame 1 corresponds to, chosen by whichever
    alignment needs the least total correction - the art may simply start the cycle elsewhere.
    """
    count = len(current)
    targets = [ARM_SWING_DEG[(i + phase_offset) % len(ARM_SWING_DEG)] for i in range(count)]
    plan = []
    for i, (now, target) in enumerate(zip(current, targets), start=1):
        entry = {"frame": i, "target_deg": round(target, 1)}
        if now is None:
            entry["status"] = "no_arm_mask"
        else:
            entry["current_deg"] = round(now, 1)
            entry["rotate_deg"] = round(target - now, 1)
        plan.append(entry)
    return plan


def best_phase(current: Sequence[float | None]) -> tuple[int, float]:
    """Which phase alignment costs least total rotation."""
    best, best_cost = 0, float("inf")
    for offset in range(len(ARM_SWING_DEG)):
        cost = 0.0
        n = 0
        for i, now in enumerate(current):
            if now is None:
                continue
            target = ARM_SWING_DEG[(i + offset) % len(ARM_SWING_DEG)]
            cost += abs(target - now)
            n += 1
        if n and cost / n < best_cost:
            best, best_cost = offset, cost / n
    return best, round(best_cost, 1)


def continuity_flags(angles: Sequence[float | None]) -> list[str]:
    """Frame-to-frame steps larger than the reference ever takes."""
    flags = []
    for i in range(len(angles)):
        a, b = angles[i], angles[(i + 1) % len(angles)]
        if a is None or b is None:
            continue
        step = abs(b - a)
        if step > MAX_STEP_DEG:
            flags.append(f"{i + 1}->{(i + 1) % len(angles) + 1}: {step:.0f}deg "
                         f"(reference max {MAX_STEP_DEG:.0f})")
    return flags


# ------------------------------------------------- parametric cycle --

# The walk cycle as a rule, not as a copy of any one sheet.
#
# Fitting the art to f1 and f2 was the wrong order: two frames of a broken cycle cannot define the
# cycle. And the 3D reference's own frames turned out to be unevenly sampled - its stride reads
# 0.448 0.335 0.171 0.173 0.112 0.130 0.364 0.218, which is not a clean wide-narrow-wide-narrow -
# so it cannot be copied frame-for-frame either. What the references DO give reliably is the
# amplitudes, and those are all this model needs:
#
#     stride peak-to-peak   0.45 of figure height   (reference range -0.26 to +0.25)
#     arm swing amplitude   43 degrees              (reference 67 to 110)
#     shoulder              fixed                   (sd 0.0076 of height)
#     hip vertical bob      twice per cycle         (hip sd 0.1118, 15x the shoulder)
#
# The cycle itself is the standard eight-pose walk, and the constraint that makes it read as
# walking is contralateral opposition: the arm swings opposite to the leg on the same side.
#
#     1 contact   2 down   3 passing   4 up   5 contact'  6 down   7 passing   8 up
#
# Phase runs one full turn over the eight frames, so every quantity is a cosine of the phase and
# the sequence is continuous by construction - there is no frame-to-frame jump to police.

PHASE_POSES = ("contact", "down", "passing", "up",
               "contact_mirror", "down_mirror", "passing_mirror", "up_mirror")
STRIDE_AMPLITUDE = 0.225       # of figure height, each foot's swing either side of centre
ARM_AMPLITUDE_DEG = 21.5       # half of the 43 degree measured swing
ARM_CENTRE_DEG = 88.4          # midpoint of the reference's 67-110 range
HIP_BOB = 0.012                # of height, twice per cycle
FOOT_LIFT = 0.06               # of height, the swing foot leaves the ground


def cycle_pose(frame: int, count: int = 8, *, mirror: bool = False) -> dict[str, Any]:
    """Where every moving part should be, for one frame of the cycle.

    All positions are in figure heights with the shoulder as the origin, so they apply to any
    proportion - which matters because the reference is 6.9 heads and Lyn is 3.8.
    """
    phase = 2.0 * np.pi * ((frame - 1) % count) / count
    if mirror:
        phase += np.pi
    near_leg = STRIDE_AMPLITUDE * float(np.cos(phase))
    far_leg = -near_leg
    # contralateral: the near arm opposes the near leg
    near_arm_deg = ARM_CENTRE_DEG - ARM_AMPLITUDE_DEG * float(np.cos(phase))
    far_arm_deg = ARM_CENTRE_DEG + ARM_AMPLITUDE_DEG * float(np.cos(phase))
    return {
        "frame": frame,
        "pose": PHASE_POSES[(frame - 1) % len(PHASE_POSES)],
        "phase_deg": round(np.degrees(phase) % 360.0, 1),
        "near_foot_x": round(near_leg, 4),
        "far_foot_x": round(far_leg, 4),
        "stride": round(abs(near_leg - far_leg), 4),
        "near_arm_deg": round(near_arm_deg, 1),
        "far_arm_deg": round(far_arm_deg, 1),
        "hip_dy": round(-HIP_BOB * float(np.cos(2.0 * phase)), 4),
        "near_foot_lift": round(max(0.0, FOOT_LIFT * float(np.sin(phase))), 4),
        "far_foot_lift": round(max(0.0, -FOOT_LIFT * float(np.sin(phase))), 4),
        "leading": "near" if near_leg > far_leg else "far",
    }


def cycle_plan(count: int = 8, *, mirror: bool = False) -> list[dict[str, Any]]:
    return [cycle_pose(i, count, mirror=mirror) for i in range(1, count + 1)]


def score_against_cycle(measured: Sequence[dict[str, Any]], *,
                        count: int = 8) -> dict[str, Any]:
    """How far a set of frames sits from the rule, trying every phase offset and both mirrors.

    Returns the best alignment and the per-frame correction, so a whole sheet is corrected as one
    cycle rather than frame by frame - fixing f1 against f2 only moves the error to f3.
    """
    best: dict[str, Any] | None = None
    for mirror in (False, True):
        for offset in range(count):
            total, n = 0.0, 0
            rows = []
            for i, obs in enumerate(measured):
                plan = cycle_pose(((i + offset) % count) + 1, count, mirror=mirror)
                row = {"frame": i + 1, "target": plan}
                if obs.get("near_arm_deg") is not None:
                    delta = plan["near_arm_deg"] - obs["near_arm_deg"]
                    row["arm_rotate_deg"] = round(delta, 1)
                    total += abs(delta)
                    n += 1
                if obs.get("stride") is not None:
                    row["stride_delta"] = round(plan["stride"] - obs["stride"], 4)
                    total += abs(row["stride_delta"]) * 100.0
                    n += 1
                rows.append(row)
            cost = total / max(1, n)
            if best is None or cost < best["cost"]:
                best = {"cost": round(cost, 2), "offset": offset, "mirror": mirror, "rows": rows}
    return best or {}
