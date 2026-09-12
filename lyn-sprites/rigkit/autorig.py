"""파츠에서 뼈대를 만든다 — 그림이 비율을 정하고 뼈대가 따라간다.

파츠마다 배율을 따로 걸어 뼈 길이에 맞추면, 길이는 맞지만 굵기 비율이 깨진다
(허벅지만 굵어지는 식). 한 시트에서 나온 파츠는 자기들끼리는 비율이 맞으므로,
**전체에 같은 배율 하나**만 걸고 관절 위치를 파츠 길이에서 계산한다.

fit.py 는 여러 시트에서 온 파츠를 억지로 맞출 때 쓰고, 이쪽이 기본이다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .fit import measured_length, opaque_box
from .spec import BODY_H

CHAIN = ["head", "torso", "thigh_r", "shin_r", "foot_r"]
ARM_CHAIN = ["upperarm_r", "forearm_r"]


def _load(parts_dir: Path) -> tuple[dict, dict[str, Image.Image]]:
    meta = json.loads((parts_dir / "parts.json").read_text())
    imgs = {k: Image.open(parts_dir / v["file"]).convert("RGBA") for k, v in meta.items()}
    return meta, imgs


def _full_height(img: Image.Image) -> float:
    x0, y0, x1, y1 = opaque_box(img)
    return float(y1 - y0)


def _width_at(img: Image.Image, y: float) -> tuple[float, float]:
    """그 높이에서 불투명 구간의 (중심 x, 너비). 파츠 이미지 좌표."""
    a = np.array(img)[..., 3] > 32
    row = int(min(max(y, 0), a.shape[0] - 1))
    xs = np.nonzero(a[row])[0]
    if len(xs) == 0:
        return (img.width / 2, 0.0)
    return ((xs.min() + xs.max()) / 2, float(xs.max() - xs.min() + 1))


def rig_from_parts(parts_dir: str | Path, *, write: bool = True) -> dict:
    """파츠 한 벌 → 관절 t + 좌우 오프셋 + 공통 배율."""
    d = Path(parts_dir)
    meta, imgs = _load(d)
    missing = [s for s in CHAIN if s not in imgs]
    if missing:
        raise KeyError(f"몸통 사슬에 빠진 파츠: {missing}")

    def _len(slot):
        return measured_length(imgs[slot], tuple(meta[slot]["anchor"]), slot,
                               meta[slot].get("length"))

    L_head = _len("head")
    L_torso = _full_height(imgs["torso"])
    L = {s: _len(s) for s in CHAIN[2:]}
    total = L_head + L_torso + L["thigh_r"] + L["shin_r"] + L["foot_r"]
    scale = BODY_H / total

    t_neck = L_head / total
    t_hips = (L_head + L_torso) / total
    t_chest = t_neck + 0.075 * L_torso / total
    t_sh = t_neck + 0.10 * L_torso / total
    joints = {
        "head_top": 0.0,
        "head": t_neck,
        "chest": t_chest,
        "shoulder": t_sh,
        "hips": t_hips,
        "knee": t_hips + L["thigh_r"] / total,
        "ankle": t_hips + (L["thigh_r"] + L["shin_r"]) / total,
        "sole": 1.0,
    }
    if all(s in imgs for s in ARM_CHAIN):
        La = {s: _len(s) for s in ARM_CHAIN}
        joints["elbow"] = t_sh + La["upperarm_r"] / total
        joints["wrist"] = joints["elbow"] + La["forearm_r"] / total
    else:
        joints["elbow"] = t_sh + (t_hips - t_sh) * 0.55
        joints["wrist"] = t_sh + (t_hips - t_sh) * 0.95

    # 좌우 오프셋: 몸통 파츠의 실제 너비에서 잰다
    x0, y0, x1, y1 = opaque_box(imgs["torso"])
    h = y1 - y0
    # 어깨는 상단 25% 안에서 가장 넓은 곳 (10% 지점은 목 경사라 너무 좁게 잡힌다)
    w_sh = max(_width_at(imgs["torso"], y0 + h * f)[1] for f in (0.06, 0.10, 0.14, 0.18, 0.22))
    _, w_hip = _width_at(imgs["torso"], y0 + h * 0.92)
    lateral = {
        "shoulder": round(w_sh * 0.52 * scale, 2),
        "elbow": round(w_sh * 0.58 * scale, 2),
        "wrist": round(w_sh * 0.62 * scale, 2),
        "hip": round(w_hip * 0.26 * scale, 2),
        "knee": round(w_hip * 0.28 * scale, 2),
        "ankle": round(w_hip * 0.30 * scale, 2),
    }

    out = {"scale": round(scale, 5),
           "measured_px": {"head": round(L_head, 1), "torso": round(L_torso, 1),
                           **{k: round(v, 1) for k, v in L.items()}, "total": round(total, 1)},
           "joints": {k: round(v, 4) for k, v in joints.items()},
           "lateral": lateral}
    if write:
        for slot in meta:
            meta[slot]["scale"] = out["scale"]
        (d / "parts.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        (d / "rig.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def apply(profile: dict) -> None:
    """계산한 프로필을 spec 에 반영한다."""
    from . import spec
    spec.apply_joints(profile["joints"])
    lat = profile["lateral"]
    spec.SHOULDER_DX = lat["shoulder"]; spec.ELBOW_DX = lat["elbow"]
    spec.WRIST_DX = lat["wrist"]; spec.HIP_DX = lat["hip"]
    spec.KNEE_DX = lat["knee"]; spec.ANKLE_DX = lat["ankle"]
