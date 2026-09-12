"""옷을 부위(사지)에 묶고 프레임마다 옮긴다.

원리
    옷은 화면 좌표가 아니라 **사지의 (s, u) 좌표**로 정의한다.
      s = 사지를 따라간 거리 (0=부착, 1=말단)
      u = 중심선에서의 좌우 오프셋 (-1..+1, 그 지점의 반경으로 정규화)
    이 좌표는 자세와 무관하므로, 프레임의 사지 리본만 알면 그대로 옮겨진다.

    고정 좌표 도장이 안 되는 이유는 실측으로 확인됐다 — 하의의 신장 정규화 y가
    걷기 0.48~0.56, 달리기 0.42~0.57 로 애니메이션마다 어긋난다. (s,u) 는 안 어긋난다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .rig2 import Rig
from .rigfit import Limb, limbs_of


@dataclass
class GarmentPatch:
    """한 부위에 붙는 옷 조각. (s,u) 격자 위의 색과 불투명도."""
    limb_name: str
    s_range: tuple[float, float]        # 사지를 따라 어디부터 어디까지
    u_range: tuple[float, float]        # 중심선에서 좌우 어디까지 (반경 배수)
    colour: tuple[int, int, int]
    shade_from_body: bool = True        # 몸의 원래 명암을 곱해 쓴다


def limb_frame(limb: Limb, height_px: float, origin: np.ndarray, root_px: np.ndarray):
    """사지의 중심선·법선·반경을 픽셀 좌표로."""
    P = limb.points * height_px + root_px
    R = limb.radius * height_px
    d = np.gradient(P, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    T = d / np.maximum(n, 1e-6)
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)
    seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    s = seg / max(seg[-1], 1e-6)
    return P, R, N, s


def patch_mask(shape, limb: Limb, height_px: float, root_px: np.ndarray,
               patch: GarmentPatch, body: np.ndarray) -> np.ndarray:
    """(s,u) 범위를 픽셀 마스크로. 몸 실루엣 안으로 제한한다."""
    P, R, N, s = limb_frame(limb, height_px, None, root_px)
    m = np.zeros(shape, np.uint8)
    lo, hi = patch.s_range
    sel = (s >= lo) & (s <= hi)
    idx = np.nonzero(sel)[0]
    if len(idx) < 2:
        return m.astype(bool)
    u0, u1 = patch.u_range
    left = P[idx] + N[idx] * (R[idx] * u1)[:, None]
    right = P[idx] + N[idx] * (R[idx] * u0)[:, None]
    poly = np.concatenate([left, right[::-1]], axis=0).astype(np.int32)
    cv2.fillPoly(m, [poly], 1)
    return (m > 0) & body


def anatomical_s(limb: Limb, height_px: float, root_px: np.ndarray,
                 hip_row: float, neck_row: float) -> tuple[np.ndarray, float, float] | None:
    """사지의 s 를 해부학 랜드마크로 다시 잡는다.

    리그 뿌리는 '골격에서 가장 두꺼운 점'이라 프레임마다 골반<->가슴으로 옮겨 다닌다.
    그 위에서 잰 s 는 프레임마다 다른 신체 부위를 가리킨다(1차 시도가 얼굴에 옷을 입힌 원인).
    힙선과 목은 실측상 훨씬 안정적이다 — 걷기 8프레임 sd 힙선 0.0118, 목 0.0048.
    두 지점이 s=0, s=1 이 되도록 다시 매개화한다.
    """
    P, R, N, s = limb_frame(limb, height_px, None, root_px)
    y = P[:, 1]
    def cross(row):
        d = y - row
        sign = np.sign(d)
        idx = np.nonzero(np.diff(sign) != 0)[0]
        if not len(idx):
            return None
        i = idx[int(np.argmin(np.abs(d[idx])))]
        a, b = d[i], d[i + 1]
        f = 0.0 if a == b else a / (a - b)
        return float(s[i] + f * (s[i + 1] - s[i]))
    t_hip, t_neck = cross(hip_row), cross(neck_row)
    if t_hip is None or t_neck is None or abs(t_neck - t_hip) < 1e-3:
        return None
    return s, t_hip, t_neck


def patch_mask_anat(shape, limb: Limb, height_px: float, root_px: np.ndarray,
                    patch: GarmentPatch, body: np.ndarray,
                    hip_row: float, neck_row: float) -> np.ndarray:
    """힙선(s'=0)~목(s'=1) 로 다시 잡은 좌표에서 옷 조각을 만든다."""
    got = anatomical_s(limb, height_px, root_px, hip_row, neck_row)
    if got is None:
        return np.zeros(shape, bool)
    s, t_hip, t_neck = got
    sp = (s - t_hip) / (t_neck - t_hip)          # 힙선=0, 목=1
    P, R, N, _ = limb_frame(limb, height_px, None, root_px)
    m = np.zeros(shape, np.uint8)
    lo, hi = patch.s_range
    idx = np.nonzero((sp >= lo) & (sp <= hi))[0]
    if len(idx) < 2:
        return m.astype(bool)
    u0, u1 = patch.u_range
    left = P[idx] + N[idx] * (R[idx] * u1)[:, None]
    right = P[idx] + N[idx] * (R[idx] * u0)[:, None]
    cv2.fillPoly(m, [np.concatenate([left, right[::-1]], axis=0).astype(np.int32)], 1)
    return (m > 0) & body


def torso_fallback(body: np.ndarray, hip_row: int, neck_row: int):
    """몸통 사지가 검출되지 않았을 때의 대체 리본.

    몸통은 힙선과 목 사이의 몸이고, 그 중심선은 각 행의 가로 중앙이다. 리그가 없어도
    두 랜드마크만 있으면 세워진 자세에서는 이것으로 충분하다.
    (g4 처럼 다리가 겹쳐 사지가 3개만 나오는 프레임을 위한 것)
    """
    lo, hi = int(min(hip_row, neck_row)), int(max(hip_row, neck_row))
    rows, cx, half = [], [], []
    for y in range(hi, lo - 1, -1):              # 힙 -> 목 방향
        xs = np.nonzero(body[y])[0]
        if xs.size < 3:
            continue
        rows.append(y); cx.append(xs.mean()); half.append((xs.max() - xs.min() + 1) / 2.0)
    if len(rows) < 4:
        return None
    P = np.stack([np.array(cx), np.array(rows, float)], axis=1)
    R = np.array(half)
    d = np.gradient(P, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    T = d / np.maximum(n, 1e-6)
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)
    sp = np.linspace(0.0, 1.0, len(rows))        # 힙=0, 목=1
    return P, R, N, sp


def patch_mask_rows(shape, body, hip_row, neck_row, patch: GarmentPatch) -> np.ndarray:
    got = torso_fallback(body, hip_row, neck_row)
    if got is None:
        return np.zeros(shape, bool)
    P, R, N, sp = got
    lo, hi = patch.s_range
    idx = np.nonzero((sp >= lo) & (sp <= hi))[0]
    if len(idx) < 2:
        return np.zeros(shape, bool)
    u0, u1 = patch.u_range
    left = P[idx] + N[idx] * (R[idx] * u1)[:, None]
    right = P[idx] + N[idx] * (R[idx] * u0)[:, None]
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.concatenate([left, right[::-1]], axis=0).astype(np.int32)], 1)
    return (m > 0) & body


def apply_patch(rgba: np.ndarray, region: np.ndarray, colour, shade_from_body=True,
                gamma: float = 1.0) -> np.ndarray:
    out = rgba.copy()
    if not region.any():
        return out
    if shade_from_body:
        L = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        ref = float(np.median(L[region]))
        ratio = np.clip((L[region] / max(ref, 1e-6)) ** gamma, 0.55, 1.45)
        new = np.clip(np.array(colour, np.float32)[None, :] * ratio[:, None], 0, 255)
    else:
        new = np.repeat(np.array(colour, np.float32)[None, :], int(region.sum()), axis=0)
    out[..., :3][region] = new.astype(np.uint8)
    return out
