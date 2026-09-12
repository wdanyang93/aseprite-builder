"""신체 구역 지도 — 옷을 붙이기 전에 '큰 정의'를 먼저 확정한다.

왜 구역이 먼저인가
    1차 시도는 옷을 s 범위 숫자로 바로 정의했다가 얼굴에 상의를 입혔다.
    2차는 s 를 힙선~목으로 다시 잡아 제자리를 찾았지만, 상의 0.30~0.95 와
    하의 -0.25~0.22 사이에 틈이 생겼다 — 허리띠가 없는데 옷이 갈라졌다.
    숫자로 범위를 잡으면 이런 틈과 겹침이 계속 생긴다.
    이름 있는 경계를 먼저 확정하고 옷이 그 경계를 **공유**하게 해야 한다.

경계 (신장 정규화, 걷기 8프레임 실측 sd)
    목    0.251  sd 0.0048   measure_landmarks 의 neck_row
    어깨  0.264  sd 0.0129   measure_landmarks 의 shoulder_row
    허리  0.490  sd 0.0060   하의 블롭의 윗변          <- 상의/하의 공유 경계
    힙선  0.562  sd 0.0118   하의 블롭의 아랫변
    무릎  비례   다리 리본의 s=0.5 (검출 아님 — 이 그림체는 무릎이 잘록하지 않다.
                반경 최소를 쓰면 발목을 찾는다: 앞다리 sd 0.198 로 실패)
    발끝  1.000             다리 리본의 말단
    손목  비례   팔 리본의 s=0.85
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

GARMENT_SAT, GARMENT_VAL = 22, 230
KNEE_S = 0.50            # 힙선~발끝 사이 비율
WRIST_S = 0.85           # 어깨~손끝 사이 비율


def body_axis(mask: np.ndarray) -> tuple[float, np.ndarray]:
    """몸의 주축. 0도=가로(누움), 90도=세로(서있음).

    구역은 화면의 y 가 아니라 **몸을 따라** 흘러야 한다. 서 있을 때는 둘이 같지만
    엎드리면 신장이 폭이 되어 y 기준 판정이 전부 깨진다 (엎드림 11장 중 10장 실패).
    """
    ys, xs = np.nonzero(mask)
    cov = np.cov(np.stack([xs, ys]).astype(float))
    ev, evec = np.linalg.eigh(cov)
    v = evec[:, int(np.argmax(ev))]
    ang = abs(np.degrees(np.arctan2(v[1], v[0])))
    return min(ang, 180.0 - ang), v


def axis_coord(mask: np.ndarray) -> np.ndarray:
    """각 화소의 '몸을 따라간' 좌표 0~1. 머리 쪽이 0."""
    ys, xs = np.nonzero(mask)
    ang, v = body_axis(mask)
    c = np.array([xs.mean(), ys.mean()])
    t = (np.stack([xs, ys], 1) - c) @ v
    if ang >= 45.0:                      # 세워짐: y 가 커지는 쪽이 발
        if v[1] < 0: t = -t
    else:                                # 누움: 머리가 어느 쪽인지 모르므로 그대로 두고 호출측이 판단
        pass
    out = np.full(mask.shape, np.nan)
    out[ys, xs] = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    return out


def briefs_band(rgba: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    """현재 착용 중인 하의 블롭의 위·아래 행 = 허리·힙선."""
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    cloth = mask & (hsv[..., 1] < GARMENT_SAT) & (hsv[..., 2] > GARMENT_VAL)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cloth.astype(np.uint8), 8)
    ys = np.nonzero(mask.any(axis=1))[0]
    top, H = int(ys.min()), int(ys.max() - ys.min() + 1)
    cand = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < int(mask.sum() * 0.004):
            continue
        y0 = int(st[k, cv2.CC_STAT_TOP])
        y1 = y0 + int(st[k, cv2.CC_STAT_HEIGHT])
        if 0.42 <= (y1 - top) / H <= 0.66:
            cand.append((int(st[k, cv2.CC_STAT_AREA]), y0, y1))
    if not cand:
        # 누운 자세: y 기준이 무의미하다. 몸축 좌표로 다시 고른다.
        ang, _ = body_axis(mask)
        if ang < 45.0:
            ax = axis_coord(mask)
            for k in range(1, n):
                if st[k, cv2.CC_STAT_AREA] < int(mask.sum() * 0.004):
                    continue
                sel = lab == k
                a_med = float(np.nanmedian(ax[sel]))
                if 0.30 <= a_med <= 0.75:
                    yy = np.nonzero(sel)[0]
                    cand.append((int(st[k, cv2.CC_STAT_AREA]), int(yy.min()), int(yy.max())))
    if not cand:
        return None
    _, y0, y1 = max(cand)
    return y0, y1


@dataclass
class ZoneMap:
    neck: int
    shoulder: int
    waist: int
    hip: int
    foot: int
    height_px: float
    top_px: int
    measured: dict[str, bool]

    def norm(self, row: int) -> float:
        return (row - self.top_px) / self.height_px

    def to_dict(self) -> dict[str, Any]:
        return {k: dict(row=int(getattr(self, k)), y=round(self.norm(getattr(self, k)), 4),
                        measured=self.measured.get(k, True))
                for k in ("neck", "shoulder", "waist", "hip", "foot")}


def zone_map(rgba: np.ndarray, mask: np.ndarray, landmarks: dict[str, Any]) -> ZoneMap | None:
    ys = np.nonzero(mask.any(axis=1))[0]
    top, bot = int(ys.min()), int(ys.max())
    H = float(bot - top + 1)
    band = briefs_band(rgba, mask)
    if band is None:
        return None
    waist, hip = band
    return ZoneMap(neck=int(landmarks["neck_row"]), shoulder=int(landmarks["shoulder_row"]),
                   waist=int(waist), hip=int(hip), foot=bot, height_px=H, top_px=top,
                   measured=dict(neck=True, shoulder=True, waist=True, hip=True, foot=True))


# ------------------------------------------------------------ 구역 마스크 --

def leg_region(mask: np.ndarray, hip_row: int, floor_tol: float = 0.10) -> np.ndarray:
    """힙선 아래에서 다리만. 꼬리는 바닥에 닿지 않으므로 빠진다.

    실측: 꼬리 y_max 0.752~0.768 / 다리 0.916~0.988. 양쪽에서 0.08 이상 떨어진
    구조적 기준이라 색·질감처럼 프레임마다 흔들리지 않는다.
    """
    ys = np.nonzero(mask.any(axis=1))[0]
    top, bot = int(ys.min()), int(ys.max())
    H = bot - top + 1
    low = mask.copy()
    low[:hip_row] = False
    n, lab, st, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), 8)
    keep = np.zeros(mask.shape, bool)
    for k in range(1, n):
        sel = lab == k
        if sel.sum() < mask.sum() * 0.01:
            continue
        if np.nonzero(sel)[0].max() >= bot - floor_tol * H:
            keep |= sel
    return keep


def band_mask(mask: np.ndarray, row_from: int, row_to: int) -> np.ndarray:
    """화면 행 기준 띠. 세워진 자세 전용."""
    lo, hi = int(min(row_from, row_to)), int(max(row_from, row_to))
    m = np.zeros(mask.shape, bool)
    m[lo:hi + 1] = mask[lo:hi + 1]
    return m


def band_mask_axis(mask: np.ndarray, row_from: int, row_to: int) -> np.ndarray:
    """몸축 기준 띠. 누운 자세에서도 몸을 따라 흐른다.

    두 경계 행이 몸축 좌표에서 차지하는 값을 구해, 그 사이의 화소를 취한다.
    세워진 자세면 행 기준과 같아지고, 누우면 자동으로 몸을 따라간다.
    """
    ang, _ = body_axis(mask)
    if ang >= 45.0:
        return band_mask(mask, row_from, row_to)
    ax = axis_coord(mask)
    vals = []
    for r in (int(row_from), int(row_to)):
        r = max(0, min(mask.shape[0] - 1, r))
        row = ax[r][mask[r]] if mask[r].any() else np.array([])
        if row.size == 0:
            # 그 행에 몸이 없으면 가장 가까운 몸 있는 행으로
            rows = np.nonzero(mask.any(axis=1))[0]
            r = int(rows[np.argmin(np.abs(rows - r))])
            row = ax[r][mask[r]]
        vals.append(float(np.nanmedian(row)))
    lo, hi = min(vals), max(vals)
    return mask & (ax >= lo) & (ax <= hi)


def garment_region(mask: np.ndarray, z: "ZoneMap", start: str, end: str,
                   knee_s: float = KNEE_S, legs_only_below_hip: bool = True) -> np.ndarray:
    """이름 있는 두 경계 사이의 몸. 힙선 아래는 다리만 취해 꼬리를 뺀다."""
    rows = {"neck": z.neck, "shoulder": z.shoulder, "waist": z.waist,
            "hip": z.hip, "foot": z.foot,
            "knee": int(z.hip + knee_s * (z.foot - z.hip))}
    a, b = rows[start], rows[end]
    reg = band_mask(mask, a, b)
    if legs_only_below_hip and max(a, b) > z.hip:
        upper = band_mask(mask, min(a, b), min(max(a, b), z.hip))
        lower = band_mask(leg_region(mask, z.hip), z.hip, max(a, b))
        reg = upper | lower
    return reg
