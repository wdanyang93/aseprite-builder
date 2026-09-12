"""회전 불변 신체 구역 — 구역은 화면이 아니라 **몸을 따라** 흐른다.

왜 다시 쓰는가
    zones.py 는 구역을 화면의 행(y)으로 잡았다. 서 있을 때는 맞지만 누우면
    신장이 폭이 되고, 무엇보다 상의/하의가 위아래가 아니라 좌우가 된다.
    엎드림 11장에서 세로 띠가 누운 몸을 가로질러 온몸이 칠해졌다.

    핵심은 축을 쓰는 것만이 아니다. **축의 어느 쪽이 머리인지**를 정해야
    상의와 하의가 뒤바뀌지 않는다.

좌표
    t = 0  머리 끝      t = 1  발 끝
    모든 경계는 t 로 표현된다. 자세가 회전해도 같은 값이다.

머리 쪽 판정
    홍채. 실측상 유일하게 뚜렷한 고채도 소형 덩어리다
    (T자세 (152,101,66) H13 822px / 측면 (142,95,68) H11 336px / 정본 g1 H14 98px).
    홍채가 안 잡히면 귀·머리카락의 밝은 영역으로 물러선다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

GARMENT_SAT, GARMENT_VAL = 22, 230
IRIS_SAT, IRIS_VAL = 100, 100
KNEE_T = 0.50          # 힙선~발끝 사이 비율


def _axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    cov = np.cov(pts.T)
    ev, evec = np.linalg.eigh(cov)
    return c, evec[:, int(np.argmax(ev))]


BRANCH_GAP = 0.012        # 단면이 갈라졌다고 볼 틈 (신장 대비)
BRANCH_MARGIN = 3         # 이 차이 미만이면 확신하지 않는다


def split_profile(mask: np.ndarray, c: np.ndarray, v: np.ndarray, n: int = 60) -> np.ndarray:
    """축을 n 등분해 각 수직 단면의 연결성분 수."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(float)
    p = (pts - c) @ v
    H = max(float(p.max() - p.min()), 1e-6)
    pn = (p - p.min()) / H
    q = pts @ np.array([-v[1], v[0]])
    out = np.ones(n)
    for i, a in enumerate(np.linspace(0, 1, n)):
        sel = np.abs(pn - a) < 0.75 / n
        if sel.sum() < 10:
            continue
        s = np.sort(q[sel])
        out[i] = 1 + int((np.diff(s) > H * BRANCH_GAP).sum())
    return out


def head_side(rgba: np.ndarray, mask: np.ndarray, c: np.ndarray, v: np.ndarray) -> int:
    """축의 어느 끝이 머리인가. +1 이면 v 방향이 머리.

    **가지가 갈라지는 쪽이 발이다.** 몸은 발 쪽으로 두 다리로 갈라지고 머리 쪽으로는
    갈라지지 않는다. 해부 구조라서 자세·의상·색과 무관하다.

    홍채는 주 판정에서 내렸다. 옷 입은 정답지 35장에서 갈색 가죽이 홍채와 같은
    (S>100, V>100) 대역에 들어와 34장이 뒤집혔다. 흰 하의 앵커가 옷에 가려진 것과
    같은 실패다 — **겉모습 앵커는 옷을 입히면 무너진다.** 가지 갈림은 34/35 였고
    틀린 한 장도 여유 1 로 스스로 낮은 확신을 드러냈다.
    """
    cmp = split_profile(mask, c, v)
    h = len(cmp) // 2
    v_side = int((cmp[h:] > 1).sum())      # v 쪽 절반에서 갈라진 단면 수
    o_side = int((cmp[:h] > 1).sum())
    if abs(v_side - o_side) >= BRANCH_MARGIN:
        return -1 if v_side > o_side else 1   # 많이 갈라진 쪽이 발

    # 확신이 낮을 때만 홍채로 물러선다 (소형 고채도 덩어리)
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    iris = mask & (hsv[..., 1] > IRIS_SAT) & (hsv[..., 2] > IRIS_VAL)
    ys, xs = np.nonzero(mask)
    proj = (np.stack([xs, ys], 1) - c) @ v
    if iris.sum() >= 20:
        iy, ix = np.nonzero(iris)
        ip = (np.stack([ix, iy], 1) - c) @ v
        return 1 if ip.mean() > proj.mean() else -1
    return -1 if v_side > o_side else 1


def head_side_confidence(mask: np.ndarray, c: np.ndarray, v: np.ndarray) -> tuple[int, int]:
    """(방향, 여유). 여유 >= BRANCH_MARGIN 이면 믿어도 된다.

    실측 (측면 34 + 측면 36 + 정면 36 = 106장, 정답은 전부 '위가 머리'):
        전체        측면 34/34, 36/36    정면 33/36
        여유>=3 만   95장 중 95장 정답 — 한 장도 안 틀렸다
    정면에서 여유가 줄어드는 이유는 고양이 귀가 머리 쪽 단면을 가르고
    두 다리가 정면에서는 붙어 보이기 때문이다. 그래서 정면은 시트 다수결로 간다.
    """
    cmp = split_profile(mask, c, v)
    h = len(cmp) // 2
    v_side = int((cmp[h:] > 1).sum())
    o_side = int((cmp[:h] > 1).sum())
    return (-1 if v_side > o_side else 1), abs(v_side - o_side)


def head_side_sheet(items: list[tuple[np.ndarray, np.ndarray]]) -> int:
    """한 시트(한 애니메이션) 전체의 머리 방향. **프레임마다 정하지 않는다.**

    목·무릎·발끝을 프레임마다 검출하지 않는 것과 같은 이유다 — 머리 방향은
    프레임의 성질이 아니라 시트의 성질이고, 프레임 검출은 잡음만 더한다.
    여유가 큰 프레임에 가중치를 준 다수결. 세 시트 모두 만장일치에 가깝게 맞았다
    (+34 / +36 / +30).
    """
    s = 0
    for rgba, mask in items:
        c, v = _axis(mask)
        if v[1] > 0:
            v = -v                      # v 를 화면 위로 맞춰 부호를 공유
        d, conf = head_side_confidence(mask, c, v)
        s += d * max(conf, 1)
    return 1 if s >= 0 else -1


def axis_t(rgba: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """화소별 t (0=머리, 1=발) 와 축 정보."""
    c, v = _axis(mask)
    sgn = head_side(rgba, mask, c, v)
    v = v * sgn                               # v 가 머리 방향
    ys, xs = np.nonzero(mask)
    p = (np.stack([xs, ys], 1) - c) @ v
    t = np.full(mask.shape, np.nan)
    t[ys, xs] = (p.max() - p) / max(p.max() - p.min(), 1e-6)   # 머리=0, 발=1
    return t, c, v, sgn


def width_profile(mask: np.ndarray, t: np.ndarray, bins: int = 80) -> np.ndarray:
    """t 를 따라간 단면 폭. 행 폭 프로파일의 회전 불변 판본."""
    ys, xs = np.nonzero(mask)
    tv = t[ys, xs]
    idx = np.clip((tv * (bins - 1)).astype(int), 0, bins - 1)
    return np.bincount(idx, minlength=bins).astype(float)


@dataclass
class Zones:
    neck: float
    waist: float
    hip: float
    foot: float = 1.0
    crown: float = 0.0
    axis_deg: float = 0.0
    measured: dict[str, bool] = None

    @property
    def knee(self) -> float:
        return self.hip + KNEE_T * (self.foot - self.hip)

    def to_dict(self) -> dict[str, Any]:
        return {"crown": 0.0, "neck": round(self.neck, 4), "waist": round(self.waist, 4),
                "hip": round(self.hip, 4), "knee": round(self.knee, 4), "foot": 1.0,
                "axis_deg": round(self.axis_deg, 1), "measured": self.measured or {}}


def find_zones(rgba: np.ndarray, mask: np.ndarray,
               profile: dict | None = None) -> tuple[Zones, np.ndarray] | None:
    t, c, v, sgn = axis_t(rgba, mask)
    ang = abs(np.degrees(np.arctan2(v[1], v[0])))
    ang = min(ang, 180 - ang)

    # 허리·힙선 = 하의 블롭을 t 로 투영
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    cloth = mask & (hsv[..., 1] < GARMENT_SAT) & (hsv[..., 2] > GARMENT_VAL)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cloth.astype(np.uint8), 8)
    cand = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < int(mask.sum() * 0.004):
            continue
        sel = lab == k
        tv = t[sel]
        lo, hi = float(np.nanmin(tv)), float(np.nanmax(tv))
        if 0.40 <= hi <= 0.70:                 # 골반 부근
            cand.append((int(st[k, cv2.CC_STAT_AREA]), lo, hi))
    if not cand:
        return None
    _, waist, hip = max(cand)

    # t 보정: 실루엣 끝에서 끝까지로 정규화하면 꼬리가 축을 늘인다.
    # 서 있는 36프레임에서 힙선 t = 0.5498 (sd 0.0091) 로 매우 안정적이므로
    # 머리끝(0)과 힙선(0.5498) 두 점으로 다시 잡는다. 꼬리가 어디까지 뻗든 무관해진다.
    HIP_T = 0.5498
    if hip > 1e-3:
        k = HIP_T / hip
        t = t * k
        waist *= k
        hip = HIP_T

    # 목·무릎·발끝은 프레임마다 검출하지 않는다.
    # t 가 힙선으로 보정된 이상 이들은 **캐릭터의 고정 비율**이고, 프레임 검출은
    # 잡음만 더한다. 엎드림에서 단면 기반 목 검출이 0.100~0.438 로 흩어진 것이 그 증거다.
    # PROFILE 값은 서 있는 36프레임 실측 (목 0.237, 무릎 0.775).
    neck = profile.get("neck", 0.237) if profile else 0.237
    return Zones(neck=neck, waist=waist, hip=hip, axis_deg=ang,
                 measured=dict(neck=False, waist=True, hip=True)), t


def zone_band(mask: np.ndarray, t: np.ndarray, a: float, b: float) -> np.ndarray:
    lo, hi = min(a, b), max(a, b)
    return mask & (t >= lo) & (t <= hi)
