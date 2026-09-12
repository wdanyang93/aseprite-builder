"""정면(남쪽) 걷기 도구 — 꼬리 분리와 몸통 정규화.

꼬리를 빼야 하는 이유: u 는 각 t 띠의 폭으로 정규화된다. 꼬리가 한쪽으로 뻗으면
그 띠의 중심과 폭이 통째로 밀려 **몸 전체의 u 가 거짓이 된다.** 측면에서는 꼬리가
진행 방향 뒤로 뻗어 영향이 작지만, 정면에서는 꼬리가 좌우 축에 그대로 얹힌다.
"""
from __future__ import annotations
import sys
sys.path.insert(0,'/home/claude/handoff/code')
import cv2
import numpy as np
from platekit.character.lineart import lineart_mask, interior_cells
from platekit.character.zones2 import find_zones

FLOOR_T   = 0.85      # 이 아래까지 안 닿으면 다리가 아니다 (실측 꼬리 0.763±0.010)
LEG_T     = 0.92
TAIL_MIN  = 1200
CELL_MIN  = 400
L_RANGE   = range(100, 205, 5)

def tail_mask(rgba, mask):
    """꼬리 화소. 선 임계를 프레임마다 올려가며 **꼬리와 다리가 갈라지는 첫 값**을 쓴다.

    고정 임계는 프레임마다 선 농도가 달라 25/36 에서 멈췄다. 탐색으로 36/36.
    """
    z, t = find_zones(rgba, mask)
    h, _ = mask.shape
    below = mask & (t > z.hip)
    tot = int(below.sum())
    for l in L_RANGE:
        lines = lineart_mask(rgba, mask, l)
        lab, n = interior_cells(mask, lines)
        tail = np.zeros_like(mask); ta = 0; la = 0
        has_leg = False
        for c in range(1, n):
            sel = (lab == c) & below
            px = int(sel.sum())
            if px < CELL_MIN:
                continue
            ymax = np.nonzero(sel)[0].max() / h
            if ymax < FLOOR_T and px >= TAIL_MIN:
                tail |= sel; ta += px
            elif ymax >= LEG_T:
                has_leg = True; la += px
        if ta and has_leg and la > 0.40 * tot and ta < 0.16 * mask.sum():
            return tail, z, t, l
    return None


# ── 옷 입은 정면 프레임용 ────────────────────────────────────────────────
# 흰 하의 앵커가 옷에 가려지므로 힙선 보정을 못 한다. 정면에서는 꼬리가 축을
# 늘이지 않아 보정계수가 k = 1.0277 ± 0.0053 로 거의 1이다 (누드 36장 실측).
# 그래서 **상수 보정 + 캐릭터 프로파일**로 대신한다.
FRONT_K = 1.0277
PROFILE = dict(neck=0.237, waist=0.4895, hip=0.5498, knee=0.775, foot=1.0)


def front_t(rgba, mask):
    from platekit.character.zones2 import axis_t
    t, c, v, sgn = axis_t(rgba, mask)
    return t * FRONT_K


def tail_from_t(rgba, mask, t, hip=None):
    """구역 검출 없이 꼬리만. 옷 입은 프레임에서 쓴다."""
    hip = PROFILE["hip"] if hip is None else hip
    h, _ = mask.shape
    below = mask & (t > hip)
    tot = int(below.sum())
    for l in L_RANGE:
        lines = lineart_mask(rgba, mask, l)
        lab, n = interior_cells(mask, lines)
        tail = np.zeros_like(mask); ta = 0; la = 0; has_leg = False
        for c in range(1, n):
            sel = (lab == c) & below
            px = int(sel.sum())
            if px < CELL_MIN:
                continue
            ymax = np.nonzero(sel)[0].max() / h
            if ymax < FLOOR_T and px >= TAIL_MIN:
                tail |= sel; ta += px
            elif ymax >= LEG_T:
                has_leg = True; la += px
        if ta and has_leg and la > 0.40 * tot and ta < 0.16 * mask.sum():
            return tail, l
    return None


def tail_bright(rgba, mask, t, tmin=0.55, amin=0.010, ymax=0.92):
    """옷 입은 프레임의 꼬리 — 흰 털이 어두운 레깅스 위에서 확 갈린다.

    선 칸 방식은 옷 입은 36장에서 0/36 이었다 (치마단·벨트가 칸을 잘게 쪼갠다).
    옷을 입으면 오히려 **밝기**가 통한다: 골반 아래에서 밝고 저채도인 큰 덩어리는
    꼬리뿐이다. 36/36, 꼬리 4.7% ± 1.6 (누드 5.5% ± 0.5 와 일치).
    """
    h = mask.shape[0]
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    br = mask & (t > tmin) & (hsv[..., 2] > 190) & (hsv[..., 1] < 55)
    br = cv2.morphologyEx(br.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(br.astype(np.uint8), 8)
    cand = [(int(st[k, cv2.CC_STAT_AREA]), k) for k in range(1, n)
            if st[k, cv2.CC_STAT_AREA] >= int(mask.sum() * amin)
            and (st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]) / h < ymax]
    if not cand:
        return np.zeros_like(mask)
    out = lab == max(cand)[1]
    return cv2.dilate(out.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 & mask
