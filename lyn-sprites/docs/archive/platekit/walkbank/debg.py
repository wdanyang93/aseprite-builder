"""배경 잔상 제거 — 팔과 몸통 사이 좁은 골에 남은 검은 배경.

원본 시트의 배경 제거가 좁은 골을 못 파냈다. 알파는 불투명인데 화소는 배경색이다.

무엇이 가르는가 — 실측
    잔상은 **순수 검정 RGB <= 8** 이고, 그림의 가장 어두운 화소는 13~30 이다.
    선화조차 갈색이다 — (97,70,63) (78,46,39) (92,70,61). 순수 검정은 그림에 없다.
    남쪽 36장에서 순수 검정은 몸의 2.37% (755~2269 px), 나머지와 겹치지 않는다.

    "어둡다"로는 못 가른다 (선화도 어둡다). "두껍다"로도 부정확하다 (눈·동공).
    배경색 자체가 유일하게 정확한 표지다. 이번에도 겉모습이 아니라 **출처**가 갈랐다.

테두리
    골 가장자리는 검정과 살색이 섞인 반투명 띠다. 순수 검정을 지운 뒤 그 이웃 중
    어두운 화소의 알파를 밝기에 비례해 낮춰 검은 테를 없앤다.
"""
from __future__ import annotations

import cv2
import numpy as np

BG_MAX   = 8       # 이 이하는 배경 (그림의 최소는 13)
FRINGE_L = 70      # 테두리로 볼 밝기 상한 (Lab L*)


def background_residue(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask & (rgba[..., :3].max(2) <= BG_MAX)


def clean(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    """잔상을 지우고 검은 테두리를 부드럽게. (정리된 rgba, 지운 화소수)"""
    mask = rgba[..., 3] > 128
    res = background_residue(rgba, mask)
    out = rgba.copy()
    if not res.any():
        return out, 0
    out[res, 3] = 0

    ring = (cv2.dilate(res.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~res & mask
    if ring.any():
        lightness = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB)[..., 0].astype(float)
        sel = ring & (lightness < FRINGE_L)
        if sel.any():
            k = np.clip(lightness[sel] / FRINGE_L, 0.0, 1.0)
            out[sel, 3] = (out[sel, 3].astype(float) * k).astype(np.uint8)
    return out, int(res.sum())
