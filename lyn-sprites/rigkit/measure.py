"""전신 원화에서 이 캐릭터의 실제 비율을 잰다.

리그의 관절 t 값을 일반적인 인체 비율로 가정하면, 캐릭터(특히 애니 비율)와 어긋난다.
턴어라운드 전신은 한 장 안에서 비율이 맞으므로, 거기서 재서 리그를 캐릭터에 맞춘다.

재는 것은 실루엣에서 확실히 보이는 것만이다 — 목(가장 좁은 곳)과 발목. 나머지는 추정하지 않고
그 두 기준 사이를 표준 비율로 나눈다. 모르는 것을 아는 척하지 않는다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .fit import opaque_box


def center_runs(mask_row: np.ndarray, cx: int) -> tuple[int, int] | None:
    """그 행에서 중심선 cx 를 포함하는 불투명 구간. 꼬리·팔처럼 떨어진 덩어리를 배제한다."""
    xs = np.nonzero(mask_row)[0]
    if len(xs) == 0:
        return None
    splits = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
    for run in splits:
        if run[0] <= cx <= run[-1]:
            return int(run[0]), int(run[-1])
    best = min(splits, key=lambda r: abs((r[0] + r[-1]) / 2 - cx))
    return int(best[0]), int(best[-1])


def width_profile(img: Image.Image, thr: int = 32) -> tuple[np.ndarray, int, int, int]:
    """(행별 중심 덩어리 너비, 머리끝 y, 발바닥 y, 중심 x)"""
    a = np.array(img.convert("RGBA"))[..., 3] > thr
    x0, y0, x1, y1 = opaque_box(img, thr)
    # 중심선은 상체(머리~가슴)에서 잡는다 — 꼬리가 닿지 않는 구간
    upper = a[y0:y0 + int((y1 - y0) * 0.30)]
    ux = np.nonzero(upper.any(axis=0))[0]
    cx = int((ux.min() + ux.max()) / 2) if len(ux) else (x0 + x1) // 2
    w = np.zeros(y1 - y0, float)
    for i, y in enumerate(range(y0, y1)):
        r = center_runs(a[y], cx)
        w[i] = 0 if r is None else r[1] - r[0] + 1
    return w, y0, y1, cx


def neck_t(img: Image.Image, lo: float = 0.08, hi: float = 0.38) -> float:
    """목 = 머리 아래에서 가장 좁은 곳. 정수리 0, 발바닥 1 기준의 t."""
    w, y0, y1, _ = width_profile(img)
    n = len(w)
    a, b = int(n * lo), int(n * hi)
    seg = w[a:b]
    seg = np.where(seg > 0, seg, seg.max() + 1)
    return (a + int(np.argmin(seg))) / n


def measure_body(img: Image.Image) -> dict:
    """전신에서 잰 것 + 표준 비율로 나눈 관절 t. 잰 것과 나눈 것을 구분해 돌려준다."""
    t_neck = neck_t(img)
    # 목 아래(1 - t_neck)를 표준 비율로 나눈다. 아래 비는 일반 인체 기준이고,
    # 캐릭터마다 다를 수 있어 '측정'이 아니라 '배분'이라고 표시한다.
    below = 1.0 - t_neck
    joints = {
        "head_top": 0.0,
        "head": round(t_neck, 4),
        "chest": round(t_neck + below * 0.075, 4),
        "shoulder": round(t_neck + below * 0.080, 4),
        "elbow": round(t_neck + below * 0.255, 4),
        "wrist": round(t_neck + below * 0.420, 4),
        "hips": round(t_neck + below * 0.440, 4),
        "knee": round(t_neck + below * 0.700, 4),
        "ankle": round(t_neck + below * 0.945, 4),
        "sole": 1.0,
    }
    heads_tall = 1.0 / t_neck if t_neck > 0 else 0.0
    return {"measured": {"neck_t": round(t_neck, 4), "heads_tall": round(heads_tall, 2)},
            "derived_joints": joints}
