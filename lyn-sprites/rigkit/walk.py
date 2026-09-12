"""걷기 사이클 — 뼈 각도 곡선. 옷과 무관하다.

이 파일이 이 프로젝트의 핵심 전환점이다. 예전 방식은 '원화 36장에서 걸음을 추출'했고,
그래서 원화가 바뀔 때마다 전부 다시 해야 했다. 여기서는 걸음을 우리가 정의하고,
원화는 그 위에 입힌다. 원화가 바뀌어도 걸음은 그대로다 = 통일감.

phi in [0,1) = 한 보행 주기(두 걸음). phi 0 에서 화면 오른쪽 다리가 최대로 앞.
각도 단위는 도(deg), 시계방향이 +.
"""
from __future__ import annotations

import math

from .spec import WALK_FRAMES

TAU = 2 * math.pi


def _sin(phi: float, amp: float, shift: float = 0.0) -> float:
    return amp * math.sin(TAU * (phi + shift))


def walk_pose(phi: float, *, stride: float = 1.0, side: str = "front") -> dict[str, float]:
    """한 위상에서의 뼈 회전각(도)과 root 오프셋을 돌려준다."""
    # 측면은 다리·팔 스윙이 그대로 보이고, 정면은 원근 때문에 회전이 작아 보인다.
    k = 1.0 if side != "front" else 0.42
    s = stride

    # 다리: phi 0 에서 오른다리가 최대로 앞 → cos 위상(shift 0.25).
    th_r = _sin(phi, 18 * s * k, 0.25)
    th_l = _sin(phi, 18 * s * k, 0.75)

    def knee(p):
        # 유각기(다리가 뒤→앞으로 올 때)에 많이 접힌다. 항상 <= 0 (뒤로만 꺾인다).
        return -max(0.0, _sin(p, 30 * s * k, 0.55)) - 4 * k

    kn_r, kn_l = knee(phi), knee(phi + 0.5)

    def foot(p, thigh, shin):
        # 발은 지면과 나란해지려 한다 → 다리 각을 되돌린다 + 뒤꿈치 들기
        return -(thigh + shin) * 0.55 + _sin(p, 12 * s * k, 0.40)

    ft_r = foot(phi, th_r, kn_r)
    ft_l = foot(phi + 0.5, th_l, kn_l)

    # 팔: 다리와 반대 위상
    ua_r = _sin(phi, 15 * s * k, 0.75)
    ua_l = _sin(phi, 15 * s * k, 0.25)
    fa_r = -abs(_sin(phi, 14 * s * k, 0.75)) - 6 * k
    fa_l = -abs(_sin(phi, 14 * s * k, 0.25)) - 6 * k

    # 몸통: 상하 바운스는 걸음의 2배 주기. 다리가 최대로 벌어진 착지(phi 0, 0.5)에서 가장 낮다.
    # +y 가 아래이므로 착지에서 +.
    bounce = abs(math.cos(TAU * phi)) * 7 * s
    lean = 2.0 * k
    twist = _sin(phi, 3.0 * k, 0.25)

    pose = {
        "root:y": bounce,
        "hips": twist,
        "chest": lean - twist * 0.6,
        "head": -lean - twist * 0.4 + _sin(phi, 1.5, 0.5),
        "thigh_r": th_r, "shin_r": kn_r, "foot_r": ft_r,
        "thigh_l": th_l, "shin_l": kn_l, "foot_l": ft_l,
        "upperarm_r": ua_r, "forearm_r": fa_r,
        "upperarm_l": ua_l, "forearm_l": fa_l,
        "hand_r": 0.0, "hand_l": 0.0,
        # 꼬리는 걸음을 반 박자 늦게 따라간다 (관성)
        "tail1": _sin(phi, 8, 0.13),
        "tail2": _sin(phi, 11, 0.05),
        "tail3": _sin(phi, 14, -0.03),
    }
    # 렌더러 규약: + 각도 = 화면 시계방향. 아래로 뻗은 뼈가 시계방향으로 돌면 화면 왼쪽으로 간다.
    # 동쪽(오른쪽 보기) 기준 '앞'은 화면 오른쪽이므로, 위에서 세운 '앞이 +' 값들을 뒤집는다.
    return {k: (v if k.startswith("root:") else -v) for k, v in pose.items()}


def walk_cycle(frames: int = WALK_FRAMES, **kw) -> list[dict[str, float]]:
    return [walk_pose(i / frames, **kw) for i in range(frames)]


def idle_pose(side: str = "front") -> dict[str, float]:
    return {k: 0.0 for k in walk_pose(0.0, side=side)}
