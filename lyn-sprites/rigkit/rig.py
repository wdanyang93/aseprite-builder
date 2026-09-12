"""뼈대 정의 — 2D 컷아웃 리그.

뼈(bone): 이름, 부모, 기준 자세에서의 머리 위치(캔버스 절대 좌표).
슬롯(slot): 그림 파츠 하나를 뼈 하나에 붙인 것. anchor = 파츠 이미지 안에서 뼈 머리와 겹치는 점.

바인딩 규칙은 하나뿐이다: 파츠는 자기 뼈를 따라 통째로 회전·이동한다. 화소를 늘리거나 자르지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .spec import (ANKLE_DX, CENTER_X, ELBOW_DX, HIP_DX, JOINT_T, KNEE_DX,
                   SHOULDER_DX, SOLE_Y, WRIST_DX, t_to_y)


@dataclass
class Bone:
    name: str
    parent: str | None
    x: float
    y: float          # 기준 자세에서 뼈 머리의 캔버스 절대 좌표

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


def _b(name, parent, dx, t):
    return Bone(name, parent, CENTER_X + dx, t_to_y(JOINT_T[t]))


def build_skeleton(side: str = "front") -> dict[str, Bone]:
    """side='front' 은 좌우 팔다리가 둘 다 보이는 정면 리그.
    side='east' 는 측면(오른쪽 보기) — 뼈 구조는 같고 좌우 오프셋만 줄어든다."""
    k = 0.30 if side != "front" else 1.0   # 측면은 좌우 벌어짐이 원근으로 줄어든다
    bones: list[Bone] = [
        Bone("root", None, CENTER_X, t_to_y(JOINT_T["hips"])),
        _b("hips", "root", 0, "hips"),
        _b("chest", "hips", 0, "chest"),
        _b("head", "chest", 0, "head"),
    ]
    for s, sgn in (("l", -1), ("r", +1)):          # l = 화면 왼쪽, r = 화면 오른쪽
        bones += [
            Bone(f"upperarm_{s}", "chest", CENTER_X + sgn * SHOULDER_DX * k, t_to_y(JOINT_T["shoulder"])),
            Bone(f"forearm_{s}", f"upperarm_{s}", CENTER_X + sgn * ELBOW_DX * k, t_to_y(JOINT_T["elbow"])),
            Bone(f"hand_{s}", f"forearm_{s}", CENTER_X + sgn * WRIST_DX * k, t_to_y(JOINT_T["wrist"])),
            Bone(f"thigh_{s}", "hips", CENTER_X + sgn * HIP_DX * k, t_to_y(JOINT_T["hips"])),
            Bone(f"shin_{s}", f"thigh_{s}", CENTER_X + sgn * KNEE_DX * k, t_to_y(JOINT_T["knee"])),
            Bone(f"foot_{s}", f"shin_{s}", CENTER_X + sgn * ANKLE_DX * k, t_to_y(JOINT_T["ankle"])),
        ]
    # 꼬리 3단 — 몸 뒤, 엉덩이에서 내려간다
    bones += [
        Bone("tail1", "hips", CENTER_X, t_to_y(JOINT_T["hips"]) + 6),
        Bone("tail2", "tail1", CENTER_X - 40, t_to_y(JOINT_T["hips"]) + 60),
        Bone("tail3", "tail2", CENTER_X - 70, t_to_y(JOINT_T["hips"]) + 130),
    ]
    return {b.name: b for b in bones}


# 슬롯: (이름, 뼈, z). z 가 클수록 앞에 그린다.
# 'far' 쪽(화면 왼쪽 팔다리)을 몸통 뒤로, 'near'(오른쪽)를 앞으로 둔다.
SLOTS: list[tuple[str, str, int]] = [
    ("tail3",        "tail3",       -40),
    ("tail2",        "tail2",       -39),
    ("tail1",        "tail1",       -38),
    ("backpack",     "chest",       -30),
    ("upperarm_l",   "upperarm_l",  -20),
    ("forearm_l",    "forearm_l",   -19),
    ("hand_l",       "hand_l",      -18),
    ("thigh_l",      "thigh_l",     -12),
    ("shin_l",       "shin_l",      -11),
    ("foot_l",       "foot_l",      -10),
    ("thigh_r",      "thigh_r",      10),
    ("shin_r",       "shin_r",       11),
    ("foot_r",       "foot_r",       12),
    ("torso",        "chest",        20),
    ("hips_wear",    "hips",         21),
    ("head",         "head",         30),
    ("upperarm_r",   "upperarm_r",   40),
    ("forearm_r",    "forearm_r",    41),
    ("hand_r",       "hand_r",       42),
    ("bag",          "chest",        50),
]

SLOT_Z = {name: z for name, _, z in SLOTS}
SLOT_BONE = {name: bone for name, bone, _ in SLOTS}


def slot_order() -> list[str]:
    return [n for n, _, _ in sorted(SLOTS, key=lambda s: s[2])]
