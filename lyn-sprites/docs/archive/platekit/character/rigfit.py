"""기준 리그를 애니메이션 프레임에 대응시킨다 — 구간이 아니라 '사지' 단위로.

v1 이 실패한 이유 (실측)
    1. 좌표를 bbox 기준으로 정규화했다. 꼬리가 흔들리면 bbox 원점이 움직여
       부착 위치가 통째로 어긋난다. -> 리그 뿌리(가장 두꺼운 점)를 원점으로 바꾼다.
    2. 구간(segment) 1:1 로 맞췄다. 무릎이 굽으면 마디가 하나 더 생겨 같은 다리가
       2구간이 되고 대응이 깨진다. -> 뿌리에서 말단까지의 **경로** 하나를 사지로 본다.
       사지 개수는 마디 분할과 무관하다.
    3. 가중치가 제각각이라 반경 항(x6)이 비용을 지배했다. -> 각 항을 자기 전형값으로
       나눠 무차원화하고 동등 가중한다.

사지(Limb) = 리그 뿌리에서 잎(말단)까지의 경로. 꼬리·다리·팔·머리가 각각 하나씩.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .rig2 import Rig, RigPart

# 각 항의 전형적 크기. 비용을 무차원화하는 데만 쓰이며 캐릭터와 무관하다.
S_LEN, S_RAD, S_TIP, S_TAPER = 0.15, 0.02, 0.25, 0.30
MARGIN_MIN = 0.6         # 2등과의 차이가 이보다 작으면 모호한 대응으로 표시


@dataclass
class Limb:
    """뿌리에서 말단까지의 경로 하나."""
    lid: int
    part_ids: list[int]
    points: np.ndarray           # 이어 붙인 중심선 (뿌리 원점 기준)
    radius: np.ndarray
    name: str = ""

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    @property
    def tip(self) -> np.ndarray:
        return self.points[-1]

    @property
    def taper(self) -> float:
        n = max(1, len(self.radius) // 5)
        return float(self.radius[-n:].mean() / max(self.radius.mean(), 1e-6))

    def to_dict(self) -> dict[str, Any]:
        return {"lid": self.lid, "name": self.name, "parts": self.part_ids,
                "length": round(self.length, 4), "taper": round(self.taper, 3),
                "tip": [round(float(v), 4) for v in self.tip],
                "radius_mean": round(float(self.radius.mean()), 4)}


def limbs_of(rig: Rig) -> list[Limb]:
    """뿌리 부위의 시작점을 원점으로 삼아 잎까지의 경로를 모은다."""
    root = rig.root
    origin = root.points[0].astype(np.float64)
    children: dict[int, list[int]] = {}
    for p in rig.parts:
        children.setdefault(p.parent, []).append(p.id)
    leaves = [p.id for p in rig.parts if p.id not in children]
    out: list[Limb] = []
    for lid, leaf in enumerate(sorted(leaves)):
        chain, cur = [], leaf
        seen = set()
        while cur != -1 and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = rig.parts[cur].parent
        chain.reverse()
        pts = np.concatenate([rig.parts[i].points for i in chain], axis=0).astype(np.float64)
        rad = np.concatenate([rig.parts[i].radius for i in chain], axis=0).astype(np.float64)
        out.append(Limb(lid=lid, part_ids=chain, points=pts - origin, radius=rad))
    return out


def limb_cost(a: Limb, b: Limb) -> float:
    return (abs(a.length - b.length) / S_LEN
            + abs(float(a.radius.mean()) - float(b.radius.mean())) / S_RAD
            + float(np.linalg.norm(a.tip - b.tip)) / S_TIP
            + abs(a.taper - b.taper) / S_TAPER)


def fit(ref: Rig, tgt: Rig) -> dict[str, Any]:
    """헝가리안으로 전역 최적 배정. 절대 문턱 대신 2등과의 여유로 신뢰도를 판정한다.

    절대 문턱은 캐릭터·자세마다 적정값이 달라 감으로 고를 수밖에 없다. 반면
    "이 대응이 차선보다 얼마나 나은가"는 스스로 척도가 있다.
    """
    A, B = limbs_of(ref), limbs_of(tgt)
    C = np.array([[limb_cost(a, b) for b in B] for a in A])
    ri, ci = linear_sum_assignment(C)
    matches = []
    for i, j in zip(ri, ci):
        others = np.delete(C[i], j)
        margin = float(others.min() - C[i, j]) if others.size else float("inf")
        matches.append(dict(ref=A[i].lid, tgt=B[j].lid, name=A[i].name,
                            cost=round(float(C[i, j]), 2), margin=round(margin, 2),
                            confident=bool(margin >= MARGIN_MIN)))
    used = {m["tgt"] for m in matches}
    return {"ref_limbs": A, "tgt_limbs": B, "matches": matches,
            "unmatched_tgt": sorted(b.lid for b in B if b.lid not in used),
            "total_cost": round(float(C[ri, ci].sum()), 2),
            "confident": sum(m["confident"] for m in matches), "n": len(matches)}
