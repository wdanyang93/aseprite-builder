"""범용 파트 리그 — 캐릭터 고유 상수 없이 기준 자세에서 부위 트리를 뽑는다.

왜 새로 쓰는가
    기존 segment._name_chains / form._name_branches 는 인체(정확히는 Lyn)를 가정한다.
    "머리 tip y<=0.00, 귀 taper 0.18~0.30, 다리 tip y 0.94~1.00" 같은 규칙과
    MAX_BRANCHES=6 ("팔2+다리2+꼬리+귀") 이 박혀 있다. 몬스터는 다리가 넷일 수도,
    꼬리가 없을 수도, 날개가 있을 수도 있다. 성장해서 등신이 바뀌어도 y 규칙은 깨진다.

설계 원칙
    1. 부위의 개수·종류·위치를 가정하지 않는다.
    2. 뿌리는 "가장 두꺼운 곳"이다. 어떤 생물이든 몸통이 사지보다 두껍다.
       이것이 유일한 생물학적 가정이고, 인체에 한정되지 않는다.
    3. 모든 치수는 신장으로 정규화해 저장한다. 등신이 달라도 같은 리그가 된다.
    4. 이름은 기하에서 추론하지 않는다. 필요하면 캐릭터당 한 번 사람이 붙인다.
       (이름은 옷 슬롯을 묶을 때만 필요하고, 기하 자체에는 필요 없다)

    실루엣 -> 중간축 + 거리변환 -> 가장 두꺼운 점을 뿌리로 한 트리
          -> 마디(분기점·끝점) 사이 구간 = 부위 -> 중심선 + 반경 프로파일
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from .form import _neighbour_count, _prune, _skeletonise

SPUR_MIN = 0.06          # 신장 대비. 이보다 짧은 가지는 골격 잡음
RESAMPLE = 32            # 부위당 중심선 표본 수
SMOOTH_SIGMA = 0.012     # 신장 대비. 털 외곽의 톱니가 스퍼를 만들므로 크기에 비례해 뭉갠다
CLOSE_FRAC = 0.008       # 신장 대비 닫힘 반경. 털 사이 틈을 메운다


@dataclass
class RigPart:
    """부위 하나. 좌표는 전부 신장으로 나눈 값(무차원)."""
    id: int
    parent: int                  # -1 이면 뿌리
    points: np.ndarray           # (RESAMPLE, 2) 부착점 -> 말단
    radius: np.ndarray           # (RESAMPLE,) 각 점의 반경
    name: str = ""

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    @property
    def root(self) -> np.ndarray:
        return self.points[0]

    @property
    def tip(self) -> np.ndarray:
        return self.points[-1]

    @property
    def taper(self) -> float:
        """말단 반경 / 평균 반경. 꼬리처럼 끝까지 두꺼운 것과 사지를 가른다."""
        n = max(1, len(self.radius) // 5)
        return float(self.radius[-n:].mean() / max(self.radius.mean(), 1e-6))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "parent": self.parent, "name": self.name,
                "length": round(self.length, 4), "taper": round(self.taper, 3),
                "root": [round(float(v), 4) for v in self.root],
                "tip": [round(float(v), 4) for v in self.tip],
                "radius_mean": round(float(self.radius.mean()), 4),
                "radius_max": round(float(self.radius.max()), 4)}


@dataclass
class Rig:
    parts: list[RigPart]
    height_px: float             # 원본 픽셀 신장. 무차원 좌표를 되돌릴 때 쓴다
    origin_px: np.ndarray        # 정규화 기준점 (실루엣 bbox 좌상단)
    source: str = ""

    def children(self, pid: int) -> list[RigPart]:
        return [p for p in self.parts if p.parent == pid]

    @property
    def root(self) -> RigPart:
        return next(p for p in self.parts if p.parent == -1)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "height_px": round(self.height_px, 1),
                "origin_px": [round(float(v), 1) for v in self.origin_px],
                "parts": [p.to_dict() for p in self.parts]}


# ------------------------------------------------------------------ 추출 --

def _graph(skel: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))
    g: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in pts:
        nb = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dy or dx) and (y + dy, x + dx) in pts]
        g[(y, x)] = nb
    return g


def _nodes(skel: np.ndarray) -> tuple[np.ndarray, int]:
    """분기점·끝점을 마디로 삼되, 붙어 있는 분기점 덩어리는 하나로 묶는다.

    분기점은 보통 한 점이 아니라 2~5px 덩어리다. 그대로 두면 덩어리 안에서
    1px 짜리 구간이 수십 개 생긴다(측면에서 279개가 나온 원인).
    """
    counts = _neighbour_count(skel)
    junction = skel & (counts >= 3)
    n, lab = cv2.connectedComponents(junction.astype(np.uint8), 8)
    ends = skel & (counts == 1)
    ey, ex = np.nonzero(ends)
    node = lab.copy()
    for k, (y, x) in enumerate(zip(ey, ex), start=n):
        node[y, x] = k
    return node, n + len(ey)


def _segments(skel: np.ndarray, node: np.ndarray) -> list[list[tuple[int, int]]]:
    """마디와 마디 사이의 골격 구간을 모두 찾는다."""
    g = _graph(skel)
    segs: list[list[tuple[int, int]]] = []
    seen: set[frozenset] = set()
    for p in list(g):
        if node[p] == 0:
            continue
        for nb in g[p]:
            if frozenset((p, nb)) in seen:
                continue
            path = [p, nb]
            seen.add(frozenset((p, nb)))
            prev, cur = p, nb
            while node[cur] == 0:
                nxt = [q for q in g[cur] if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                seen.add(frozenset((path[-1], cur)))
                path.append(cur)
            if node[cur] != 0 and node[cur] == node[p] and len(path) < 6:
                continue                      # 같은 분기 덩어리 안의 짧은 고리
            segs.append(path)
    return segs


def _resample(path: np.ndarray, n: int) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    if d[-1] <= 0:
        return np.repeat(path[:1], n, axis=0)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, path[:, 0]), np.interp(t, d, path[:, 1])], axis=1)


def extract_rig(mask: np.ndarray, source: str = "", spur_min: float = SPUR_MIN) -> Rig:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("빈 마스크")
    height = float(ys.max() - ys.min() + 1)
    origin = np.array([float(xs.min()), float(ys.min())], np.float64)

    # 털·옷단의 톱니를 신장에 비례해 뭉갠다. 상수는 픽셀이 아니라 비율이라 크기에 무관하다.
    k = max(3, int(height * CLOSE_FRAC) | 1)
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, height * SMOOTH_SIGMA)) > 0.5

    skel, dist = _skeletonise(m)
    skel = _prune(skel, max(3, int(height * spur_min)))
    if not skel.any():
        raise ValueError("골격 없음")

    node, _ = _nodes(skel)
    sy, sx = np.nonzero(skel)
    ri = int(np.argmax(dist[sy, sx]))
    root_px = (int(sy[ri]), int(sx[ri]))

    segs = _segments(skel, node)
    min_len = height * spur_min
    segs = [s for s in segs
            if np.linalg.norm(np.diff(np.array(s, float), axis=0), axis=1).sum() >= min_len]
    if not segs:
        raise ValueError("의미 있는 구간 없음")

    # 뿌리 마디에서 시작해 폭 우선으로 부모를 정한다
    def dist_to_root(p):
        return abs(p[0] - root_px[0]) + abs(p[1] - root_px[1])
    remaining = list(segs)
    parts: list[RigPart] = []
    node_owner: dict[int, int] = {}
    start_node = int(node[root_px]) if node[root_px] else 0
    frontier = [start_node] if start_node else []
    if not frontier:
        frontier = [int(node[s[0]]) for s in sorted(remaining, key=lambda s: dist_to_root(s[0]))[:1]]
    node_owner[frontier[0]] = -1
    while remaining:
        progressed = False
        for s in list(remaining):
            a, b = int(node[s[0]]), int(node[s[-1]])
            for near, far, path in ((a, b, s), (b, a, s[::-1])):
                if near in node_owner:
                    pts = np.array([[float(x), float(y)] for y, x in path])
                    rad = np.array([float(dist[y, x]) for y, x in path])
                    P = _resample(pts, RESAMPLE)
                    R = np.interp(np.linspace(0, 1, RESAMPLE), np.linspace(0, 1, len(rad)), rad)
                    pid = len(parts)
                    parts.append(RigPart(id=pid, parent=node_owner[near],
                                         points=((P - origin) / height).astype(np.float32),
                                         radius=(R / height).astype(np.float32)))
                    node_owner.setdefault(far, pid)
                    remaining.remove(s)
                    progressed = True
                    break
            else:
                continue
            break
        if not progressed:
            s = min(remaining, key=lambda s: dist_to_root(s[0]))
            node_owner[int(node[s[0]])] = -1
    # 뿌리 마디에 여러 부위가 달리면 가장 두꺼운 것을 뿌리로 삼고 나머지를 그 자식으로
    roots = [q for q in parts if q.parent == -1]
    if len(roots) > 1:
        main = max(roots, key=lambda q: float(q.radius.mean()))
        for q in roots:
            if q.id != main.id:
                q.parent = main.id
    return Rig(parts=parts, height_px=height, origin_px=origin, source=source)
