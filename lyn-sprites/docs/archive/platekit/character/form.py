"""Silhouette form: outline + centerline, as a linear description of a figure.

Joint segmentation failed on real art for a structural reason: in a side view both legs are one
connected blob, so no amount of component labelling separates them, and the arms sit inside the
torso corridor and vanish entirely. Measured on a real E-facing frame the "leg" parts came out
814 px wide - the whole figure - because a principal-axis split cut two legs crosswise.

The medial axis does not have that problem. It is the set of points equidistant from two or more
boundary points, so a limb - long and thin - collapses to a single curve regardless of whether it
touches its neighbour, and the distance transform at each centerline point gives that limb's
thickness. The result is a graph: a trunk with branches, each branch a polyline with a width
profile. That is a form description you can measure, compare and retarget.

    silhouette -> outline (resampled, normalised)
               -> medial axis -> graph -> branch chains + width profiles

Everything here is derived from the alpha mask alone, so it works the same on painted art and on
a plain black gait reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np
from scipy import ndimage

OUTLINE_POINTS = 192          # resampled contour length; enough to keep fingers and ear tips
MIN_BRANCH_LEN = 0.12         # of figure height; shorter branches are skeleton noise
PRUNE_ITERATIONS = 8          # spur removal must converge, not run a fixed couple of passes
MAX_BRANCHES = 6              # 2 arms + 2 legs + tail + an ear is the most a figure yields
SMOOTH_SIGMA = 1.2


@dataclass
class Branch:
    """One centerline chain: a polyline plus the silhouette thickness along it."""
    name: str
    points: np.ndarray        # (n, 2) float, source-frame pixels
    widths: np.ndarray        # (n,) float, radius of the inscribed circle
    end_kind: str             # "tip" (free end) or "internal"

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "end_kind": self.end_kind,
                "length_px": round(self.length, 1),
                "mean_width_px": round(float(self.widths.mean()), 1) if len(self.widths) else 0.0,
                "points": [[round(float(x), 1), round(float(y), 1)] for x, y in self.points]}


@dataclass
class Form:
    """The linear description of one silhouette."""
    height: int
    width: int
    ground: list[float]              # foot contact point
    outline: np.ndarray              # (OUTLINE_POINTS, 2) normalised to height=1, pivot at ground
    branches: list[Branch] = field(default_factory=list)
    trunk: Branch | None = None
    area_ratio: float = 0.0
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"height": self.height, "width": self.width,
                "ground": [round(v, 1) for v in self.ground],
                "area_ratio": round(self.area_ratio, 4),
                "trunk": self.trunk.to_dict() if self.trunk else None,
                "branches": [b.to_dict() for b in self.branches],
                "outline_points": len(self.outline), "flags": self.flags}


# ------------------------------------------------------------- outline --

def outline_of(mask: np.ndarray, points: int = OUTLINE_POINTS) -> np.ndarray:
    """Largest contour, resampled to a fixed point count at uniform arc length.

    A fixed count makes two outlines directly comparable point-by-point, which is what lets one
    frame's silhouette be measured against another's without any correspondence search.
    """
    contours, _h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((points, 2), np.float32)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    closed = np.vstack([contour, contour[:1]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if arc[-1] <= 0:
        return np.zeros((points, 2), np.float32)
    targets = np.linspace(0.0, arc[-1], points, endpoint=False)
    out = np.stack([np.interp(targets, arc, closed[:, 0]),
                    np.interp(targets, arc, closed[:, 1])], axis=1)
    return out.astype(np.float32)


def normalise_outline(outline: np.ndarray, ground: Sequence[float], height: float) -> np.ndarray:
    """Ground pivot to origin, scaled so the figure is 1.0 tall. Scale/position invariant."""
    if height <= 0:
        return outline
    out = outline.copy()
    out[:, 0] -= float(ground[0])
    out[:, 1] -= float(ground[1])
    return out / float(height)


def start_at_topmost(outline: np.ndarray) -> np.ndarray:
    """Rotate the point order so index 0 is the highest point (the head).

    Without this, two outlines of the same pose compare badly purely because the contour tracer
    started somewhere else.
    """
    if len(outline) == 0:
        return outline
    start = int(np.argmin(outline[:, 1]))
    return np.roll(outline, -start, axis=0)


# ------------------------------------------------------------ centerline --

def _skeletonise(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True 1-px medial axis plus the inscribed-circle radius at each point.

    The classic *morphological* skeleton (iterated erode/open/subtract) is not a thin curve: on a
    real silhouette it produced 597 pixels of which 342 counted as junctions, i.e. a thick blob
    that no chain walk can follow. Zhang-Suen style thinning gives a connected 1-px curve, and
    ``medial_axis`` returns the distance map with it, which is the limb thickness for free.
    """
    from skimage.morphology import medial_axis

    skeleton, distance = medial_axis(mask.astype(bool), return_distance=True)
    return skeleton, distance


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)


def _prune(skel: np.ndarray, min_len: int) -> np.ndarray:
    """Remove short spurs, and only spurs.

    A spur is a short chain that runs from a free end *into a junction* - a bump on the outline.
    The trunk's own two ends are also free ends, so pruning every short chain from every endpoint
    eats the skeleton from both tips until nothing is left (observed: 0 branches on every figure).
    A chain is therefore only removed when it terminates at a junction.
    """
    out = skel.copy()
    for _ in range(PRUNE_ITERATIONS):
        counts = _neighbour_count(out)
        ends = list(zip(*np.nonzero(out & (counts == 1))))
        if not ends:
            break
        removed = False
        for y, x in ends:
            if not out[y, x]:
                continue
            chain = _walk_from(out, (y, x), max_len=min_len + 2)
            if len(chain) > min_len:
                continue                      # long enough to be a limb
            tail_counts = _neighbour_count(out)
            if tail_counts[chain[-1]] < 3:
                continue                      # ends in open space: this is a trunk tip, keep it
            for cy, cx in chain[:-1]:          # leave the junction pixel itself
                out[cy, cx] = False
            removed = True
        if not removed:
            break
    return out


def _walk_from(skel: np.ndarray, start: tuple[int, int], max_len: int = 100000,
               stop_at_junction: bool = True) -> list[tuple[int, int]]:
    """Follow a 1-px chain from an endpoint until it branches or ends."""
    counts = _neighbour_count(skel)
    chain = [start]
    visited = {start}
    current = start
    while len(chain) < max_len:
        y, x = current
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if not (0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]):
                    continue
                if skel[ny, nx] and (ny, nx) not in visited:
                    nxt = (ny, nx)
                    break
            if nxt:
                break
        if nxt is None:
            break
        chain.append(nxt)
        visited.add(nxt)
        current = nxt
        if stop_at_junction and counts[nxt] >= 3:
            break
    return chain


def centerline(mask: np.ndarray) -> tuple[Branch | None, list[Branch]]:
    """Medial axis as a trunk plus tip branches, each with a width profile.

    The trunk is the longest chain between two junctions or ends - for a standing figure that is
    head-to-foot. Every other chain that terminates in a free end is a branch: an arm, the other
    leg, the tail, an ear.
    """
    height = mask.shape[0]
    smooth = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), SMOOTH_SIGMA) > 0.5
    raw, distance = _skeletonise(smooth)
    skel = _prune(raw, max(4, int(height * MIN_BRANCH_LEN)))
    if not skel.any():
        return None, []
    counts = _neighbour_count(skel)
    endpoints = list(zip(*np.nonzero(skel & (counts == 1))))
    if not endpoints:
        return None, []

    chains: list[list[tuple[int, int]]] = []
    for endpoint in endpoints:
        chain = _walk_from(skel, endpoint)
        if len(chain) >= max(4, int(height * MIN_BRANCH_LEN)):
            chains.append(chain)
    if not chains:
        return None, []
    chains.sort(key=len, reverse=True)
    # Thinning still leaves spurs on a bumpy outline; without a cap a single silhouette produced
    # 12 "limbs". Keeping the longest few is what a limb actually is.
    chains = chains[: MAX_BRANCHES + 1]

    def to_branch(chain: list[tuple[int, int]], name: str) -> Branch:
        pts = np.array([[float(x), float(y)] for y, x in chain], np.float32)
        widths = np.array([float(distance[y, x]) for y, x in chain], np.float32)
        return Branch(name=name, points=pts, widths=widths, end_kind="tip")

    trunk = to_branch(chains[0], "trunk")
    branches = [to_branch(c, f"branch_{i}") for i, c in enumerate(chains[1:], start=1)]
    return trunk, branches


def _name_branches(trunk: Branch, branches: list[Branch], mask: np.ndarray) -> None:
    """Label branches by where their free tip sits relative to the figure.

    Deliberately geometric, not semantic: a tip low and far from the body midline is a leg or the
    tail, a tip at chest height is an arm, a tip above the head is an ear. The thickness profile
    separates the tail from a leg - a tail is thick to its very tip, a leg tapers to an ankle.
    """
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    mid_x = float(np.median(xs))
    for branch in branches:
        tip = branch.points[0]
        ty = tip[1] / height
        tip_width = float(branch.widths[:max(1, len(branch.widths) // 5)].mean())
        mean_width = float(branch.widths.mean()) if len(branch.widths) else 0.0
        taper = tip_width / max(1e-6, mean_width)
        if ty < 0.18:
            branch.name = "ear"
        elif ty > 0.62:
            branch.name = "tail" if taper > 0.75 else "leg"
        elif 0.25 <= ty <= 0.62:
            branch.name = "arm"
        else:
            branch.name = "limb"
        branch.name += "_L" if tip[0] < mid_x else "_R"


# --------------------------------------------------------------- form --

def extract_form(mask: np.ndarray, ground: Sequence[float] | None = None) -> Form:
    """Full linear description of one silhouette."""
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return Form(height=height, width=width, ground=[0.0, 0.0],
                    outline=np.zeros((OUTLINE_POINTS, 2), np.float32), flags=["empty"])
    if ground is None:
        from .walk import ground_pivot
        gx, gy, _c = ground_pivot(mask)
        ground = [gx, float(gy)]
    figure_height = float(ground[1] - float(ys.min()))
    outline = start_at_topmost(outline_of(mask))
    trunk, branches = centerline(mask)
    if trunk is not None:
        _name_branches(trunk, branches, mask)
    flags = []
    if trunk is None:
        flags.append("no_centerline")
    form = Form(height=height, width=width, ground=[float(ground[0]), float(ground[1])],
                outline=normalise_outline(outline, ground, figure_height),
                branches=branches, trunk=trunk,
                area_ratio=float(mask.sum()) / max(1.0, float(height * width)), flags=flags)
    return form


def outline_distance(a: Form, b: Form) -> float:
    """Mean point-to-point distance between two normalised outlines, in figure heights.

    Both are resampled to the same count, ground-anchored and height-normalised, so this is a pure
    shape difference: 0 means the same silhouette at any size or position.
    """
    if len(a.outline) != len(b.outline) or len(a.outline) == 0:
        return float("inf")
    best = float("inf")
    for shift in range(0, len(a.outline), 4):
        rolled = np.roll(b.outline, shift, axis=0)
        best = min(best, float(np.linalg.norm(a.outline - rolled, axis=1).mean()))
    return best


def limb_summary(form: Form) -> dict[str, Any]:
    """Branch lengths and thicknesses as fractions of figure height. The comparable numbers."""
    if form.trunk is None:
        return {}
    scale = max(1.0, form.trunk.length)
    out: dict[str, Any] = {"trunk_length_px": round(form.trunk.length, 1),
                           "trunk_mean_width_px": round(float(form.trunk.widths.mean()), 1)}
    grouped: dict[str, list[Branch]] = {}
    for branch in form.branches:
        grouped.setdefault(branch.name.rsplit("_", 1)[0], []).append(branch)
    for kind, items in grouped.items():
        out[kind] = {"count": len(items),
                     "length_over_trunk": [round(b.length / scale, 3) for b in items],
                     "mean_width_px": [round(float(b.widths.mean()), 1) for b in items]}
    return out
