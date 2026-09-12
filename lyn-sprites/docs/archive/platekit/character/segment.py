"""Centerline part segmentation. The replacement for connected-component limb splitting.

Connected components cannot separate a side view's two legs: they touch, so they are one blob, and
splitting that blob along its principal axis cut both legs crosswise - the "near leg upper" part
came out 93 % of the frame width. Colour cannot separate the tail either: measured on the real
sheet, tail, skin and hair all sit at H 9-12, S 34-120, V 149-254. There is no chromatic gap.

Geometry works. The medial axis of the whole figure yields a stable set of branches, and each
branch's free tip position plus its width taper says what it is. Measured across the eight frames
of one real sheet, at a 10 % minimum branch length:

    tip y/H ~0.00, taper ~0.25   head (thin at the crown)
    tip y/H ~0.20, taper ~0.18   ear
    tip x/W at 0.00 or 1.00      tail  (reaches the frame edge, stays thick: taper 0.44-0.91)
    tip y/H >0.90                foot  (a leg tapers to an ankle: taper 0.56-0.74)

Joints are then the local minima of the inscribed-circle radius along a branch, because a
silhouette narrows at a knee and an ankle. That gives upper/lower/foot without a hand-placed
skeleton and without assuming a fixed number of joints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from .form import _neighbour_count, _prune, _skeletonise, _walk_from

MIN_BRANCH_FRACTION = 0.10     # of figure height
EDGE_TOUCH = 0.04              # tip within this fraction of the frame edge = reaches the edge
TAIL_MIN_TAPER = 0.40          # a tail stays thick to its tip; a limb does not
TAIL_MIN_Y = 0.50              # and it hangs below mid-height
FOOT_MIN_Y = 0.88              # tip below this fraction of the height is a foot
HEAD_MAX_Y = 0.10
EAR_MAX_Y = 0.30
EAR_MAX_TAPER = 0.35           # ears are pointed; a thick tip at that height is not an ear
ARM_MIN_Y = 0.38               # a hand rests around hip height
ARM_MAX_Y = 0.75
ARM_MIN_TAPER = 0.50           # a hand is chunky, unlike an ear tip
JOINT_SMOOTH = 5
JOINT_MIN_GAP = 0.18           # joints must be at least this far apart along the branch
SEAM_OVERLAP_FRACTION = 0.016  # cuts overlap by this share of figure height
DEPTH_MIN_GAP = 6.0           # L* gap below this cannot decide near from far
OUTLINE_MARGIN = 2            # px kept outside the silhouette; the rest is alpha 0
ALPHA_MARGIN_VALUE = 40       # the margin ring is faint, not opaque
MERGED_SPLIT_MIN_PX = 2000    # below this a blob is too small to be two limbs
MERGED_MIN_SHARE = 0.06       # the shaded limb held 12-14 % on the real sheet
BIMODAL_BINS = 16
BIMODAL_RANGE = (150.0, 255.0)
TEXTURE_WINDOW = 9            # px box for the local standard deviation
# Texture mismatch was applied as a multiplier on the distance, which let a distant chain win
# whenever its texture matched better - regions fragmented and three frames lost the tail entirely.
# It is now an ADDITIVE penalty in pixels, capped, so geometry stays in charge and texture only
# breaks ties between chains at comparable distance.
TEXTURE_WEIGHT = 0.0          # multiplicative form: disabled, it fragmented the regions
TEXTURE_PENALTY_PX = 14.0     # maximum distance-equivalent penalty for a texture mismatch
TEXTURE_SCALE = 4.0           # keeps the ratio finite for smooth chains


@dataclass
class Chain:
    """One centerline branch with its width profile and the parts cut from it."""
    name: str
    points: np.ndarray          # (n, 2) x, y in frame pixels, tip first
    widths: np.ndarray          # (n,) inscribed radius
    tip: list[float]
    taper: float
    joints: list[int] = field(default_factory=list)   # indices into points

    @property
    def arc_length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arc_length": round(self.arc_length, 1),
                "tip": [round(v, 1) for v in self.tip], "taper": round(self.taper, 3),
                "mean_width": round(float(self.widths.mean()), 1),
                "joints": [int(j) for j in self.joints],
                "joint_points": [[round(float(self.points[j][0]), 1),
                                  round(float(self.points[j][1]), 1)] for j in self.joints]}


def extract_chains(mask: np.ndarray, min_fraction: float = MIN_BRANCH_FRACTION) -> list[Chain]:
    """Medial-axis branches of the whole figure, named by geometry."""
    height, width = mask.shape
    smooth = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2) > 0.5
    raw, distance = _skeletonise(smooth)
    minimum = max(4, int(height * min_fraction))
    skeleton = _prune(raw, minimum)
    if not skeleton.any():
        return []
    counts = _neighbour_count(skeleton)
    chains: list[Chain] = []
    for endpoint in zip(*np.nonzero(skeleton & (counts == 1))):
        walk = _walk_from(skeleton, endpoint)
        if len(walk) < minimum:
            continue
        points = np.array([[float(x), float(y)] for y, x in walk], np.float32)
        widths = np.array([float(distance[y, x]) for y, x in walk], np.float32)
        tip_width = float(widths[: max(1, len(widths) // 5)].mean())
        taper = tip_width / max(1e-6, float(widths.mean()))
        chains.append(Chain(name="", points=points, widths=widths,
                            tip=[float(points[0][0]), float(points[0][1])], taper=taper))
    chains.sort(key=lambda c: -c.arc_length)
    _name_chains(chains, height, width)
    for chain in chains:
        chain.joints = find_joints(chain)
    return chains


def _name_chains(chains: list[Chain], height: int, width: int) -> None:
    """Geometric naming, with every threshold read off the measured corpus.

    Colour is unusable here - tail, skin and hair all sit at H 9-12, S 34-120, V 149-254 - so the
    rules are tip position, tip-to-mean width ratio, and whether the tip reaches a frame edge.
    Measured across the eight frames of one sheet:

        head   tip y<=0.00          taper 0.25-0.27   (the crown is thin)
        ear    tip y 0.12-0.26      taper 0.18-0.30   (pointed)
        tail   tip x at 0.00/1.00   taper 0.42-0.60   y 0.56-0.73  (thick to the tip)
        leg    tip y 0.94-1.00      taper 0.55-0.74   (tapers to an ankle)
        arm    tip y 0.55-0.57      taper 0.60-0.66   x 0.77-0.90, NOT at an edge

    The arm rule had it backwards: it required the tip to touch a frame edge, but a hand rests at
    hip height *inside* the silhouette. Five of eight frames' arms were therefore falling through
    to the generic "limb" bucket, which is why the review saw no arm on frames that plainly have
    one.
    """
    for chain in chains:
        tx, ty = chain.tip[0] / width, chain.tip[1] / height
        at_edge = tx <= EDGE_TOUCH or tx >= 1.0 - EDGE_TOUCH
        if ty <= HEAD_MAX_Y:
            chain.name = "head"
        elif ty >= FOOT_MIN_Y:
            chain.name = "leg"
        elif at_edge and ty >= TAIL_MIN_Y and chain.taper >= TAIL_MIN_TAPER:
            chain.name = "tail"
        elif ty <= EAR_MAX_Y and chain.taper <= EAR_MAX_TAPER:
            chain.name = "ear"
        elif ARM_MIN_Y <= ty <= ARM_MAX_Y and not at_edge and chain.taper >= ARM_MIN_TAPER:
            chain.name = "arm"
        else:
            chain.name = "limb"

    # A figure has one tail and at most two ears. Extra claimants are demoted by how badly they
    # fit, which stopped one sheet reporting three ears and no tail.
    for name, limit in (("tail", 1), ("ear", 2)):
        claimants = [c for c in chains if c.name == name]
        if len(claimants) <= limit:
            continue
        if name == "tail":
            claimants.sort(key=lambda c: -c.arc_length)
        else:
            claimants.sort(key=lambda c: c.taper)
        for chain in claimants[limit:]:
            chain.name = "limb"

    groups: dict[str, list[Chain]] = {}
    for chain in chains:
        groups.setdefault(chain.name, []).append(chain)
    for name, items in groups.items():
        if len(items) < 2 or name in ("head", "tail"):
            continue
        items.sort(key=lambda c: c.tip[0])
        for i, chain in enumerate(items):
            chain.name = f"{name}_{'L' if i == 0 else 'R'}" if len(items) == 2 else f"{name}_{i}"


def find_joints(chain: Chain, max_joints: int = 2) -> list[int]:
    """Local minima of the width profile: a knee and an ankle narrow the silhouette.

    Returned tip-first, so for a leg chain walked from the foot the first joint is the ankle and
    the second is the knee.
    """
    widths = chain.widths
    if len(widths) < 3 * JOINT_SMOOTH:
        return []
    kernel = np.ones(JOINT_SMOOTH) / JOINT_SMOOTH
    smooth = np.convolve(widths, kernel, mode="same")
    candidates: list[tuple[float, int]] = []
    margin = max(2, int(len(smooth) * 0.12))
    for i in range(margin, len(smooth) - margin):
        if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]:
            neighbourhood = smooth[max(0, i - margin):i + margin + 1]
            depth = float(neighbourhood.max() - smooth[i])
            candidates.append((-depth, i))
    candidates.sort()
    chosen: list[int] = []
    gap = int(len(smooth) * JOINT_MIN_GAP)
    for _depth, index in candidates:
        if all(abs(index - c) >= gap for c in chosen):
            chosen.append(index)
        if len(chosen) >= max_joints:
            break
    return sorted(chosen)


# ------------------------------------------------------------- cutting --

def local_texture(rgba: np.ndarray) -> np.ndarray:
    """Local standard deviation of luminance. Fur is noisy, skin is smooth.

    This is the discriminator lightness cannot provide. The tail and the shaded far leg sit at the
    same L* - and which of the two is darker even flips between frames (measured: tail minus leg
    ranged from -25 to +23) - so a purely photometric split merges them, which is exactly what the
    review saw on frames 5 and 8. Texture does not flip: the tail measured 1.6-2.0x the local
    standard deviation of a leg on six of eight frames.
    """
    grey = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(grey, (TEXTURE_WINDOW, TEXTURE_WINDOW))
    mean_sq = cv2.blur(grey * grey, (TEXTURE_WINDOW, TEXTURE_WINDOW))
    return np.sqrt(np.maximum(0.0, mean_sq - mean * mean))


def _chain_texture(chain: Chain, texture: np.ndarray, mask: np.ndarray) -> float:
    """The chain's own texture, sampled from the pixels its centreline actually sits on."""
    values = []
    height, width = mask.shape
    for (x, y), radius in zip(chain.points, chain.widths):
        r = max(1, int(radius * 0.5))
        y0, y1 = max(0, int(y) - r), min(height, int(y) + r + 1)
        x0, x1 = max(0, int(x) - r), min(width, int(x) + r + 1)
        window = texture[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
        if window.size:
            values.append(float(np.median(window)))
    return float(np.median(values)) if values else 0.0


def chain_regions(mask: np.ndarray, chains: Sequence[Chain],
                  rgba: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Assign every foreground pixel to its nearest centerline point's owning segment.

    This is what makes two touching legs separable: ownership follows the centerline, not
    connectivity, so a pixel between the legs goes to whichever centerline is closer.
    """
    height, width = mask.shape
    labels: list[tuple[str, np.ndarray]] = []
    for chain in chains:
        bounds = [0] + list(chain.joints) + [len(chain.points)]
        names = _segment_names(chain.name, len(bounds) - 1)
        for i, name in enumerate(names):
            segment = chain.points[bounds[i]:bounds[i + 1]]
            if len(segment):
                labels.append((name, segment))
    if not labels:
        return {}
    all_points = np.vstack([pts for _n, pts in labels])
    owner = np.concatenate([[i] * len(pts) for i, (_n, pts) in enumerate(labels)])

    # Texture-weighted ownership. Pure nearest-centreline assignment sends tail pixels to a leg and
    # leg pixels to the tail wherever the two lie close, because distance alone cannot tell them
    # apart. Multiplying the distance by a texture-mismatch factor keeps fur with fur.
    texture = local_texture(rgba) if rgba is not None else None
    chain_tex = np.zeros(len(labels), np.float32)
    if texture is not None:
        per_chain = {c.name: _chain_texture(c, texture, mask) for c in chains}
        for i, (name, _pts) in enumerate(labels):
            base = name
            while base and base not in per_chain:
                base = base.rsplit("_", 1)[0] if "_" in base else ""
            chain_tex[i] = per_chain.get(base, 0.0)

    ys, xs = np.nonzero(mask)
    pixels = np.stack([xs, ys], axis=1).astype(np.float32)
    nearest = np.empty(len(pixels), np.int64)
    step = 20000
    for start in range(0, len(pixels), step):
        block = pixels[start:start + step]
        cost = np.sqrt(((block[:, None, 0] - all_points[None, :, 0]) ** 2
                        + (block[:, None, 1] - all_points[None, :, 1]) ** 2))
        if texture is not None:
            pixel_tex = texture[ys[start:start + step], xs[start:start + step]].astype(np.float32)
            mismatch = np.abs(pixel_tex[:, None] - chain_tex[None, owner])
            normalised = mismatch / (TEXTURE_SCALE + chain_tex[None, owner])
            cost = cost + TEXTURE_PENALTY_PX * np.clip(normalised, 0.0, 1.0)
        nearest[start:start + step] = owner[np.argmin(cost, axis=1)]
    out: dict[str, np.ndarray] = {}
    for i, (name, _pts) in enumerate(labels):
        region = np.zeros((height, width), bool)
        sel = nearest == i
        region[ys[sel], xs[sel]] = True
        if region.any():
            out[name] = out.get(name, np.zeros((height, width), bool)) | region
    return out


def _segment_names(base: str, count: int) -> list[str]:
    if count <= 1:
        return [base]
    if base.startswith("leg"):
        parts = ["foot", "lower", "upper"]
    elif base.startswith("arm"):
        parts = ["hand", "lower", "upper"]
    elif base.startswith("tail"):
        parts = ["tip", "mid", "base"]
    else:
        parts = [f"seg{i}" for i in range(count)]
    return [f"{base}_{parts[i]}" if i < len(parts) else f"{base}_seg{i}" for i in range(count)]


def classify_depth(rgba: np.ndarray, regions: dict[str, np.ndarray],
                   kinds: Sequence[str] = ("leg", "arm")) -> dict[str, Any]:
    """Near limb versus far limb, from shading. The signal I had been ignoring.

    The art marks depth with value: the far limb is shaded, the near one is lit. Measured on the
    real sheet, per-region median L*:

        frame 3  leg_L 227  leg_R 203      frame 4  leg_L 203  leg_R 228
        frame 6  leg_L 203  leg_R 228      frame 7  leg_L 219  leg_R 225

    A 25-level gap, and it flips between frames 3 and 4 exactly as a walk cycle should. Without it
    there is no way to say which limb is which when they overlap, no way to set the compositing
    order, and no way to notice that two frames have their arms swapped - which is what the review
    saw between frames 4 and 5.
    """
    lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB)[..., 0]
    out: dict[str, Any] = {}
    for kind in kinds:
        # Group by LIMB, not by segment. Splitting on the last underscore turned one leg's
        # foot/lower/upper into three "limbs" and then declared the thigh near and the shin far -
        # a single leg cannot be both. The limb identity is the chain name, which is the part
        # before any segment suffix.
        groups: dict[str, list[str]] = {}
        for name in regions:
            if not name.startswith(kind):
                continue
            head = _limb_identity(name, kind)
            groups.setdefault(head, []).append(name)
        if len(groups) < 2:
            for head, names in groups.items():
                out[head] = {"depth": "unknown", "reason": "단일 체인 - 두 팔다리가 겹쳐 있다",
                             "lightness": _group_lightness(lab, regions, names)}
            continue
        scored = {head: _group_lightness(lab, regions, names) for head, names in groups.items()}
        brightest = max(scored, key=lambda k: scored[k])
        darkest = min(scored, key=lambda k: scored[k])
        gap = scored[brightest] - scored[darkest]
        for head in groups:
            if gap < DEPTH_MIN_GAP:
                out[head] = {"depth": "unknown", "reason": f"밝기 차 {gap:.1f} < {DEPTH_MIN_GAP}",
                             "lightness": round(scored[head], 1)}
            else:
                out[head] = {"depth": "near" if head == brightest else "far",
                             "lightness": round(scored[head], 1), "gap": round(gap, 1)}
    return out


SEGMENT_SUFFIXES = ("upper", "lower", "foot", "hand", "base", "mid", "tip")


def _limb_identity(name: str, kind: str) -> str:
    """leg_L_upper -> leg_L, leg_upper -> leg, arm_hand -> arm."""
    parts = name.split("_")
    while len(parts) > 1 and parts[-1] in SEGMENT_SUFFIXES:
        parts.pop()
    return "_".join(parts) if parts else kind


def _group_lightness(lab_l: np.ndarray, regions: dict[str, np.ndarray], names: Sequence[str]) -> float:
    """Median L* over a limb's segments, weighted by nothing: the thigh dominates by area anyway,
    so the upper segment is used when present because it is the least occluded."""
    preferred = [n for n in names if n.endswith("_upper")] or list(names)
    values = np.concatenate([lab_l[regions[n]] for n in preferred if regions[n].any()])
    return float(np.median(values)) if values.size else 0.0


def split_merged_limb(rgba: np.ndarray, region: np.ndarray) -> dict[str, np.ndarray] | None:
    """Split one merged limb blob into near and far by shading.

    When two legs overlap the medial axis gives a single chain, so geometry is exhausted. Shading
    is not: the lightness histogram of the merged region is bimodal at exactly the same two values
    on every frame of the real sheet - peaks at L* 202 and 228, with the shaded (far) limb holding
    12-14 % of the pixels. Otsu is useless here because the distribution is so lopsided (it chose
    153-160, below both peaks); the threshold is taken between the two peaks instead.
    """
    if int(region.sum()) < MERGED_SPLIT_MIN_PX:
        return None
    lightness = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB)[..., 0]
    values = lightness[region]
    hist, edges = np.histogram(values, bins=BIMODAL_BINS, range=BIMODAL_RANGE)
    peaks = [i for i in range(1, BIMODAL_BINS - 1)
             if hist[i] > hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > values.size * 0.04]
    if len(peaks) < 2:
        return None
    peaks.sort(key=lambda i: -hist[i])
    low, high = sorted(peaks[:2])
    valley = low + int(np.argmin(hist[low:high + 1]))
    threshold = float(edges[valley])
    if edges[high] - edges[low] < DEPTH_MIN_GAP:
        return None
    far = region & (lightness < threshold)
    near = region & (lightness >= threshold)
    share = float(far.sum()) / max(1, int(region.sum()))
    if not (MERGED_MIN_SHARE <= share <= 1.0 - MERGED_MIN_SHARE):
        return None
    # keep only the largest component of each side; shading noise elsewhere is not a limb
    far = _largest_component(far)
    near = _largest_component(near)
    if far is None or near is None:
        return None
    return {"far": far, "near": near, "threshold": threshold, "far_share": round(share, 3)}


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    n, labels, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[biggest, cv2.CC_STAT_AREA]) < MERGED_SPLIT_MIN_PX // 4:
        return None
    return labels == biggest


def free_form_alpha(rgba: np.ndarray, mask: np.ndarray, margin: int = OUTLINE_MARGIN) -> np.ndarray:
    """Keep the silhouette plus a thin margin; everything else is alpha 0.

    A rectangular crop is the wrong shape for this character: the tail sweeps sideways and the
    figures overlap, so a box that contains one figure also contains pieces of its neighbours, and
    a box drawn to avoid the neighbours clips the tail, hand or toes. Cutting along the outline
    with a 2-3 px margin has neither problem.
    """
    grown = mask
    if margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1,) * 2)
        grown = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    out = rgba.copy()
    if out.shape[2] == 3:
        out = np.dstack([out, np.zeros(out.shape[:2], np.uint8)])
    out[..., 3] = np.where(grown, out[..., 3], 0)
    # the margin ring has no colour of its own; carry the nearest silhouette colour outward so
    # resampling later does not pull background into the edge
    if margin > 0:
        ring = grown & ~mask
        if ring.any():
            filled = cv2.inpaint(cv2.cvtColor(out[..., :3], cv2.COLOR_RGB2BGR),
                                 ring.astype(np.uint8), 3, cv2.INPAINT_NS)
            rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
            out[..., :3] = np.where(ring[..., None], rgb, out[..., :3])
            out[..., 3] = np.where(ring, ALPHA_MARGIN_VALUE, out[..., 3])
    return out


def torso_region(mask: np.ndarray, regions: dict[str, np.ndarray]) -> np.ndarray:
    """Whatever the limbs and head did not claim. The torso has no branch of its own.

    The medial axis produces branches for things that stick out; the trunk is the body itself, so
    it never appears as a tip-terminated chain and every torso pixel was being absorbed by
    whichever limb centreline happened to be nearest. That is why the decomposition showed the
    torso painted as leg or tail.
    """
    claimed = np.zeros_like(mask)
    for name, region in regions.items():
        if name.startswith(("leg", "arm", "tail", "head", "ear", "limb")):
            claimed |= region
    return mask & ~claimed


def resolve_limbs(rgba: np.ndarray, regions: dict[str, np.ndarray],
                  chains: Sequence[Chain]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Give every limb a near/far identity, splitting merged blobs by shading when needed.

    Geometry handles the frames where the limbs are apart; shading handles the rest. Both arms and
    legs are bimodal at the same two lightness values on the real sheet (L* 202 and 228), with the
    shaded limb holding 35-44 % of an arm blob and 51-52 % of a leg blob.
    """
    out = dict(regions)
    info: dict[str, Any] = {}
    for kind in ("leg", "arm"):
        identities = sorted({_limb_identity(n, kind) for n in regions if n.startswith(kind)})
        if kind == "arm":
            identities += sorted({_limb_identity(n, "limb") for n in regions if n.startswith("limb")})
        if not identities:
            continue
        if len(identities) >= 2:
            depth = classify_depth(rgba, regions, kinds=(kind,))
            for identity in identities:
                info[identity] = dict(depth.get(identity, {}), method="geometry")
            continue
        identity = identities[0]
        merged = np.zeros_like(next(iter(regions.values())))
        members = [n for n in regions if _limb_identity(n, kind) == identity]
        for name in members:
            merged |= regions[name]
        result = split_merged_limb(rgba, merged)
        if result is None:
            info[identity] = {"depth": "unknown", "method": "merged_unsplit",
                              "reason": "명암이 이봉분포가 아니다"}
            continue
        for name in members:
            out.pop(name, None)
        out[f"{kind}_near"] = result["near"]
        out[f"{kind}_far"] = result["far"]
        info[f"{kind}_near"] = {"depth": "near", "method": "shading",
                                "threshold": round(result["threshold"], 1)}
        info[f"{kind}_far"] = {"depth": "far", "method": "shading",
                               "threshold": round(result["threshold"], 1),
                               "far_share": result["far_share"]}
    return out, info


def segment_by_centerline(rgba: np.ndarray, mask: np.ndarray) -> tuple[dict[str, np.ndarray], list[Chain], dict[str, Any]]:
    """Full centerline segmentation: chains, joints, one region per segment, near/far, torso."""
    chains = extract_chains(mask)
    regions = chain_regions(mask, chains, rgba)
    height, width = mask.shape
    regions, depth = resolve_limbs(rgba, regions, chains)
    torso = torso_region(mask, regions)
    if torso.any():
        regions["torso"] = torso
    report = {
        "depth": depth,
        "frame_size": [width, height],
        "chains": [c.to_dict() for c in chains],
        "segments": {name: {"pixels": int(region.sum()),
                            "width_fraction": round(float(
                                (np.nonzero(region.any(axis=0))[0].max()
                                 - np.nonzero(region.any(axis=0))[0].min() + 1) / width), 3)}
                     for name, region in regions.items()},
        "leg_chains": sum(1 for c in chains if c.name.startswith("leg")),
        "tail_found": any(c.name == "tail" for c in chains),
        "torso_px": int(regions.get("torso", np.zeros(1, bool)).sum()),
        "unclaimed_px": int((mask & ~np.logical_or.reduce(
            [r for r in regions.values()] or [np.zeros_like(mask)])).sum()),
    }
    return regions, chains, report
