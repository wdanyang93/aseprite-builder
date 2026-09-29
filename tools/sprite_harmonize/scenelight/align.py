"""Automatic alignment of the neutral frame onto the hand-lit reference, and label-free fitting regions.

The reference is usually a repaint of the same pose (slightly different proportions / pose). We find a
similarity transform by maximising silhouette IoU, then refine with dense optical flow (DIS). Every
reference pixel then knows which neutral pixel it corresponds to, so colour and shading can be compared
pixel by pixel. No body-part labels are needed: fitting regions are figure-relative grid cells split by
material, weighted by how well the two drawings line up locally.
"""
import cv2
import numpy as np

from .color import materials


def _bbox(a):
    ys, xs = np.nonzero(a > 0.5)
    return xs.min(), ys.min(), xs.max(), ys.max()


def _similarity(na, ra):
    """Scale / rotation / translation of neutral alpha onto reference alpha (bottom-centre anchored)."""
    nb, rb = _bbox(na), _bbox(ra)
    s0 = (rb[3] - rb[1]) / (nb[3] - nb[1])
    H, W = ra.shape
    # search at <= 420 px figure height for speed
    q = min(1.0, 420.0 / (rb[3] - rb[1]))
    raq = cv2.resize(ra, (int(W * q) + 1, int(H * q) + 1), interpolation=cv2.INTER_AREA) > 0.5
    Hq, Wq = raq.shape

    def M_of(s, dx, dy, rot, qq=1.0):
        c, si = np.cos(rot) * s, np.sin(rot) * s
        cx, cy = (nb[0] + nb[2]) / 2, nb[3]
        tx, ty = (rb[0] + rb[2]) / 2 + dx, rb[3] + dy
        M = np.array([[c, -si, tx - (c * cx - si * cy)], [si, c, ty - (si * cx + c * cy)]], np.float32)
        return M * qq

    def iou(p, full=False):
        if full:
            wa = cv2.warpAffine(na, M_of(*p), (W, H)) > 0.5
            R = ra > 0.5
        else:
            wa = cv2.warpAffine(na, M_of(*p, qq=q), (Wq, Hq)) > 0.5
            R = raq
        return (wa & R).sum() / max((wa | R).sum(), 1)

    Hr = rb[3] - rb[1]
    best = (-1, None)
    for s in s0 * np.linspace(0.92, 1.08, 9):
        for rot in np.radians([-3, -1.5, 0, 1.5, 3]):
            for dx in np.linspace(-0.05, 0.05, 11) * Hr:
                for dy in np.linspace(-0.036, 0.036, 7) * Hr:
                    v = iou((s, dx, dy, rot))
                    if v > best[0]:
                        best = (v, (s, dx, dy, rot))
    s, dx, dy, rot = best[1]
    best = (iou(best[1], True), best[1])
    step = 0.004 * Hr
    for _ in range(3):
        for ds in np.linspace(-0.01, 0.01, 5) * s0:
            for dr in np.radians([-0.5, 0, 0.5]):
                for ddx in (-step, 0, step):
                    for ddy in (-step, 0, step):
                        p = (s + ds, dx + ddx, dy + ddy, rot + dr)
                        v = iou(p, True)
                        if v > best[0]:
                            best = (v, p)
        s, dx, dy, rot = best[1]
    return M_of(*best[1]), float(best[0])


def _gray(rgba):
    return cv2.cvtColor((np.clip(rgba[..., :3] * rgba[..., 3:4], 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


class Pairs:
    """Neutral frame aligned into the reference frame + label-free fitting regions."""

    def __init__(self, neutral, ref, cell=0.12):
        self.neutral, self.ref = neutral, ref
        H, W = ref.shape[:2]
        self.shape = (H, W)
        M, self.iou_affine = _similarity(neutral[..., 3], ref[..., 3])
        wa = cv2.warpAffine(neutral, M, (W, H))
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(_gray(ref), _gray(wa), None)
        flow = cv2.GaussianBlur(flow, (0, 0), 5)
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        qx, qy = gx + flow[..., 0], gy + flow[..., 1]
        Mi = cv2.invertAffineTransform(M)
        self.mapx = (Mi[0, 0] * qx + Mi[0, 1] * qy + Mi[0, 2]).astype(np.float32)
        self.mapy = (Mi[1, 0] * qx + Mi[1, 1] * qy + Mi[1, 2]).astype(np.float32)
        self.neutral_w = self.warp(neutral)
        a, r = self.neutral_w[..., 3] > 0.5, ref[..., 3] > 0.5
        self.iou = float((a & r).sum() / max((a | r).sum(), 1))
        both = (self.neutral_w[..., 3] > 0.9) & (ref[..., 3] > 0.9)
        self.reliable = cv2.erode(both.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.mat = materials(self.neutral_w)
        line = cv2.dilate(self.mat["line"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.reliable &= ~line
        self._make_regions(cell)

    def warp(self, x):
        """neutral-space image -> reference frame"""
        return cv2.remap(x, self.mapx, self.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _make_regions(self, cell):
        """Grid cells (relative to figure height) x material; weight = local alignment quality."""
        ra = self.ref[..., 3]
        x0, y0, x1, y1 = _bbox(ra)
        Hf = float(y1 - y0 + 1)
        self.Hf = Hf
        c = max(8, int(round(cell * Hf)))
        # local agreement of edge structure between the two drawings
        gn = cv2.GaussianBlur(_gray(self.neutral_w).astype(np.float32), (0, 0), 1.5)
        gr = cv2.GaussianBlur(_gray(self.ref).astype(np.float32), (0, 0), 1.5)
        en = np.hypot(cv2.Sobel(gn, cv2.CV_32F, 1, 0), cv2.Sobel(gn, cv2.CV_32F, 0, 1))
        er = np.hypot(cv2.Sobel(gr, cv2.CV_32F, 1, 0), cv2.Sobel(gr, cv2.CV_32F, 0, 1))
        min_px = max(150, int((0.02 * Hf) ** 2))
        self.regions = []
        for yy in range(y0, y1 + 1, c):
            for xx in range(x0, x1 + 1, c):
                cellm = np.zeros(ra.shape, bool)
                cellm[yy:yy + c, xx:xx + c] = True
                inside = cellm & (ra > 0.5) & (self.neutral_w[..., 3] > 0.5)
                if inside.sum() < min_px:
                    continue
                a, b = en[inside], er[inside]
                ncc = float(((a - a.mean()) * (b - b.mean())).mean() / max(a.std() * b.std(), 1e-6))
                w = float(np.clip((ncc - 0.2) / 0.4, 0, 1))
                if w <= 0:
                    continue
                for mat in ("skin", "light"):
                    region = cellm & (self.mat[mat] > 0.5) & (ra > 0.5)
                    m = region & self.reliable & (self.mat[mat] > 0.7)
                    if m.sum() < min_px:
                        continue
                    self.regions.append({"name": f"{yy}_{xx}_{mat}", "mat": mat, "region": region, "mask": m, "w": w})
