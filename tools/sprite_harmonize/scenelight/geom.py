"""Silhouette geometry of one frame (design 2.5 / 3 / 6): figure height, depth, outward normals, contour arc index,
local width and thin parts. Everything comes from the frame's own alpha, so it follows every pose and direction.

Conventions: lengths are fractions of the figure height H (silhouette height, alpha > 0.5) unless a name ends in _px;
angles are degrees, 0 = screen right, 90 = screen up; normal vectors are (nx, ny, nz) in IMAGE axes (x right, y down,
z toward the viewer), so a light at screen angle phi is light_vec(phi) = (cos phi, -sin phi).

Normals (Lumo, Johnston 2002): the contour normal (smoothed alpha gradient) is known on the silhouette; the interior is
filled by repeated masked normalized convolution of those normals (4-neighbour relaxation = harmonic interpolation,
solved exactly at a coarse level and relaxed level by level up to the frame's resolution). On a disc this gives the
sphere's normals exactly; on a limb the cylinder's. In the middle of a limb the opposite contour normals cancel and the
2D part shrinks smoothly to 0 (= facing the viewer), so there is no seam on the medial axis and no fan streaks (the
distance-transform-gradient normals had both: full-length normals flipping on the axis, rays from every corner).
"""
import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.ndimage import distance_transform_edt

GEOM = {
    "edge_sig": 0.001,     # alpha smoothing for the contour normals (fraction of H, at least 0.6 px)
    "coarse_h": 200,       # figure height (px) of the exact sparse harmonic solve; finer levels relax from it
    "sor_iters": 20,       # relaxation sweeps per finer level
    "sor_omega": 1.6,
    "band": 0.03,          # arc index for pixels with depth <= band (and the outer anti-aliased ring)
    "arc_tau": 0.35,       # soft nearest contour point: distance tolerance = arc_tau x depth (at least 0.75 px)
    "arc_win": 1.2,        # ... over +-(arc_win x depth + 2) contour points around the nearest one
    "arc_samples": 12,     # at most this many samples per side of that window
    "thin_max": 0.035,     # thin part: local width below this (soft over +-15 %)
    "width_step": 1.12,    # geometric step of the local-width levels
    "ring_alpha": 0.02,    # outer anti-aliased ring = alpha above this outside the 0.5 contour
}


# ------------------------------------------------------------------ basics
def figure_height(alpha):
    """Silhouette height in px (rows with alpha > 0.5); 0 for an empty frame."""
    ys = np.nonzero(np.asarray(alpha).max(1) > 0.5)[0]
    return float(ys.max() - ys.min() + 1) if len(ys) else 0.0


def light_vec(deg):
    """Screen light direction (image axes, y down) for an angle in degrees (0 = right, 90 = up)."""
    r = np.radians(deg)
    return np.array([np.cos(r), -np.sin(r)], np.float32)


def crop_box(alpha, margin):
    """Slices of the bounding box of every non-transparent pixel plus margin px (None for an empty frame)."""
    ys, xs = np.nonzero(alpha > 0)
    if not len(ys):
        return None
    H, W = alpha.shape
    return (slice(max(0, ys.min() - margin), min(H, ys.max() + margin + 1)),
            slice(max(0, xs.min() - margin), min(W, xs.max() + margin + 1)))


def _edt(mask):
    """Exact Euclidean distance (px) of every pixel to the nearest pixel NOT in mask."""
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def signed_depth(alpha):
    """Distance to the alpha = 0.5 contour in px: positive inside, negative outside. The first ring on either side uses
    the pixel's own alpha (sub-pixel position of the anti-aliased edge)."""
    a = alpha.astype(np.float32)
    m = a > 0.5
    din, dout = _edt(m), _edt(~m)
    d = np.where(m, din - 0.5, -(dout - 0.5))
    ring = (m & (din <= 1.0)) | (~m & (dout <= 1.0))
    return np.where(ring, a - 0.5, d).astype(np.float32)


def contour_normals(alpha, H, p=None):
    """Unit outward normal of the smoothed alpha (image axes) everywhere it is defined, (nx, ny) HxWx2; 0 where flat."""
    p = dict(GEOM, **(p or {}))
    s = max(0.6, p["edge_sig"] * H)
    g = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), s)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    n = np.hypot(gx, gy)
    ok = n > 1e-6
    inv = np.where(ok, 1.0 / np.maximum(n, 1e-12), 0.0)
    return np.dstack([-gx * inv, -gy * inv]).astype(np.float32)


# ------------------------------------------------------------------ harmonic fill (Lumo)
def _cross_erode(m):
    return cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0


def _solve_exact(mask, fixed, V):
    """Discrete Laplace equation on the free pixels of mask (all their 4 neighbours are in mask), V fixed elsewhere."""
    free = mask & ~fixed
    H, W = mask.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.nonzero(free)
    n = len(ys)
    if n == 0:
        return V
    idx[ys, xs] = np.arange(n)
    rows, cols = [np.arange(n)], [np.arange(n)]
    vals = [np.full(n, 4.0)]
    rhs = np.zeros((n, V.shape[2]))
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = ys + dy, xs + dx
        j = idx[yy, xx]
        f = j >= 0
        rows.append(np.nonzero(f)[0]); cols.append(j[f]); vals.append(-np.ones(f.sum()))
        rhs[~f] += V[yy[~f], xx[~f]]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    sol = spl.spsolve(A.tocsc(), rhs)
    out = V.copy()
    out[ys, xs] = sol.reshape(n, -1)
    return out


def _relax(mask, fixed, V, iters, omega):
    """Red-black SOR sweeps of the same Laplace equation, starting from V (whole-array 4-neighbour averages)."""
    free = mask & ~fixed
    if iters <= 0 or not free.any():
        return V
    yy, xx = np.indices(mask.shape)
    nc = V.shape[2]
    wts = [np.repeat((omega * (free & (((yy + xx) & 1) == par))).astype(np.float32)[..., None], nc, 2)
           for par in (0, 1)]
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], np.float32)
    V = np.ascontiguousarray(V, np.float32)
    for _ in range(iters):
        for w in wts:
            d = cv2.subtract(cv2.filter2D(V, -1, k, borderType=cv2.BORDER_CONSTANT), V)
            V = cv2.add(V, cv2.multiply(d, w))
    return V


def _fill_outside(V, mask):
    """Values outside mask copied from the nearest pixel inside (for interpolating a coarse level up)."""
    if mask.all() or not mask.any():
        return V
    _, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
    return V[iy, ix]


def lumo_normals(alpha, H=None, p=None):
    """Outward screen-space normals HxWx3 (nx, ny, nz), image axes. Silhouette pixels carry the unit contour normal;
    the interior is the harmonic interpolation of it (|n_xy| shrinks smoothly toward the middle of every part);
    pixels outside alpha 0.5 carry the contour normal (nz = 0)."""
    p = dict(GEOM, **(p or {}))
    a = np.pad(alpha.astype(np.float32), 2)          # no silhouette pixel on the array border
    H = H or figure_height(alpha)
    if H <= 0:
        return np.zeros(alpha.shape + (3,), np.float32)
    levels = [1.0]
    while H * levels[-1] / 2 >= p["coarse_h"] and min(a.shape) * levels[-1] / 2 >= 8:
        levels.append(levels[-1] / 2)
    V = None
    prev_shape = None
    for q in reversed(levels):
        if q < 1.0:
            al = cv2.resize(a, (max(4, int(round(a.shape[1] * q))), max(4, int(round(a.shape[0] * q)))),
                            interpolation=cv2.INTER_AREA)
            al[[0, -1], :] = 0
            al[:, [0, -1]] = 0
        else:
            al = a
        m = al > 0.5
        fixed = m & ~_cross_erode(m)
        nc = contour_normals(al, H * q, p)
        if V is None:
            V0 = np.zeros(al.shape + (2,), np.float64)
            V0[fixed] = nc[fixed]
            V = _solve_exact(m, fixed, V0)
        else:
            up = cv2.resize(_fill_outside(V, prev_m).astype(np.float32), (al.shape[1], al.shape[0]),
                            interpolation=cv2.INTER_LINEAR).astype(np.float64)
            up[fixed] = nc[fixed]
            V = _relax(m, fixed, up, p["sor_iters"], p["sor_omega"])
        prev_m, prev_shape = m, al.shape
    m = a > 0.5
    nc = contour_normals(a, H, p)
    V = np.where(m[..., None], V, nc).astype(np.float32)
    nrm = np.hypot(V[..., 0], V[..., 1])
    V = V / np.maximum(nrm, 1.0)[..., None]           # |n_xy| <= 1 (it is, up to round-off)
    nz = np.sqrt(np.clip(1 - (V ** 2).sum(-1), 0, 1))
    nz = np.where(m | (nrm > 0), nz, 0)
    return np.dstack([V, nz])[2:-2, 2:-2].astype(np.float32)


def normal_angle(n):
    """Screen angle (deg, 0 = right, 90 = up, 0..360) of the 2D part of a normal field; undefined where it is ~0."""
    return (np.degrees(np.arctan2(-n[..., 1], n[..., 0])) % 360).astype(np.float32)


# ------------------------------------------------------------------ contour arc index
def contours(mask):
    """Silhouette contours (outer boundaries and holes) of mask: points (N, 2) int (x, y) in contour order, contour id
    per point, start and length per contour, and an image of global point indices (-1 elsewhere)."""
    cnts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    cnts = [c[:, 0, :] for c in cnts if len(c)]
    gi = -np.ones(mask.shape, np.int64)
    pts, cid, starts, lens, off = [], [], [], [], 0
    for i, c in enumerate(cnts):
        pts.append(c); cid.append(np.full(len(c), i)); starts.append(off); lens.append(len(c))
        gi[c[:, 1], c[:, 0]] = off + np.arange(len(c))
        off += len(c)
    if not pts:
        return {"pts": np.zeros((0, 2), np.int64), "cid": np.zeros(0, np.int64), "starts": np.zeros(0, np.int64),
                "lens": np.zeros(0, np.int64), "gi": gi}
    return {"pts": np.concatenate(pts).astype(np.int64), "cid": np.concatenate(cid), "starts": np.array(starts),
            "lens": np.array(lens), "gi": gi}


def arc_index(mask, depth, H, alpha=None, p=None):
    """Fractional arc index (global contour point index, float) for every pixel near the silhouette, NaN elsewhere,
    plus the contour structure. A pixel takes the soft nearest contour point: the weighted mean arc position of the
    contour points around its nearest one, weighted by how nearly they are as close as the nearest (tolerance
    arc_tau x depth). So the index sweeps smoothly through the wedge inside a concave corner instead of stamping one
    corner point over it (Voronoi blocks), while the two sides of a narrow part never mix (only arc neighbours of the
    nearest point are considered)."""
    p = dict(GEOM, **(p or {}))
    C = contours(mask)
    arc = np.full(mask.shape, np.nan, np.float32)
    if not len(C["pts"]):
        return arc, C
    bnd = C["gi"] >= 0
    d_e, (iy, ix) = distance_transform_edt(~bnd, return_indices=True)
    band = mask & (depth <= p["band"] * H)
    if alpha is not None:
        band |= ~mask & (alpha > p["ring_alpha"]) & (depth > -3)
    ys, xs = np.nonzero(band)
    g0 = C["gi"][iy[ys, xs], ix[ys, xs]]
    c = C["cid"][g0]
    st, ln = C["starts"][c], C["lens"][c]
    loc = g0 - st
    d0 = d_e[ys, xs].astype(np.float64)
    K = np.minimum(np.ceil(p["arc_win"] * d0 + 2), (ln - 1) // 2).astype(np.int64)
    step = np.maximum(1, -(-K // p["arc_samples"]))
    tau2 = 2 * np.maximum(0.75, p["arc_tau"] * d0) ** 2
    px, py = C["pts"][:, 0], C["pts"][:, 1]
    sw = np.ones(len(ys))                             # k = 0: the nearest point itself (weight 1)
    sk = np.zeros(len(ys))
    for j in range(1, int(p["arc_samples"]) + 1):
        k = j * step
        act = k <= K
        if not act.any():
            break
        for sgn in (1, -1):
            kk = sgn * k[act]
            g = st[act] + (loc[act] + kk) % ln[act]
            dist = np.hypot(px[g] - xs[act], py[g] - ys[act])
            w = np.exp(-(dist - d0[act]) ** 2 / tau2[act])
            sw[act] += w
            sk[act] += w * kk
    arc[ys, xs] = st + (loc + sk / sw) % ln
    return arc, C


# ------------------------------------------------------------------ local width / thin parts
def local_width(mask, depth, p=None):
    """Local width in px: diameter of the largest disc inside the silhouette that covers the pixel (the medial axis'
    depth propagated over its inscribed disc), from openings of the silhouette by discs of geometric radii."""
    p = dict(GEOM, **(p or {}))
    width = np.where(mask, 2 * np.clip(depth, 0.25, None), 0).astype(np.float32)
    dmax = float(depth[mask].max()) if mask.any() else 0.0
    r = 0.5
    while r <= dmax:
        E = mask & (depth >= r)
        ys, xs = np.nonzero(E)
        if not len(ys):
            break
        m = int(np.ceil(r)) + 2
        y0, y1 = max(0, ys.min() - m), min(mask.shape[0], ys.max() + m + 1)
        x0, x1 = max(0, xs.min() - m), min(mask.shape[1], xs.max() + m + 1)
        dE = _edt(~E[y0:y1, x0:x1])
        cov = mask[y0:y1, x0:x1] & (dE <= r)
        sub = width[y0:y1, x0:x1]
        sub[cov] = np.maximum(sub[cov], 2 * r)
        r *= p["width_step"]
    return width


def thin_map(width_h, p=None):
    """Soft thin-part weight from the local width (fraction of H): 1 below 0.85 x thin_max, 0 above 1.15 x thin_max."""
    p = dict(GEOM, **(p or {}))
    t = p["thin_max"]
    return np.clip((1.15 * t - width_h) / (0.3 * t), 0, 1).astype(np.float32)


# ------------------------------------------------------------------ everything for one frame
def analyze(alpha, p=None, arc=True):
    """Geometry of one frame from its alpha (HxW, 0..1). Returns a dict of full-frame maps:
    H (px), mask (alpha > 0.5), depth_px / depth (signed, fraction of H), n (HxWx3 normals), angle (deg),
    arc (fractional global contour index near the silhouette, NaN elsewhere) and contour (structure),
    width_px / width (local width), thin (soft thin-part weight). Outer anti-aliased ring pixels take the width of
    the nearest silhouette pixel."""
    p = dict(GEOM, **(p or {}))
    alpha = np.asarray(alpha, np.float32)
    Hfull, Wfull = alpha.shape
    H = figure_height(alpha)
    out = {"H": H, "mask": alpha > 0.5}
    box = crop_box(alpha, 4)
    if H <= 0 or box is None:
        z = np.zeros(alpha.shape, np.float32)
        out.update(depth_px=z - 1, depth=z - 1, n=np.zeros(alpha.shape + (3,), np.float32), angle=z,
                   arc=np.full(alpha.shape, np.nan, np.float32), contour=contours(out["mask"]), width_px=z, width=z,
                   thin=z)
        return out
    a = alpha[box]
    m = a > 0.5
    d = signed_depth(a)
    n = lumo_normals(a, H, p)
    w = local_width(m, d, p)
    ring = ~m & (a > p["ring_alpha"])
    if ring.any() and m.any():
        _, (iy, ix) = distance_transform_edt(~m, return_indices=True)
        w = np.where(ring, w[iy, ix], w)

    def full(x, fill):
        f = np.full(alpha.shape + x.shape[2:], fill, x.dtype)
        f[box] = x
        return f
    d_full = full(d, np.float32(-1e3))
    # far outside the crop the depth is only "far": recompute exactly there is not needed by any consumer
    out.update(depth_px=d_full, depth=d_full / H, n=full(n, np.float32(0)), width_px=full(w, np.float32(0)))
    out["angle"] = normal_angle(out["n"])
    out["width"] = out["width_px"] / H
    out["thin"] = np.where(out["mask"] | full(ring, False), thin_map(out["width"], p), 0).astype(np.float32)
    if arc:
        ac, C = arc_index(m, d, H, a, p)
        y0, x0 = box[0].start, box[1].start
        C = dict(C)
        C["pts"] = C["pts"] + np.array([x0, y0])
        gi = -np.ones(alpha.shape, np.int64)
        gi[box] = C["gi"]
        C["gi"] = gi
        out["arc"], out["contour"] = full(ac, np.float32(np.nan)), C
    return out
