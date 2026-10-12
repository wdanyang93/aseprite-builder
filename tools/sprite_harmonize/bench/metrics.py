"""One metrics module for the bench (unifies R_methods common.score, R_audit common.eval_vs, dir2 rim/sunrim.py).

Conventions
  * angles: degrees, 0 = screen right, 90 = screen UP. Image rows grow DOWN, so a screen direction `deg` is the image
    vector (cos deg, -sin deg) and the angle of an image vector (nx, ny) is atan2(-ny, nx).
  * H = figure height (alpha > 0.5 rows); every length is a fraction of H unless it says px.
  * all frames are float RGBA (straight alpha, sRGB) in the SAME geometry as their truth.

Canonical numbers (score):
  dEa   low-pass (sigma 1 % H) Lab distance, mean over reliable non-dark pixels (both alphas > 0.9, eroded, no lines,
        input not dark)                                           - colour + light error
  dEst  the same after removing each colour region's constant Lab offset (regions = k-means of the INPUT's colours)
                                                                  - structure only (the two refs disagree in grade)
  rL    local low-pass L* correlation over grid cells (12 % H) x colour region
  sun / shd   mean |L*out - L*truth| on the sun- / shadow-facing silhouette band, depth 0-2 and 2-4 px (px at the
        reference figure height 1392, scaled with the frame, at least 1 px), averaged over body parts (polygons) where the
        frame has them, else over screen-space normal-angle bins relative to the light azimuth
  con   sun-minus-shadow edge L* (0-2 px) of the output (con_ref: of the truth)
  det   band-pass (0.1 %..0.8 % H) L* correlation output vs INPUT  - drawn detail kept
  lnC   line contrast kept (median ring-minus-line L*, output / input) on thin dark strokes (thin_lines)
  hp    high-pass L* std ratio output / truth (1 = same detail contrast)
  alpha_same (bitwise), lit_lines (interior thin-stroke pixels > 1 L* lighter than the input), nan_out, nan_keys
NaN guard: a value that cannot be computed is NaN and is LISTED in `nan_keys`; NaN in an output image is `nan_out`.
Nothing is silently dropped.

Compat replicas (numerically the research scripts' own definitions, used by T0 to reproduce their reported numbers):
  score_rmethods (R_methods common.score), eval_audit (R_audit common.eval_vs), practice_valid / score_practice /
  edge_practice (R_practice scenemodel), sunrim_bands (dir2 sunrim.evaluate band means).
"""
import numpy as np
import cv2

from scenelight.color import lowpass, materials, to_lab, srgb_to_linear

from . import parts as PARTS

REF_H = 1392.0          # figure height (px) the edge bands are defined at (Lyn right neutral)
BANDS_PX = ((0, 2), (2, 4))
SUN_COS = 0.4           # n.s > 0.4 = sun side, < -0.4 = shadow side (sunrim convention)


# ------------------------------------------------------------------ conventions
def unit(deg):
    """Screen direction (deg, 0 = right, 90 = up) as an image-space vector (x right, y DOWN)."""
    t = np.radians(deg)
    return np.array([np.cos(t), -np.sin(t)], np.float32)


def angle_of(nx, ny):
    """Screen angle (deg, 0 = right, 90 = up) of image-space vectors (x right, y down), in (-180, 180]."""
    return np.degrees(np.arctan2(-np.asarray(ny), np.asarray(nx)))


def ang_diff(a, b):
    """Smallest absolute difference of two angles in degrees (0..180)."""
    return np.abs((np.asarray(a, np.float64) - np.asarray(b, np.float64) + 180.0) % 360.0 - 180.0)


def fig_h(alpha):
    ys = np.nonzero(alpha.max(1) > 0.5)[0]
    return float(ys.max() - ys.min() + 1) if len(ys) else float("nan")


def L_of(x):
    return to_lab(np.clip(x[..., :3], 0, 1))[..., 0]


# ------------------------------------------------------------------ geometry and line art
def edge_geometry(alpha, scale=1.0):
    """Depth (px) from the silhouette and the outward normal (image coords) from the blurred distance transform
    (sunrim.geometry, sigma 2.5 px x scale, at least 1.5 px)."""
    a = (alpha > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(a, cv2.DIST_L2, 5)
    gy, gx = np.gradient(cv2.GaussianBlur(d, (0, 0), 2.5 * max(scale, 0.6)))
    nn = np.hypot(gx, gy) + 1e-6
    return d, -gx / nn, -gy / nn


def thin_lines(rgba, Hf=None):
    """Thin dark strokes: black-hat of the slightly blurred L* (closing ~0.6 % H wide, so darker areas wider than that
    are not lines), hysteresis: seed pixels >= 20 L* darker than the closed surroundings and below L* 60, grown over
    pixels >= 10 L* darker. Dark cloth / dark skin are areas, not lines; brown (L* ~50) lines are still lines.
    (Bench metric only; the tool takes its line art from Adobe: scenelight.parts.load_lines.)"""
    a = rgba[..., 3] > 0.5
    Hf = Hf or fig_h(rgba[..., 3])
    L = cv2.GaussianBlur(L_of(rgba), (0, 0), 0.6)
    k = int(max(3, round(0.006 * Hf))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(L, cv2.MORPH_CLOSE, ker) - L
    strong = (bh > 20) & a & (L < 60)
    weak = (bh > 10) & a
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab]


def _regions(lab_in, mask, k=6, seed=0):
    """Colour regions of the INPUT drawing: k-means over (L, 1.5a, 1.5b) of masked pixels -> label image (-1 outside)."""
    from sklearn.cluster import KMeans
    out = np.full(mask.shape, -1, np.int32)
    if mask.sum() < 10 * k:
        return out
    f = np.stack([lab_in[..., 0], 1.5 * lab_in[..., 1], 1.5 * lab_in[..., 2]], -1)[mask]
    rng = np.random.default_rng(seed)
    sub = f[rng.choice(len(f), min(len(f), 12000), replace=False)]
    k = min(k, len(np.unique(np.round(sub, 1), axis=0)))     # flat dolls have fewer colours than k
    km = KMeans(k, n_init=2, random_state=seed).fit(sub)
    out[mask] = km.predict(f)
    return out


# ------------------------------------------------------------------ edge (sun / shadow) metric
def edge_bands(imgs, frame, light_deg, labels=None, Hf_ref=REF_H, bands=BANDS_PX):
    """Mean L* of every image in `imgs` (name -> RGBA) on the sun / shadow side, per group and depth band.
    Groups = body parts from `labels` (LIMB parts present) or, without labels, normal-angle bins relative to the light
    (4 bins of 33 deg on each side). Returns {group: {side: {band: {name: L}}}}, mode."""
    alpha = frame[..., 3]
    Hf = fig_h(alpha)
    s = Hf / Hf_ref
    d, nx, ny = edge_geometry(alpha, s)
    u = unit(light_deg)
    face = nx * u[0] + ny * u[1]
    ok = alpha > 0.5
    for v in imgs.values():
        ok = ok & (v[..., 3] > 0.5)
    Ls = {k: L_of(v) for k, v in imgs.items()}
    groups = {}
    if labels is not None and any((labels == PARTS.NAMES.index(p) + 1).any() for p in PARTS.LIMB):
        mode = "parts"
        for p in PARTS.LIMB:
            groups[p] = labels == PARTS.NAMES.index(p) + 1
    else:
        mode = "angle"
        rel = (angle_of(nx, ny) - light_deg + 180.0) % 360.0 - 180.0
        lim = np.degrees(np.arccos(SUN_COS))          # 66.4 deg
        ed = np.linspace(-lim, lim, 5)
        for i in range(4):
            groups[f"sun{i}"] = (rel >= ed[i]) & (rel < ed[i + 1])
            r2 = (rel + 360.0) % 360.0 - 180.0         # angle relative to the anti-light direction
            groups[f"shd{i}"] = (r2 >= ed[i]) & (r2 < ed[i + 1])
    res = {}
    for g, gm in groups.items():
        res[g] = {}
        for side, sm in (("sun", face > SUN_COS), ("shadow", face < -SUN_COS)):
            if mode == "angle" and not g.startswith("sun" if side == "sun" else "shd"):
                continue
            sel = gm & ok & sm
            res[g][side] = {}
            for lo, hi in bands:
                blo, bhi = max(lo * s, lo / 2), max(hi * s, hi / 2)      # bands never thinner than 1 px
                mm = sel & (d > blo) & (d <= bhi)
                res[g][side][f"L_{lo}_{hi}"] = {k: (float(L[mm].mean()) if mm.sum() >= 5 else np.nan) for k, L in Ls.items()}
    return res, mode


def edge_summary(eb, out_key="out", truth_key="truth"):
    """sun / shd gaps (mean |out - truth| over groups x bands), per band, and sun-minus-shadow contrast (0-2 px)."""
    r = {}
    for side, tag in (("sun", "sun"), ("shadow", "shd")):
        allv, per = [], {}
        for g, sides in eb.items():
            if side not in sides:
                continue
            for b, vals in sides[side].items():
                o, t = vals.get(out_key, np.nan), vals.get(truth_key, np.nan)
                if np.isfinite(o) and np.isfinite(t):
                    allv.append(abs(o - t))
                    per.setdefault(b, []).append(abs(o - t))
        r[tag] = float(np.mean(allv)) if allv else np.nan
        for b, v in per.items():
            r[tag + b.replace("L_", "").replace("_", "")] = float(np.mean(v))
    # contrast: parts mode pairs sun/shadow within the same part; angle mode compares the two sides' means
    con_o, con_t = [], []
    sun_o, shd_o, sun_t, shd_t = [], [], [], []
    for g, sides in eb.items():
        if "sun" in sides and "shadow" in sides:
            o1, o2 = sides["sun"]["L_0_2"].get(out_key, np.nan), sides["shadow"]["L_0_2"].get(out_key, np.nan)
            t1, t2 = sides["sun"]["L_0_2"].get(truth_key, np.nan), sides["shadow"]["L_0_2"].get(truth_key, np.nan)
            if np.isfinite(o1) and np.isfinite(o2):
                con_o.append(o1 - o2)
            if np.isfinite(t1) and np.isfinite(t2):
                con_t.append(t1 - t2)
        else:
            for side, lo_, lt_ in (("sun", sun_o, sun_t), ("shadow", shd_o, shd_t)):
                if side in sides:
                    v = sides[side]["L_0_2"]
                    if np.isfinite(v.get(out_key, np.nan)):
                        lo_.append(v[out_key])
                    if np.isfinite(v.get(truth_key, np.nan)):
                        lt_.append(v[truth_key])
    if not con_o and sun_o and shd_o:
        con_o = [np.mean(sun_o) - np.mean(shd_o)]
    if not con_t and sun_t and shd_t:
        con_t = [np.mean(sun_t) - np.mean(shd_t)]
    r["con"] = float(np.mean(con_o)) if con_o else np.nan
    r["con_ref"] = float(np.mean(con_t)) if con_t else np.nan
    return r


# ------------------------------------------------------------------ canonical score
def score(out, truth, inp, light_deg=None, labels=None, Hf_ref=REF_H, sigma_frac=0.01, regions=None, keys_required=None):
    """Canonical bench numbers for one output frame (see the module docstring). truth may be None (then only the
    truth-free numbers). regions: optional precomputed colour-region label image of the input."""
    res = {}
    a = inp[..., 3]
    Hf = fig_h(a)
    res["Hf"] = Hf
    res["alpha_same"] = bool(np.array_equal(out[..., 3], inp[..., 3]))
    res["nan_out"] = bool(not np.isfinite(out).all())
    o = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0) if res["nan_out"] else out
    lo, li = to_lab(np.clip(o[..., :3], 0, 1)), to_lab(np.clip(inp[..., :3], 0, 1))
    af = a > 0.5
    lines = thin_lines(inp, Hf)
    linesd = cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # drawn detail kept (band-pass correlation with the input), line contrast kept, lightened interior lines
    er = cv2.erode(af.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    bp = lambda L: cv2.GaussianBlur(L, (0, 0), max(0.7, 0.001 * Hf)) - cv2.GaussianBlur(L, (0, 0), 0.008 * Hf)
    m = af & er
    res["det"] = _corr(bp(lo[..., 0])[m], bp(li[..., 0])[m])
    lm = lines & af
    ring = (cv2.dilate(lm.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~linesd & af
    if lm.sum() >= 20 and ring.sum() >= 20:
        ci = lowpass(li[..., 0], ring, 3.0)[lm] - li[..., 0][lm]
        co = lowpass(lo[..., 0], ring, 3.0)[lm] - lo[..., 0][lm]
        res["lnC"] = float(np.median(co) / max(np.median(ci), 1e-3))
    else:
        res["lnC"] = np.nan
    d = cv2.distanceTransform(af.astype(np.uint8), cv2.DIST_L2, 5)
    interior = lm & (d > 0.004 * Hf)
    lit = interior & (lo[..., 0] > li[..., 0] + 1.0)
    res["n_line"] = int(interior.sum())
    res["lit_lines"] = int(lit.sum())
    res["lit_share"] = float(lit.sum() / max(interior.sum(), 1))
    res["mL"] = float(lo[..., 0][af].mean())
    if light_deg is not None:
        eb, mode = edge_bands({"out": o, "inp": inp} if truth is None else {"out": o, "truth": truth}, inp, light_deg,
                              labels, Hf_ref)
        res["edge_mode"] = mode
        es = edge_summary(eb, "out", "truth" if truth is not None else "inp")
        res["con"] = es["con"]
        if truth is not None:
            res.update({k: v for k, v in es.items() if k != "con"})
    if truth is not None:
        lt = to_lab(np.clip(truth[..., :3], 0, 1))
        rel = (a > 0.9) & (truth[..., 3] > 0.9)
        rel = (cv2.erode(rel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~linesd
        mats = materials(inp)
        nd = rel & (mats["dark"] < 0.4)
        sg = max(1.0, sigma_frac * Hf)
        lpo, lpt = lowpass(lo, rel, sg), lowpass(lt, rel, sg)
        dd = lpo - lpt
        res["dEa"] = float(np.linalg.norm(dd[nd], axis=1).mean()) if nd.sum() >= 200 else np.nan
        res["mL_ref"] = float(lt[..., 0][af & (truth[..., 3] > 0.5)].mean())
        res["rG"] = _corr(lpo[..., 0][rel], lpt[..., 0][rel])
        reg = _regions(li, rel) if regions is None else regions
        dc, cs, ws = [], [], []
        for r_ in np.unique(reg[reg >= 0]):
            mm = nd & (reg == r_)
            if mm.sum() >= 200:
                x = dd[mm]
                dc.append(np.linalg.norm(x - x.mean(0), axis=1))
        res["dEst"] = float(np.concatenate(dc).mean()) if dc else np.nan
        ys, xs = np.nonzero(af)
        c = max(4, int(0.12 * Hf))
        min_px = max(30, int(400 * (Hf / REF_H) ** 2))
        mg = int(np.ceil(4 * sg)) + 2           # the masked low-pass of a cell only sees the cell: crop with a kernel margin
        H_, W_ = af.shape
        for yy in range(ys.min(), ys.max() + 1, c):
            for xx in range(xs.min(), xs.max() + 1, c):
                y0, y1_, x0, x1_ = max(yy - mg, 0), min(yy + c + mg, H_), max(xx - mg, 0), min(xx + c + mg, W_)
                cell = np.zeros((y1_ - y0, x1_ - x0), bool)
                cell[yy - y0:yy - y0 + c, xx - x0:xx - x0 + c] = True
                rc, relc = reg[y0:y1_, x0:x1_], rel[y0:y1_, x0:x1_]
                for r_ in np.unique(rc[cell & (rc >= 0)]):
                    regm = cell & (rc == r_)
                    mm = regm & relc
                    if mm.sum() < min_px:
                        continue
                    x1 = lowpass(lo[y0:y1_, x0:x1_, 0], regm, sg)[mm]
                    y1 = lowpass(lt[y0:y1_, x0:x1_, 0], regm, sg)[mm]
                    if y1.std() < 0.5 or x1.std() < 1e-3:
                        continue
                    cs.append(np.corrcoef(x1, y1)[0, 1])
                    ws.append(np.sqrt(mm.sum()))
        res["rL"] = float(np.average(cs, weights=ws)) if cs else np.nan
        hm = rel
        if hm.sum() >= 200:
            hpo = (lo[..., 0] - lowpass(lo[..., 0], hm, sg))[hm]
            hpt = (lt[..., 0] - lowpass(lt[..., 0], hm, sg))[hm]
            res["hp"] = float(hpo.std() / max(hpt.std(), 1e-3))
        else:
            res["hp"] = np.nan
    req = keys_required or (CANON_TRUTH if truth is not None else CANON_NOTRUTH)
    res["nan_keys"] = [k for k in req if k in res and not _finite(res[k])] + [k for k in req if k not in res
                                                                               and not (k in EDGE_KEYS and light_deg is None)]
    if res["nan_out"]:
        res["nan_keys"].append("nan_out")
    return res


CANON_TRUTH = ("dEa", "dEst", "rL", "sun", "shd", "con", "con_ref", "det", "lnC", "hp", "mL", "mL_ref")
CANON_NOTRUTH = ("det", "lnC", "mL", "con")
EDGE_KEYS = ("sun", "shd", "con", "con_ref")


def _finite(v):
    try:
        return bool(np.isfinite(v))
    except TypeError:
        return True


def _corr(x, y):
    if len(x) < 10 or np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def region_dE(out, truth, masks, Hf=None, sig_frac=0.01):
    """Median low-pass Lab distance per named mask (R_audit style), for recolour / material reporting."""
    return {k: v["lp_dE"] for k, v in eval_audit(out, truth, masks, Hf or fig_h(out[..., 3]), sig_frac).items()}


# ============================================================ compat replicas (T0)
def sunrim_bands(out, frame, truth, labels, Hf_neutral, sun_deg=40.0):
    """dir2 sunrim.evaluate restricted to the band means used by R_methods: per LIMB part / side, L_0_2 and L_2_4 of
    'out' and 'reference'. Same geometry (blurred-DT normals), same thresholds."""
    alpha = frame[..., 3]
    Hf = fig_h(alpha)
    scale = Hf / Hf_neutral
    a = (alpha > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(a, cv2.DIST_L2, 5)
    gy, gx = np.gradient(cv2.GaussianBlur(d, (0, 0), 2.5 * max(scale, 0.6)))
    nn = np.hypot(gx, gy) + 1e-6
    sun = np.array([np.cos(np.radians(sun_deg)), -np.sin(np.radians(sun_deg))])
    face = (-gx / nn) * sun[0] + (-gy / nn) * sun[1]
    ok = (alpha > 0.5) & (truth[..., 3] > 0.5)
    Ls = {"out": L_of(out), "reference": L_of(truth)}
    res = {}
    for pi, pn in enumerate(PARTS.NAMES):
        if pn in PARTS.NOLEAK:
            continue
        pm = (labels == pi + 1) & ok
        res[pn] = {}
        for side, sm in (("sun", face > 0.4), ("shadow", face < -0.4)):
            sel = pm & sm & (d <= 16 * scale + 1)
            r = {}
            for k, L in Ls.items():
                r[k] = {}
                for lo, hi in ((0, 2), (2, 4)):
                    mm = sel & (d > lo * scale) & (d <= hi * scale)
                    r[k]["L_%d_%d" % (lo, hi)] = float(L[mm].mean()) if mm.sum() >= 5 else np.nan
            res[pn][side] = r
    return res


def score_rmethods(out, truth, inp, kind, Hf_neutral=None):
    """R_methods common.score (dE per material at sigma 0.6 % H, dE_all non-dark, dE_struct, corrL global/local,
    detail_kept, line_contrast, rim sun/shadow gaps and contrast with the sun fixed at 40 deg)."""
    a = inp[..., 3]
    Hf = fig_h(a)
    rel = (inp[..., 3] > 0.9) & (truth[..., 3] > 0.9)
    rel = cv2.erode(rel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    mats = materials(inp)
    line = cv2.dilate(mats["line"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rel &= ~line
    lo, lt = to_lab(out[..., :3]), to_lab(truth[..., :3])
    s = 0.006 * Hf
    A = rel
    lpo = np.dstack([lowpass(lo[..., c], A, s) for c in range(3)])
    lpt = np.dstack([lowpass(lt[..., c], A, s) for c in range(3)])
    res = {}
    for k in ("skin", "light", "dark"):
        m = A & (mats[k] > 0.6)
        if m.sum() < 200:
            continue
        dd = lpo[m] - lpt[m]
        res[f"dE_{k}"] = float(np.linalg.norm(dd, axis=1).mean())
    m = A & (mats["dark"] < 0.4)
    res["dE_all"] = float(np.linalg.norm(lpo[m] - lpt[m], axis=1).mean())
    res["corrL_global"] = float(np.corrcoef(lpo[..., 0][A], lpt[..., 0][A])[0, 1])
    dc = []
    for k in ("skin", "light"):
        m = A & (mats[k] > 0.6)
        if m.sum() >= 200:
            dd = lpo[m] - lpt[m]
            dc.append(np.linalg.norm(dd - dd.mean(0), axis=1))
    res["dE_struct"] = float(np.concatenate(dc).mean())
    li = to_lab(inp[..., :3])[..., 0]
    af = a > 0.5
    er = af & (cv2.erode(af.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    bp = lambda L: cv2.GaussianBlur(L, (0, 0), max(0.7, 0.001 * Hf)) - cv2.GaussianBlur(L, (0, 0), 0.008 * Hf)
    res["detail_kept"] = float(np.corrcoef(bp(lo[..., 0])[er], bp(li)[er])[0, 1])
    lm = mats["line"] & af
    ring = (cv2.dilate(lm.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~line & af
    ci = lowpass(li, ring, 3.0)[lm] - li[lm]
    co = lowpass(lo[..., 0], ring, 3.0)[lm] - lo[..., 0][lm]
    res["line_contrast"] = float(np.median(co) / max(np.median(ci), 1e-3))
    ys, xs = np.nonzero(a > 0.5)
    c = int(0.12 * Hf)
    cs, ws = [], []
    for yy in range(ys.min(), ys.max() + 1, c):
        for xx in range(xs.min(), xs.max() + 1, c):
            for k in ("skin", "light"):
                reg = np.zeros(a.shape, bool)
                reg[yy:yy + c, xx:xx + c] = True
                reg &= (mats[k] > 0.5) & (a > 0.5)
                mm = reg & A
                if mm.sum() < 400:
                    continue
                x1 = lowpass(lo[..., 0], reg, s)[mm]
                y1 = lowpass(lt[..., 0], reg, s)[mm]
                if y1.std() < 0.5 or x1.std() < 1e-3:
                    continue
                cs.append(np.corrcoef(x1, y1)[0, 1]); ws.append(np.sqrt(mm.sum()))
    res["corrL_local"] = float(np.average(cs, weights=ws)) if cs else float("nan")
    res["mean_L"] = float(lo[..., 0][A].mean()); res["mean_L_ref"] = float(lt[..., 0][A].mean())
    if kind is None:
        return res
    labels = PARTS.labels(kind, a.shape)
    r = sunrim_bands(out, inp, truth, labels, Hf_neutral or Hf)
    sun, sh = [], []
    for pn in PARTS.LIMB:
        if pn not in r:
            continue
        for side, acc in (("sun", sun), ("shadow", sh)):
            for b in ("L_0_2", "L_2_4"):
                o, t = r[pn][side]["out"][b], r[pn][side]["reference"][b]
                if np.isfinite(o) and np.isfinite(t):
                    acc.append(abs(o - t))
    res["rim_sun_gap"] = float(np.mean(sun)); res["rim_shadow_gap"] = float(np.mean(sh))
    con_o, con_t = [], []
    for pn in PARTS.LIMB:
        if pn in r:
            o1, o2 = r[pn]["sun"]["out"]["L_0_2"], r[pn]["shadow"]["out"]["L_0_2"]
            t1, t2 = r[pn]["sun"]["reference"]["L_0_2"], r[pn]["shadow"]["reference"]["L_0_2"]
            if all(np.isfinite([o1, o2, t1, t2])):
                con_o.append(o1 - o2); con_t.append(t1 - t2)
    res["rim_contrast"] = float(np.mean(con_o)); res["rim_contrast_ref"] = float(np.mean(con_t))
    return res


def eval_audit(out, gt, masks, Hf, sig_frac=0.01):
    """R_audit common.eval_vs: per mask median low-pass Lab distance, mean dL/da/db, high-pass ratio."""
    sig = max(1.0, sig_frac * Hf)
    lo, lg = to_lab(out[..., :3]), to_lab(gt[..., :3])
    both = (out[..., 3] > 0.5) & (gt[..., 3] > 0.5)
    res = {}
    for k, m in masks.items():
        m = m & both
        if m.sum() < 50:
            continue
        lpo = np.dstack([lowpass(lo[..., c], m, sig) for c in range(3)])
        lpg = np.dstack([lowpass(lg[..., c], m, sig) for c in range(3)])
        dd = (lpo - lpg)[m]
        hpo = (lo[..., 0] - lpo[..., 0])[m]
        hpg = (lg[..., 0] - lpg[..., 0])[m]
        res[k] = {"n": int(m.sum()), "lp_dE": round(float(np.median(np.linalg.norm(dd, axis=1))), 2),
                  "dL": round(float(dd[:, 0].mean()), 2), "da": round(float(dd[:, 1].mean()), 2),
                  "db": round(float(dd[:, 2].mean()), 2),
                  "hp_ratio": round(float(hpo.std() / max(hpg.std(), 1e-3)), 3)}
    return res


def practice_valid(n, r):
    """R_practice scenemodel.Pair.valid: both alphas > 0.9 (eroded), not line art (L* < 38 on the neutral, dilated),
    the two drawings agree locally (L* window correlation > 0.3 or both flat), neutral not black.
    n, r: RGBA; r should already be defringed like Pair does."""
    labN, labR = to_lab(n[..., :3]), to_lab(r[..., :3])
    ys = np.nonzero(n[..., 3] > 0.5)[0]
    H = float(ys.max() - ys.min() + 1)
    both = (n[..., 3] > 0.9) & (r[..., 3] > 0.9)
    both = cv2.erode(both.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    line = cv2.dilate((labN[..., 0] < 38).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    w = int(0.02 * H) | 1
    Ln, Lr = labN[..., 0], labR[..., 0]
    mu = lambda x: cv2.blur(x, (w, w))
    va = np.maximum(mu(Ln * Ln) - mu(Ln) ** 2, 0); vb = np.maximum(mu(Lr * Lr) - mu(Lr) ** 2, 0)
    cc = (mu(Ln * Lr) - mu(Ln) * mu(Lr)) / np.sqrt(np.maximum(va * vb, 1e-6))
    agree = (cc > 0.3) | ((va < 4) & (vb < 4))
    Nl = srgb_to_linear(np.clip(n[..., :3], 0, 1))
    return both & ~line & agree & (Nl.min(-1) > 0.01)


def score_practice(out, r, valid, H):
    """R_practice scenemodel.score: mean / median Lab dE on the valid mask, mean dL, low-pass (1 % H) L* correlation."""
    m = valid
    lo, lr = to_lab(out[..., :3]), to_lab(r[..., :3])
    dE = np.linalg.norm(lo - lr, axis=-1)[m]
    s = 0.01 * H
    den = cv2.GaussianBlur(m.astype(np.float32), (0, 0), s)
    lp = lambda L: cv2.GaussianBlur(np.where(m, L, 0).astype(np.float32), (0, 0), s) / np.maximum(den, 1e-4)
    return dict(dE_mean=float(dE.mean()), dE_med=float(np.median(dE)), dL_mean=float((lo[..., 0] - lr[..., 0])[m].mean()),
                L_corr=float(np.corrcoef(lp(lo[..., 0])[m], lp(lr[..., 0])[m])[0, 1]))
