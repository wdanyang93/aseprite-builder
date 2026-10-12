"""Reference look: learn a light look from ONE original + lit reference pair (same pose) and put it on other frames and
motions by each frame's own geometry (the lamp-side leg gets the light after the feet swap).

    from scenelight.parts import load_parts, load_lines
    look = fit_look(orig, ref, load_parts("masks/orig"), load_lines("masks/orig/vector.svg", orig.shape))
    out = apply_look(frame, load_parts("masks/frame"), look, load_lines("masks/frame/vector.svg", frame.shape))
    json.dump(look, f)                          # one look per light (lamp only / moon only / moon + lantern)

parts : {name: HxW mask} Adobe part masks (image_select_subject bodyParts / image_select_by_prompt), >= 0.5 = in.
        First claim wins where masks overlap. Same names on every frame (hair, skin, upper, lower, shoes, tail ...);
        left/right pairs merge (leftleg + rightleg -> legs: light follows where a leg stands, never its label);
        'subject', names the look does not know and unclaimed figure pixels use the whole-figure table '*'.
lines : HxW Adobe line art (thin dark paths of the image_vectorize SVG), optional. Held: never lighter.
orig/ref/frame : HxWx4 uint8 or float 0..1, straight alpha; the reference is aligned onto the original.

What a look stores, per part (+ '*'): the change as a per-channel LINEAR-RGB log ratio for every
(depth band from the silhouette) x (facing bin = silhouette outward normal relative to the light azimuth), plus one
tone slope per depth band (darker bases gain relatively more: a lamp adds light to dark leather but turns light
cloth orange by lowering blue, as the painted references do). Fitted by ridge least squares on the pair.
    out = (orig + EPS) * ratio - EPS      coloured multiplicative light; on near-black bases the EPS term is additive
The azimuth is estimated from the pair (direction of strongest edge brightening); bins are relative to it, so a
look can be re-aimed by editing look["az"]. Apply interpolates smoothly between bins, feathers part borders, and
compresses near white with a hue-preserving soft knee (light materials take the light's colour, they never blow out
to white). Alpha is copied bit-identical; nothing depends on screen position.
"""
import cv2
import numpy as np

from .color import figure_height, linear_to_srgb, lowpass, srgb_to_linear

EPS = 0.04            # linear-RGB offset of the ratio: multiplicative on light bases, additive on near-black ones
NF = 8                # facing bins (bin 0 faces the light)
D_CENTERS = np.array([1.0, 3.0, 7.0, 18.0, 50.0]) / 1405.0   # depth band centres (fraction of figure height):
                      # bands 0-2, 2-4, 4-10, 10-30, 30+ px at a 1405 px figure
NSCALES = (0.0025, 0.005, 0.01, 0.02, 0.04)  # normal field scales (fraction of H): fine at the rim, coarse inside
EDGE_MAG = 8.0 / np.sqrt(2 * np.pi)          # sigma * Sobel response of a blurred unit step, at the step
N0 = 1e-6             # ridge prior weight (pixels, fraction of H^2): part -> whole figure, cell -> band -> mean
FEATHER = 0.003       # part border feather (fraction of H)
KNEE = 0.8            # linear max-channel level where the soft knee starts
MIN_CONF = 0.2        # alignment confidence needed to fit a pixel
TONE_RANGE = (10, 99) # base-tone percentiles the tone slope is fitted / applied within (no extrapolation into the
                      # darkest creases: at the 1st percentile near-black browns turned blue on a transfer)
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ------------------------------------------------------------------ geometry (alpha only)
def geometry(alpha):
    """-> depth from the silhouette (fraction of H), outward normal angle (rad, CCW from screen-right, y up),
    normal confidence (1 near an edge, -> 0 at medial points where the normal is undefined)."""
    alpha = alpha.astype(np.float32)
    H = figure_height(alpha)
    body = alpha > 0.5
    d = cv2.distanceTransform(body.astype(np.uint8), cv2.DIST_L2, 5)
    ds = lowpass(d, body, max(1.0, 0.003 * H))               # no creases along the medial axis deeper inside
    w = np.clip((d - 0.003 * H) / (0.006 * H), 0, 1)
    d = (1 - w) * d + w * ds
    gx = np.zeros_like(alpha); gy = np.zeros_like(alpha); ref = np.zeros_like(alpha)
    for s in NSCALES:
        sig = max(1.0, s * H)
        g = cv2.GaussianBlur(alpha, (0, 0), sig)
        gx += sig * cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy += sig * cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        ref += EDGE_MAG * np.exp(-d ** 2 / (2 * sig ** 2))
    ang = np.arctan2(gy, -gx)                                # outward = -grad(alpha); y up
    conf = np.clip(np.hypot(gx, gy) / np.maximum(ref, EDGE_MAG), 0, 1)   # |g| -> 0 deep inside: no cut-off seam
    return (d / H).astype(np.float32), ang.astype(np.float32), conf.astype(np.float32)


def _coords(d, ang, az):
    """bilinear cell coordinates: depth (j0, j1, s) and facing (i0, i1, t)."""
    v = np.interp(d, D_CENTERS, np.arange(len(D_CENTERS)))
    j0 = np.floor(v).astype(np.int32)
    j1 = np.minimum(j0 + 1, len(D_CENTERS) - 1)
    s = (v - j0).astype(np.float32)
    u = np.mod(ang - np.radians(az), 2 * np.pi) / (2 * np.pi / NF)
    i0 = np.floor(u).astype(np.int32) % NF
    i1 = (i0 + 1) % NF
    t = (u - np.floor(u)).astype(np.float32)
    return j0, j1, s, i0, i1, t


SIDES = {"leftleg": "legs", "rightleg": "legs", "lefthand": "hands", "righthand": "hands",
         "leftarm": "arms", "rightarm": "arms"}


def merge_parts(parts):
    """bool masks; anatomical left/right pairs merged, 'subject' (the whole figure) dropped."""
    out = {}
    for name, m in parts.items():
        name = SIDES.get(name.lower(), name.lower())
        if name != "subject":
            m = np.asarray(m, np.float32) >= 0.5
            out[name] = out[name] | m if name in out else m
    return out


def labels(parts, fig):
    """first-claim label map: index into list(parts), -1 = unclaimed figure / outside."""
    lab = np.full(fig.shape, -1, np.int32)
    for k, m in enumerate(parts.values()):
        lab[(lab < 0) & fig & (np.asarray(m, np.float32) >= 0.5)] = k
    return lab


# ------------------------------------------------------------------ fit
def light_azimuth(lin_o, lin_r, d, ang, w):
    """Azimuth (deg) of the strongest edge brightening, and the 36-bin brightening profile (log luminance change of
    the outer band relative to the whole figure's median change)."""
    b = np.log((lin_r @ LUM + EPS) / (lin_o @ LUM + EPS))
    edge = (w > 0) & (d < D_CENTERS[2])
    b = b - np.median(b[w > 0])
    k = (np.mod(np.degrees(ang[edge]), 360) // 10).astype(int) % 36
    num = np.bincount(k, b[edge] * w[edge], 36)
    den = np.bincount(k, w[edge], 36)
    prof = num / np.maximum(den, 1e-6)
    for _ in range(2):                                       # circular [1 2 1] smoothing
        prof = 0.25 * np.roll(prof, 1) + 0.5 * prof + 0.25 * np.roll(prof, -1)
    i = int(np.argmax(prof))
    a, c, e = prof[i - 1], prof[i], prof[(i + 1) % 36]
    off = 0.5 * (a - e) / (a - 2 * c + e) if (a - 2 * c + e) < 0 else 0.0
    return float(np.mod((i + 0.5 + off) * 10.0, 360)), prof


def _corners(j0, j1, s, i0, i1, t, c):
    """the 6 table entries a pixel reads (4 facing x depth cells, 2 all-facing band cells) and their weights."""
    nd = len(D_CENTERS) * NF
    idx = [j0 * NF + i0, j0 * NF + i1, j1 * NF + i0, j1 * NF + i1, nd + j0, nd + j1]
    wts = [c * (1 - s) * (1 - t), c * (1 - s) * t, c * s * (1 - t), c * s * t, (1 - c) * (1 - s), (1 - c) * s]
    return idx, wts


def _solve(idx, wts, y, w, reg, rhs):
    """ridge least squares for the table: the entries whose interpolation (exactly as apply reads them) best fits y."""
    n = reg.shape[0]
    AtA, Atb = reg.copy(), rhs.copy()
    for ia, wa in zip(idx, wts):
        for c in range(3):
            Atb[:, c] += np.bincount(ia, w * wa * y[:, c], n)
        for ib, wb in zip(idx, wts):
            AtA += np.bincount(ia * n + ib, w * wa * wb, n * n).reshape(n, n)
    return np.linalg.solve(AtA, Atb)


def fit_look(orig_rgba, ref_rgba, parts, lines=None, align=True):
    """Learn the look of one lit reference of the same pose. orig/ref: HxWx4 (uint8 or float 0..1, straight alpha);
    the reference is aligned onto the original (scenelight.align2) unless align=False. -> look dict (JSON-able)."""
    o, _ = _as_float(orig_rgba)
    r, _ = _as_float(ref_rgba)
    conf_a = np.ones(o.shape[:2], np.float32)
    if align:
        from .align2 import Pairs
        P = Pairs(r, o, regions=False)                       # reference -> original grid
        pm = np.dstack([r[..., :3] * r[..., 3:], r[..., 3]])  # warp premultiplied: no background in the rim
        rw = cv2.remap(pm, P.mapx, P.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        r = np.dstack([rw[..., :3] / np.maximum(rw[..., 3:], 1e-4), rw[..., 3]])
        conf_a = P.conf
    alpha = o[..., 3]
    H = figure_height(alpha)
    d, ang, conf = geometry(alpha)
    w = ((alpha > 0.5) & (r[..., 3] > 0.5) & (conf_a >= MIN_CONF)).astype(np.float32) * np.clip(conf_a, 0, 1)
    if lines is not None:
        w *= np.asarray(lines, np.float32) < 0.5
    lin_o = srgb_to_linear(np.clip(o[..., :3], 0, 1))
    lin_r = srgb_to_linear(np.clip(r[..., :3], 0, 1))
    az, prof = light_azimuth(lin_o, lin_r, d, ang, w)
    parts = merge_parts(parts)
    lab = labels(parts, alpha > 0)
    m = w > 0
    tone = np.log(lin_o[m] @ LUM + EPS)                      # base tone: darker bases gain relatively more
    x0, lo, hi = float(np.average(tone, weights=w[m])), *np.percentile(tone, TONE_RANGE).tolist()
    xt = np.clip(tone, lo, hi) - x0
    nb, nd, n0 = len(D_CENTERS), len(D_CENTERS) * NF, N0 * H * H
    n = nd + 2 * nb                                          # entries: facing x depth, depth, tone slope per depth
    j0, j1, s, i0, i1, t = _coords(d[m], ang[m], az)
    idx, wts = _corners(j0, j1, s, i0, i1, t, conf[m])
    idx, wts = idx + [nd + nb + j0, nd + nb + j1], wts + [(1 - s) * xt, s * xt]
    wm, lm = w[m], lab[m]
    # per-pixel log ratio: dark texture and strand gaps count as much as light pixels
    y = np.clip(np.log((lin_r[m] + EPS) / (lin_o[m] + EPS)), -3, 3)
    # whole figure: each facing cell is pulled toward its band's all-facing value, each band toward the figure mean,
    # slopes toward 0
    reg = n0 * np.eye(n)
    for c in range(nd):
        b = nd + c // NF
        reg[b, b] += n0; reg[c, b] -= n0; reg[b, c] -= n0
    rhs = np.zeros((n, 3))
    rhs[nd:nd + nb] = n0 * (wm[:, None] * y).sum(0) / wm.sum()
    x = _solve(idx, wts, y, wm, reg, rhs)
    tabs = {"*": x}
    for k, name in enumerate(parts):                         # parts: pulled toward the whole-figure table
        sel = lm == k
        if sel.sum() >= 20:
            tabs[name] = _solve([i[sel] for i in idx], [v[sel] for v in wts], y[sel], wm[sel], n0 * np.eye(n), n0 * x)
    r4 = lambda v: np.round(v, 4).tolist()
    tabs = {k: {"dir": r4(v[:nd].reshape(nb, NF, 3)), "iso": r4(v[nd:nd + nb]), "slope": r4(v[nd + nb:])}
            for k, v in tabs.items()}
    return {"version": 1, "az": round(az, 1), "profile": np.round(prof, 3).tolist(), "eps": EPS,
            "tone": [round(x0, 4), round(lo, 4), round(hi, 4)], "d_centers": D_CENTERS.tolist(), "nf": NF,
            "tables": tabs}


# ------------------------------------------------------------------ apply
def soft_knee(lin, base=None, t=KNEE):
    """hue-preserving soft knee on the max channel: all channels scaled by one factor, nothing clips to white.
    base: the original (linear); the knee starts at its own max channel when that is above t, so an unchanged pixel
    stays exactly unchanged."""
    lin = np.maximum(lin, 0)
    mx = lin.max(-1)
    t = np.full(mx.shape, t, np.float32) if base is None else np.clip(base.max(-1), t, 0.999)
    k = np.where(mx > t, t + (1 - t) * np.tanh((mx - t) / (1 - t)), mx)
    return lin * (k / np.maximum(mx, 1e-6))[..., None]


def log_ratio(alpha, parts, look, lin=None):
    """per-pixel linear-RGB log ratio (HxWx3) of the look on a frame, from its alpha, part masks and (linear) colours."""
    fig = alpha > 0
    H = figure_height(alpha)
    d, ang, conf = geometry(alpha)
    tabs = look["tables"]
    x0, lo, hi = look["tone"]
    xt = np.zeros(alpha.shape, np.float32) if lin is None else np.clip(np.log(lin @ LUM + look["eps"]), lo, hi) - x0
    parts = merge_parts(parts)
    names = list(parts)
    lab = labels(parts, fig)
    # part weights: one-hot per part (unknown names and unclaimed pixels -> '*'), feathered across part borders
    keys = sorted({nm if nm in tabs else "*" for nm in names} | {"*"})
    key_of = np.array([keys.index(nm if nm in tabs else "*") for nm in names] + [keys.index("*")])
    kl = key_of[lab]                                          # lab -1 -> last entry ('*')
    q = np.zeros(alpha.shape + (3,), np.float32)
    wsum = np.zeros(alpha.shape, np.float32)
    coords = _coords(d, ang, look["az"])
    for ki, key in enumerate(keys):
        wk = lowpass((kl == ki).astype(np.float32), fig, max(1.0, FEATHER * H)) * fig
        m = wk > 1e-3
        if not m.any():
            continue
        tb = tabs[key]
        A = np.concatenate([np.reshape(tb["dir"], (-1, 3)), tb["iso"]]).astype(np.float32)
        Sl = np.asarray(tb["slope"], np.float32)
        j0, j1, s, i0, i1, t = (c[m] for c in coords)
        idx, wts = _corners(j0, j1, s, i0, i1, t, conf[m])
        qk = ((1 - s)[:, None] * Sl[j0] + s[:, None] * Sl[j1]) * xt[m, None]
        for i, wt in zip(idx, wts):
            qk += wt[:, None] * A[i]
        q[m] += wk[m, None] * qk
        wsum[m] += wk[m]
    return q / np.maximum(wsum, 1e-6)[..., None]


def apply_look(frame_rgba, parts, look, lines=None):
    """Put the look on a frame (any pose / motion of a character with the same lights). Returns the same dtype as the
    input; alpha bit-identical, transparent pixels unchanged, Adobe line pixels never lighter in any channel."""
    x, is_u8 = _as_float(frame_rgba)
    alpha = x[..., 3]
    fig = alpha > 0
    eps = float(look.get("eps", EPS))
    lin_o = srgb_to_linear(np.clip(x[..., :3], 0, 1))
    q = log_ratio(alpha, parts, look, lin_o)
    lin = soft_knee((lin_o + eps) * np.exp(q) - eps, lin_o)
    held = fig & (np.asarray(lines, np.float32) >= 0.5) if lines is not None else np.zeros_like(fig)
    if held.any():                                           # lit hue, scaled down until no channel is lighter
        k = np.min(np.where(lin > 1e-6, lin_o / np.maximum(lin, 1e-6), 1.0), -1)
        lin = np.where(held[..., None], lin * np.minimum(k, 1.0)[..., None], lin)
    rgb = np.where(fig[..., None], linear_to_srgb(lin), x[..., :3]).astype(np.float32)
    if is_u8:
        src = np.asarray(frame_rgba)
        out = src.copy()
        u8 = np.floor(np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        u8 = np.where(held[..., None], np.minimum(u8, src[..., :3]), u8)
        out[..., :3] = np.where(fig[..., None], u8, src[..., :3])
        return out
    rgb = np.where(held[..., None], np.minimum(rgb, x[..., :3]), rgb)
    out = np.array(frame_rgba, np.float32, copy=True)
    out[..., :3] = rgb
    return out


def _as_float(img):
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0, True
    return a.astype(np.float32), False
