"""Label-free quality report for a fitted scene: neutral | relit | reference, chest-level crops and
in-region shading maps (red = brighter / blue = darker than the region median), plus numbers."""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .color import lowpass, materials, to_lab


def metrics(relit, pairs):
    """Shading-shape agreement (low-pass L* correlation inside every fitting region, weighted),
    low-pass colour error and mean L*, all measured in the reference frame."""
    ow = pairs.warp(relit)
    lo, lr, ln = to_lab(ow[..., :3]), to_lab(pairs.ref[..., :3]), to_lab(pairs.neutral_w[..., :3])
    sig = max(2.0, 0.006 * pairs.Hf)
    c_out, c_neu, w_all, de = [], [], [], []
    for R in pairs.regions:
        reg, m = R["region"], R["mask"]
        a = lowpass(lo[..., 0], reg, sig)[m]
        b = lowpass(lr[..., 0], reg, sig)[m]
        n = lowpass(ln[..., 0], reg, sig)[m]
        if b.std() < 0.5:
            continue
        c_out.append(np.corrcoef(a, b)[0, 1] if a.std() > 1e-3 else 0.0)
        c_neu.append(np.corrcoef(n, b)[0, 1] if n.std() > 1e-3 else 0.0)
        w_all.append(R["w"] * np.sqrt(m.sum()))
        lpo = np.dstack([lowpass(lo[..., c], reg, sig) for c in range(3)])[m]
        lpr = np.dstack([lowpass(lr[..., c], reg, sig) for c in range(3)])[m]
        de.append(np.median(np.linalg.norm(lpo - lpr, axis=1)))
    w = np.array(w_all)
    A = (ow[..., 3] > 0.9) & pairs.reliable
    return {
        "shape_agreement": float(np.average(c_out, weights=w)),
        "shape_agreement_neutral": float(np.average(c_neu, weights=w)),
        "color_error_lowfreq": float(np.average(de, weights=w)),
        "mean_L": float(lo[..., 0][A].mean()),
        "mean_L_reference": float(lr[..., 0][A].mean()),
        "drawn_form_kept": drawn_form_kept(pairs.neutral, relit),
        "align_iou": pairs.iou,
        "regions": len(pairs.regions),
    }


def drawn_form_kept(neutral, relit):
    """How much of the drawing's own chest form survives, measured in the NEUTRAL frame (nothing warped):
    correlation of the band-passed L* (scales below ~6% of figure height: cups, cleavage, painted shading)
    between relit and neutral in the chest centre (upper-torso band, away from the silhouette edge).
    ~1.0 = the drawn shape is untouched; the added scene light lives at larger scales."""
    a = neutral[..., 3] > 0.5
    ys = np.nonzero(a.any(1))[0]
    y0, Hf = ys.min(), ys.max() - ys.min()
    mats = materials(neutral)
    dist = cv2.distanceTransform(a.astype(np.uint8), cv2.DIST_L2, 5)
    band = np.zeros_like(a)
    band[y0 + int(0.17 * Hf): y0 + int(0.36 * Hf)] = True
    line = cv2.dilate(mats["line"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    core = a & band & (dist > 0.04 * Hf) & ~line & (mats["dark"] < 0.3)
    Ln, Lo = to_lab(neutral[..., :3])[..., 0], to_lab(relit[..., :3])[..., 0]
    cs, ws = [], []
    for sel in (mats["skin"] > 0.5, mats["light"] > 0.5):
        m = core & sel
        if m.sum() < 100:
            continue
        bp = lambda L: lowpass(L, m, 0.004 * Hf) - lowpass(L, m, 0.06 * Hf)
        bn, bo = bp(Ln)[m], bp(Lo)[m]
        if bn.std() < 1e-3:
            continue
        cs.append(np.corrcoef(bn, bo)[0, 1]); ws.append(m.sum())
    return float(np.average(cs, weights=ws)) if cs else float("nan")


def added_light_edges(neutral, relit, base):
    """Sharpest bend of the ADDED light inside the body (L* per (1% of figure height)^2, 99th percentile).
    A hard terminator or crease shows up as a large value; a smooth scene light stays small."""
    a = neutral[..., 3] > 0.5
    ys = np.nonzero(a.any(1))[0]
    Hf = ys.max() - ys.min()
    dist = cv2.distanceTransform(a.astype(np.uint8), cv2.DIST_L2, 5)
    core = a & (dist > 0.03 * Hf)
    S = to_lab(relit[..., :3])[..., 0] - to_lab(base[..., :3])[..., 0]
    S = lowpass(S, a, 0.004 * Hf)
    lap = cv2.Laplacian(S.astype(np.float32), cv2.CV_32F, ksize=5) / 8.0  # ~ second derivative per px^2
    u = (0.01 * Hf) ** 2
    return float(np.percentile(np.abs(lap[core]) * u, 99))


def _font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(img, text):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    d.text((6, 3), text, font=_font(18), fill=(255, 255, 255))
    return np.asarray(im).astype(np.float32) / 255


def on_grey(rgba, g=0.38):
    return rgba[..., :3] * rgba[..., 3:4] + g * (1 - rgba[..., 3:4])


def _grid_regions(alpha, mats, cell=0.12):
    ys, xs = np.nonzero(alpha > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    c = max(8, int(round(cell * (y1 - y0 + 1))))
    out = []
    for yy in range(y0, y1 + 1, c):
        for xx in range(x0, x1 + 1, c):
            for mat in ("skin", "light"):
                reg = np.zeros(alpha.shape, bool)
                reg[yy:yy + c, xx:xx + c] = True
                reg &= (mats[mat] > 0.5) & (alpha > 0.5)
                if reg.sum() >= 50:
                    out.append(reg)
    return out, float(y1 - y0 + 1)


def shade_map(rgba, mats):
    """Low-pass L* relative to the median of each grid cell x material (red brighter, blue darker)."""
    lab = to_lab(rgba[..., :3])
    out = np.full(rgba.shape[:2] + (3,), 0.14, np.float32)
    regions, Hf = _grid_regions(rgba[..., 3], mats)
    sig = max(2.0, 0.006 * Hf)
    for reg in regions:
        lp = lowpass(lab[..., 0], reg, sig)
        d = np.clip((lp - np.median(lp[reg])) / 8.0, -1, 1)[..., None]
        col = np.where(d > 0, (1 - d) * 0.55 + d * np.array([1, 0.15, 0.05]), (1 + d) * 0.55 - d * np.array([0.05, 0.1, 1]))
        out[reg] = col[reg]
    return out


def _crop_boxes(alpha):
    """Full-figure box and upper-body box (15%..50% of figure height, centred on the torso) in own geometry."""
    ys, xs = np.nonzero(alpha > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    Hf = y1 - y0 + 1
    pad = int(0.02 * Hf)
    full = (max(y0 - pad, 0), min(y1 + pad, alpha.shape[0]), max(x0 - pad, 0), min(x1 + pad, alpha.shape[1]))
    cy0, cy1 = y0 + int(0.15 * Hf), y0 + int(0.5 * Hf)
    tx = int(np.median(np.nonzero(alpha[cy0:cy1] > 0.5)[1]))
    half = int(0.25 * Hf)
    return full, (cy0, cy1, max(tx - half, 0), min(tx + half, alpha.shape[1]))


def _fit(img, box, H, interp=cv2.INTER_AREA):
    y0, y1, x0, x1 = box
    s = H / (y1 - y0)
    return cv2.resize(img[y0:y1, x0:x1], (max(1, int((x1 - x0) * s)), H), interpolation=interp)


def sheet(relit, pairs, path, H=800):
    """neutral | relit | reference, each in its OWN geometry (nothing warped, so every shape is the real one)."""
    n, r = pairs.neutral, pairs.ref
    n_mats = materials(n)
    rows = [[], [], []]
    for t, img, mats in (("neutral", n, n_mats), ("relit", relit, n_mats), ("reference", r, pairs.mat)):
        full, up = _crop_boxes(img[..., 3])
        rows[0].append(_label(_fit(on_grey(img), full, H), t))
        rows[1].append(_label(_fit(on_grey(img), up, int(H * 0.7)), t + " upper body"))
        rows[2].append(_label(_fit(shade_map(img, mats), up, int(H * 0.7), cv2.INTER_NEAREST), t + " shading"))
    rows = [np.hstack([np.pad(x, ((0, 0), (0, 6), (0, 0)), constant_values=0.1) for x in row]) for row in rows]
    w = max(r_.shape[1] for r_ in rows)
    rows = [np.pad(r_, ((0, 6), (0, w - r_.shape[1]), (0, 0)), constant_values=0.1) for r_ in rows]
    Image.fromarray((np.clip(np.vstack(rows), 0, 1) * 255).astype(np.uint8)).save(path)


def before_after(frames, outs, path, height=480):
    tiles = []
    for f, o in zip(frames, outs):
        s = height / f.shape[0]
        pair = np.hstack([on_grey(f), on_grey(o)])
        tiles.append(cv2.resize(pair, (int(pair.shape[1] * s), height), interpolation=cv2.INTER_AREA))
    tiles = [np.pad(t, ((0, 0), (0, 8), (0, 0)), constant_values=0.1) for t in tiles]
    Image.fromarray((np.clip(np.hstack(tiles), 0, 1) * 255).astype(np.uint8)).save(path)
