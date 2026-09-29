"""Label-free quality report for a fitted scene: neutral | relit | reference, chest-level crops and
in-region shading maps (red = brighter / blue = darker than the region median), plus numbers."""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .color import lowpass, to_lab


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
        "align_iou": pairs.iou,
        "regions": len(pairs.regions),
    }


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


def shade_map(rgba, pairs):
    lab = to_lab(rgba[..., :3])
    out = np.full(rgba.shape[:2] + (3,), 0.14, np.float32)
    sig = max(2.0, 0.006 * pairs.Hf)
    for R in pairs.regions:
        reg = R["region"] & (rgba[..., 3] > 0.5)
        if reg.sum() < 50:
            continue
        lp = lowpass(lab[..., 0], reg, sig)
        d = np.clip((lp - np.median(lp[reg])) / 8.0, -1, 1)[..., None]
        col = np.where(d > 0, (1 - d) * 0.55 + d * np.array([1, 0.15, 0.05]), (1 + d) * 0.55 - d * np.array([0.05, 0.1, 1]))
        out[reg] = col[reg]
    return out


def sheet(relit, pairs, path):
    ow = pairs.warp(relit)
    ys, xs = np.nonzero(pairs.ref[..., 3] > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = 10
    y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0)
    y1, x1 = min(y1 + pad, pairs.shape[0]), min(x1 + pad, pairs.shape[1])
    Hf = y1 - y0
    # upper-body crop: 15%..50% of figure height, centred on the torso, zoomed to the full-figure panel width
    cy0, cy1 = y0 + int(0.15 * Hf), y0 + int(0.5 * Hf)
    tx = int(np.median(np.nonzero(pairs.ref[cy0:cy1, :, 3] > 0.5)[1]))
    half = int(0.25 * Hf)
    cx0, cx1 = max(tx - half, 0), min(tx + half, pairs.shape[1])
    cols = [("neutral", pairs.neutral_w), ("relit", ow), ("reference", pairs.ref)]
    rows = [np.hstack([_label(on_grey(x)[y0:y1, x0:x1], t) for t, x in cols])]
    z = (x1 - x0) / (cx1 - cx0)
    up = lambda img, interp=cv2.INTER_CUBIC: cv2.resize(img[cy0:cy1, cx0:cx1], (x1 - x0, int((cy1 - cy0) * z)), interpolation=interp)
    rows.append(np.hstack([_label(up(on_grey(x)), t + " upper body") for t, x in cols]))
    rows.append(np.hstack([_label(up(shade_map(x, pairs), cv2.INTER_NEAREST), t + " shading") for t, x in cols]))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0)), constant_values=0.1) for r in rows]
    Image.fromarray((np.clip(np.vstack(rows), 0, 1) * 255).astype(np.uint8)).save(path)


def before_after(frames, outs, path, height=480):
    tiles = []
    for f, o in zip(frames, outs):
        s = height / f.shape[0]
        pair = np.hstack([on_grey(f), on_grey(o)])
        tiles.append(cv2.resize(pair, (int(pair.shape[1] * s), height), interpolation=cv2.INTER_AREA))
    tiles = [np.pad(t, ((0, 0), (0, 8), (0, 0)), constant_values=0.1) for t in tiles]
    Image.fromarray((np.clip(np.hstack(tiles), 0, 1) * 255).astype(np.uint8)).save(path)
