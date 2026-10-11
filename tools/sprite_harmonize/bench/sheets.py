"""Contact sheets: per case, rows = frames, columns = input | truth | every method (colour on grey, optional L* map).
Written under <out>/sheets/ (private: they show the user's art)."""
import os

import numpy as np
import cv2
from PIL import Image, ImageDraw

from . import config, metrics as MX
from . import cases as CS
from . import methods as MT
from . import run as RUN


def _on_grey(x, g=0.38):
    return np.clip(x[..., :3], 0, 1) * x[..., 3:4] + g * (1 - x[..., 3:4])


def _lmap(x):
    L = MX.L_of(x)
    im = cv2.applyColorMap((np.clip(L / 100.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1] / 255.0
    a = x[..., 3:4] > 0.5
    return im * a + 0.12 * (~a)


def _tile(img, box, h, label, lstar=False):
    if img is None:
        t = np.full((h, max(8, int(h * 0.5)), 3), 0.2, np.float32)
    else:
        y0, y1, x0, x1 = box
        x = (_lmap if lstar else _on_grey)(img)[y0:y1, x0:x1]
        s = h / x.shape[0]
        t = cv2.resize(x.astype(np.float32), (max(1, int(round(x.shape[1] * s))), h),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    pi = Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pi)
    d.rectangle([0, 0, pi.width, 13], fill=(0, 0, 0))
    d.text((2, 1), label[:max(4, pi.width // 6)], fill=(255, 255, 255))
    return np.asarray(pi)


def _box(a, margin=0.03):
    ys, xs = np.nonzero(a > 0.5)
    m = int(margin * (ys.max() - ys.min() + 1))
    return max(0, ys.min() - m), min(a.shape[0], ys.max() + m + 1), max(0, xs.min() - m), min(a.shape[1], xs.max() + m + 1)


def case_sheet(cn, method_names, lstar=False, max_frames=13):
    c = CS.get(cn)
    nf = min(len(c.frames), max_frames)
    h = 300 if nf <= 2 else (220 if nf <= 5 else 150)
    rows = []
    for fr in c.frames[:nf]:
        F = fr.img()
        box = _box(F[..., 3])
        cells = [_tile(F, box, h, f"{fr.name} input", lstar)]
        cells.append(_tile(fr.truth() if fr.has_truth else None, box, h, "truth" if fr.has_truth else "no truth", lstar))
        for m in method_names:
            p = RUN.out_path(c.fit_id, fr.name, m)
            img = None
            if os.path.exists(p):
                with np.load(p) as z:
                    img = z["x"].astype(np.float32)
            cells.append(_tile(img, box, h, m if img is not None else f"{m} n/a", lstar))
        rows.append(np.concatenate([np.pad(x, ((0, 0), (0, 4), (0, 0)), constant_values=30) for x in cells], 1))
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=30) for r in rows]
    sheet = np.concatenate(rows, 0)
    p = os.path.join(config.out_dir("sheets"), f"{cn}{'_L' if lstar else ''}.png")
    Image.fromarray(sheet).save(p)
    return p


def _edge_boxes(F, labels, light_deg, half=110):
    """Crop boxes around the sun-facing edge: per limb part (arm_front, tail, legs) when labelled, else the silhouette
    point furthest toward the light."""
    from . import parts as PARTS
    d, nx, ny = MX.edge_geometry(F[..., 3])
    u = MX.unit(light_deg)
    sun = (F[..., 3] > 0.5) & (d <= 4) & (nx * u[0] + ny * u[1] > 0.4)
    H, W = F.shape[:2]
    boxes = []
    def box(cy, cx, name):
        return (name, (max(0, cy - half), min(H, cy + half), max(0, cx - half), min(W, cx + half)))
    if labels is not None:
        for pn in ("arm_front", "tail", "legs"):
            m = sun & (labels == PARTS.NAMES.index(pn) + 1)
            if m.sum() >= 30:
                ys, xs = np.nonzero(m)
                boxes.append(box(int(np.median(ys)), int(np.median(xs)), pn))
    if not boxes and sun.any():
        ys, xs = np.nonzero(sun)
        k = np.argmax(xs * u[0] + ys * u[1])
        boxes.append(box(int(ys[k]), int(xs[k]), "toward light"))
    return boxes


def crop_sheet(cn, method_names, frame_index=0, S=2):
    """2x crops of the sun-side edge (colour row + L* row per box) for one frame of a case."""
    c = CS.get(cn)
    fr = c.frames[frame_index]
    F = fr.img()
    imgs = [("input", F), ("truth", fr.truth() if fr.has_truth else None)]
    for m in method_names:
        p = RUN.out_path(c.fit_id, fr.name, m)
        if os.path.exists(p):
            with np.load(p) as z:
                imgs.append((m, z["x"].astype(np.float32)))
    rows = []
    for name, (y0, y1, x0, x1) in _edge_boxes(F, fr.labels(), fr.light_deg or 40.0):
        for lstar in (False, True):
            cells = [_tile(x, (y0, y1, x0, x1), (y1 - y0) * S, f"{name}: {t}{' L*' if lstar else ''}", lstar)
                     for t, x in imgs]
            rows.append(np.concatenate([np.pad(x, ((0, 0), (0, 4), (0, 0)), constant_values=30) for x in cells], 1))
    if not rows:
        return None
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=30) for r in rows]
    p = os.path.join(config.out_dir("sheets"), f"{cn}_crops.png")
    Image.fromarray(np.concatenate(rows, 0)).save(p)
    return p


def write_all(case_names, method_names, log=print):
    out = []
    for cn in case_names:
        try:
            out.append(case_sheet(cn, method_names))
            if CS.get(cn).group in ("real", "frames"):
                out.append(case_sheet(cn, method_names, lstar=True))
            if CS.get(cn).group in ("real", "frames", "recolour") and CS.get(cn).frames[0].has_truth:
                out.append(crop_sheet(cn, method_names))
        except Exception as e:
            log(f"  sheet {cn}: ERROR {e.__class__.__name__}: {e}")
    return out
