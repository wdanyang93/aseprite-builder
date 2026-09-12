"""bank — 지금까지 받은 모든 걷기 시트를 한 프레임씩 잘라 라벨링하고, 하나의 기준으로 정규화한다.

1. 슬라이스  알파 투영의 빈 간격으로 격자를 찾는다 (붙은 행/열은 기대 격자로 등분).
2. 라벨      {set}_{index:02d}.png + labels.json (출처 시트, 뷰, 착의/나체, 원래 방향, 반전 여부, 크기)
3. 방향      측면: 꼬리(밝고 채도 낮은 털, 허리 아래)가 몸 중심의 어느 쪽인가 → 꼬리 반대가 진행 방향.
             서쪽(왼쪽 보기)은 좌우반전해 전부 동쪽으로.
4. 정규화    꼬리 제외 몸 높이 H → H_REF 로 균일 스케일(키·너비 같은 비율), 머리 중심 x 와 발바닥 y 를
             고정 위치에 둔 캔버스에 저장 (norm/{id}.png). 배경 잔여(순흑) 제거 포함.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label, binary_dilation

sys.path.insert(0, "/home/claude/handoff/work")
import debg

UP = Path("/root/.claude/uploads/60a1100f-4d22-5314-b5c3-abe3f28692c8")
W = Path("/home/claude/handoff/work/bank")
H_REF = 600
CANVAS = (640, 720)
HEAD_X = 330
SOLE_Y = 680

# id, file prefix, view, clothed, expected (cols, rows), note
SHEETS = [
    ("cE1", "e1dcfb21", "side", True,  (6, 6), "clothed east 35 (=gt)"),
    ("cE2", "7cd45a07", "side", True,  (6, 6), "clothed east 36 (new)"),
    ("cW1", "c6f74a94", "side", True,  (6, 6), "clothed west 36"),
    ("cW2", "4d896041", "side", True,  (6, 6), "clothed west 36 (new)"),
    ("nE1", "8d1a674b", "side", False, (6, 6), "nude east 36 (=walkpng)"),
    ("nE2", "9aa81035", "side", False, (7, 7), "nude east 49 (new)"),
    ("nE3", "7ca1f857", "side", False, (8, 8), "nude east 64 (new)"),
    ("cS1", "5c6882bc", "front", True,  (6, 6), "clothed south 36"),
    ("cS2", "aaf3bf24", "front", True,  (6, 6), "clothed south 36"),
    ("nS1", "4b55fd4a", "front", False, (6, 6), "nude south 36"),
    ("nS2", "2470743b", "front", False, (6, 6), "nude south 36"),
]


def find_file(prefix):
    for f in os.listdir(UP):
        if f.startswith(prefix):
            return UP / f
    raise FileNotFoundError(prefix)


def runs(v):
    v = np.r_[0, v.astype(int), 0]; d = np.diff(v)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def slice_sheet(a, expected):
    m = a[..., 3] > 0
    cols = runs(m.any(0)); rows = runs(m.any(1))
    ec, er = expected
    if len(cols) != ec:
        step = a.shape[1] / ec; cols = [(int(i * step), int((i + 1) * step)) for i in range(ec)]
    if len(rows) != er:
        step = a.shape[0] / er; rows = [(int(i * step), int((i + 1) * step)) for i in range(er)]
    cells = []
    for r0, r1 in rows:
        for c0, c1 in cols:
            sub = a[r0:r1, c0:c1]
            sm = sub[..., 3] > 0
            if sm.sum() < 2000:
                continue
            ys, xs = np.nonzero(sm)
            cells.append(sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy())
    return cells


def tail_mask_side(a, m):
    """측면 꼬리 (방향 무관): 털색 & 허리 아래 & 몸통 중심에서 0.12H 이상 떨어진 쪽. 가장 큰 덩어리."""
    rgb = a[..., :3].astype(int); v = rgb.max(2); r, b = rgb[..., 0], rgb[..., 2]
    ys, xs = np.nonzero(m); top, bot = ys.min(), ys.max(); H = bot - top + 1
    t = (np.arange(a.shape[0]) - top) / H
    torso = m & (t[:, None] > 0.28) & (t[:, None] < 0.45)
    xc = np.median(np.nonzero(torso)[1])
    fur = (v > 140) & ((r - b) < 34)
    X = np.arange(a.shape[1])[None, :]
    cand = m & fur & (t[:, None] > 0.36) & (t[:, None] < 0.95) & (np.abs(X - xc) > 0.12 * H)
    lab, n = cc_label(binary_dilation(cand, iterations=2) & m)
    if n == 0:
        return np.zeros_like(m), xc, H, 0
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    tl = lab == sizes.argmax()
    side = np.sign(np.nonzero(tl)[1].mean() - xc)      # +1 꼬리가 오른쪽
    # 행 채움 (꼬리 안쪽 그림자·겹친 부분)
    for y in np.nonzero(tl.any(1))[0]:
        xx = np.nonzero(tl[y])[0]
        if side > 0:
            lo = max(xx.min(), int(xc + 0.02 * H)); tl[y, lo:xx.max() + 1] = m[y, lo:xx.max() + 1]
        else:
            hi = min(xx.max(), int(xc - 0.02 * H)); tl[y, xx.min():hi + 1] = m[y, xx.min():hi + 1]
    return tl, xc, H, int(side)


def normalize(a, view):
    a, removed = debg.clean(a)
    m = a[..., 3] > 128
    info = dict(black_residue=int(removed))
    if view == "side":
        tl, xc, H0, side = tail_mask_side(a, m)
        facing = "west" if side > 0 else "east"          # 꼬리가 오른쪽이면 왼쪽을 본다
        if facing == "west":
            a = a[:, ::-1].copy(); m = m[:, ::-1]; tl = tl[:, ::-1]
        info.update(facing_original=facing, mirrored=facing == "west")
        body = m & ~tl
    else:
        info.update(facing_original="south", mirrored=False)
        body = m
    ys, xs = np.nonzero(body)
    top, bot = ys.min(), ys.max(); H = bot - top + 1
    t = (ys - top) / H
    head = t < 0.2
    hx = xs[head].mean()
    s = H_REF / H
    new = Image.fromarray(a).resize((max(1, int(round(a.shape[1] * s))), max(1, int(round(a.shape[0] * s)))), Image.LANCZOS)
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    ox = int(round(HEAD_X - hx * s)); oy = int(round(SOLE_Y - bot * s))
    canvas.alpha_composite(new, (ox, oy)) if (0 <= ox < CANVAS[0] and 0 <= oy < CANVAS[1]) else canvas.paste(new, (ox, oy), new)
    info.update(H_px=int(H), scale=round(float(s), 4), head_x_src=round(float(hx), 1), sole_y_src=int(bot),
                canvas=list(CANVAS), head_x=HEAD_X, sole_y=SOLE_Y, H_ref=H_REF)
    return canvas, info


def main():
    (W / "frames").mkdir(parents=True, exist_ok=True); (W / "norm").mkdir(exist_ok=True)
    labels = []
    for sid, prefix, view, clothed, grid, note in SHEETS:
        f = find_file(prefix)
        a = np.array(Image.open(f).convert("RGBA"))
        cells = slice_sheet(a, grid)
        for i, c in enumerate(cells):
            fid = f"{sid}_{i:02d}"
            Image.fromarray(c).save(W / "frames" / f"{fid}.png")
            canvas, info = normalize(c, view)
            canvas.save(W / "norm" / f"{fid}.png")
            labels.append(dict(id=fid, set=sid, index=i, source=f.name, note=note, view=view,
                               clothed=clothed, src_size=[int(c.shape[1]), int(c.shape[0])], **info))
        fac = [l["facing_original"] for l in labels if l["set"] == sid]
        print(f"{sid:4s} {note:28s} frames={len(cells):3d}  facing={ {x: fac.count(x) for x in set(fac)} }")
    json.dump(labels, open(W / "labels.json", "w"), indent=1, ensure_ascii=False)
    print("total frames:", len(labels))


if __name__ == "__main__":
    main()
