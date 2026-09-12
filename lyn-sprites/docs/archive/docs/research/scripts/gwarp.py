"""원화에서 옷을 잘라 **해부 좌표로 워프해** 붙인다. 색칠이 아니라 화소 이식.

1판 — (t,u) 240x128 격자에 색 중앙값
    칸 하나가 몇 화소라 셔츠 무늬·찢어진 레깅스·버클이 계단으로 뭉개졌다.
    격자는 표현이 아니라 손실이었다.            IoU 0.762

2판 — 격자를 버리고 행마다 u 를 1차원 되샘플
    무늬와 버클이 살아났다.                     IoU 0.829
    그런데 다리가 깨졌다. 가랑이 아래 한 행에는 **다리가 둘**인데 u 하나로
    이으면 왼다리 화소가 틈을 건너 오른다리로 번진다.

3판 (이 파일) — 행을 **덩어리(run)** 로 쪼개고 덩어리끼리 맞춘다
    한 행의 화소를 끊긴 구간으로 나눈다. 가랑이 위는 [팔|몸통|팔], 아래는 [다리|다리].
    레퍼런스의 같은 덩어리를 찾아 그 안에서만 0~1 로 되샘플한다.
    **틈을 건너뛰지 않으므로 다리가 섞이지 않는다.** 부위 리본의 1차원 판이다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/claude/handoff/code")
sys.path.insert(0, "/home/claude/handoff/work")

ROWS = 512
GAP = 2                # 이만큼 떨어지면 다른 덩어리


def _rows_runs(mask, t, rows=ROWS):
    """행마다 끊긴 구간으로 나누고, 화소별 (행, 덩어리, 구간내 0~1, 덩어리 중심)."""
    ys, xs = np.nonzero(mask)
    tv = t[ys, xs]
    ti = np.clip((tv * (rows - 1)).astype(int), 0, rows - 1)
    order = np.lexsort((xs, ti))
    ys, xs, ti = ys[order], xs[order], ti[order]

    run = np.zeros(len(xs), np.int32)
    v = np.zeros(len(xs), np.float32)
    cen = np.zeros(len(xs), np.float32)
    per_row = {}
    start = 0
    while start < len(ti):
        end = start
        while end + 1 < len(ti) and ti[end + 1] == ti[start]:
            end += 1
        sl = slice(start, end + 1)
        x = xs[sl]
        brk = np.nonzero(np.diff(x) > GAP)[0]
        bounds = [0] + (brk + 1).tolist() + [len(x)]
        lo, hi = float(x.min()), float(x.max())
        half = max((hi - lo) / 2, 1e-6)
        mid = (lo + hi) / 2
        info = []
        for r in range(len(bounds) - 1):
            a, b = bounds[r], bounds[r + 1]
            seg = x[a:b]
            x0, x1 = float(seg[0]), float(seg[-1])
            run[start + a:start + b] = r
            v[start + a:start + b] = (seg - x0) / max(x1 - x0, 1e-6)
            c = ((x0 + x1) / 2 - mid) / half
            cen[start + a:start + b] = c
            info.append(c)
        per_row[int(ti[start])] = info
        start = end + 1
    return ys, xs, ti, run, v, cen, per_row


def build(ref_rgba, ref_mask, ref_t, ref_garment, ref_tail=None, rows=ROWS):
    """레퍼런스를 '행 x 덩어리' 대응표로. 의상 한 벌이 이 표 하나다.

    꼬리가 가린 자리는 **같은 행의 거울 덩어리로 메운다.** 정면에서 u 는 몸의 좌우이고
    옷은 좌우대칭이므로, 꼬리가 덮은 다리의 옷은 반대쪽 다리에 그대로 있다.
    (몸통 좌우대칭 IoU 0.924 로 실측했다.)
    메우지 않으면 그 행의 옷 정보가 비어 대상 프레임의 허벅지가 맨살로 남는다 —
    실제로 레퍼런스 t 0.6~0.8 의 21~32% 를 꼬리가 덮어 그 일이 났다.

    행 전체를 통째로 뒤집던 4판은 **왼다리의 거울이 망토 자락이 되는** 일이 있어
    허벅지에 검은 덩어리가 찍혔다. 5판은 **덩어리(다리)끼리** 짝지어 뒤집는다 —
    왼다리의 거울은 오른다리다.
    """
    ys, xs, ti, run, v, cen, info = _rows_runs(ref_mask, ref_t, rows)
    g = ref_garment[ys, xs].astype(np.float32)
    sx = xs.astype(np.float32); sy = ys.astype(np.float32)
    if ref_tail is not None and ref_tail.any():
        tl = ref_tail[ys, xs]
        for k in np.unique(ti[tl]):
            row = ti == k
            if not (row & tl).any():
                continue
            lo, hi = xs[row].min(), xs[row].max()
            half = max((hi - lo) / 2, 1e-6); mid = (lo + hi) / 2
            # 덩어리 목록과 각 덩어리의 정규화 중심
            rl = sorted(set(run[row].tolist()))
            info_r = {}
            for r in rl:
                q = row & (run == r)
                x0, x1 = float(xs[q].min()), float(xs[q].max())
                info_r[r] = ((x0 + x1) / 2 - mid) / half
            for r in rl:
                q = row & (run == r)
                hurt = q & tl
                if not hurt.any():
                    continue
                # 거울 짝 덩어리 = 중심이 -c 에 가장 가까운, 꼬리가 덜 덮은 덩어리
                c = info_r[r]
                cands = [(abs(info_r[o] + c), o) for o in rl
                         if o != r and (row & (run == o) & ~tl).sum() >= 4]
                if not cands:
                    cands = [(abs(info_r[o] + c), o) for o in rl
                             if (row & (run == o) & ~tl).sum() >= 4]
                if not cands:
                    continue
                o = min(cands)[1]
                src = row & (run == o) & ~tl
                oo = np.argsort(v[src])
                vs = v[src][oo]
                sxs = xs[src][oo].astype(np.float32); sys_ = ys[src][oo].astype(np.float32)
                gs = g[src][oo]
                vm = 1.0 - v[hurt]                     # 덩어리 안에서 좌우 뒤집기
                sx[hurt] = np.interp(vm, vs, sxs)
                sy[hurt] = np.interp(vm, vs, sys_)
                g[hurt] = np.interp(vm, vs, gs)
    table = {}
    for k in range(rows):
        s = ti == k
        if not s.any():
            continue
        entry = []
        for r in range(run[s].max() + 1):
            q = s & (run == r)
            if q.sum() < 2:
                entry.append(None); continue
            o = np.argsort(v[q])
            entry.append((v[q][o], sx[q][o], sy[q][o], g[q][o]))
        # 행 전체를 하나로 본 좌표도 담는다 — 실루엣 바깥을 채울 때 쓴다 (gwarp2)
        lo, hi = float(xs[s].min()), float(xs[s].max())
        half = max((hi - lo) / 2, 1e-6); mid = (lo + hi) / 2
        gu = (xs[s] - mid) / half
        oo = np.argsort(gu)
        table[k] = dict(runs=entry, cen=info.get(k, []),
                        whole=(gu[oo], sx[s][oo], sy[s][oo], g[s][oo]))
    keys = sorted(table)
    near = {}
    for k in range(rows):
        near[k] = k if k in table else min(keys, key=lambda j: abs(j - k))
    return dict(rows=rows, table=table, near=near, rgba=ref_rgba)


def _lookup(ref, tgt_mask, tgt_t):
    rows = ref["rows"]
    ys, xs, ti, run, v, cen, _ = _rows_runs(tgt_mask, tgt_t, rows)
    n = len(ys)
    rx = np.zeros(n, np.float32); ry = np.zeros(n, np.float32); gv = np.zeros(n, np.float32)
    ok = np.zeros(n, bool)
    for k in np.unique(ti):
        ent = ref["table"][ref["near"][int(k)]]
        runs, rcen = ent["runs"], ent["cen"]
        avail = [i for i, r in enumerate(runs) if r is not None]
        if not avail:
            continue
        s = ti == k
        for r in np.unique(run[s]):
            q = s & (run == r)
            c = float(cen[q][0])
            # 같은 자리의 덩어리를 고른다 — 정규화된 중심이 가장 가까운 것
            j = min(avail, key=lambda i: abs(rcen[i] - c) if i < len(rcen) else 9.0)
            ur, xr, yr, gr = runs[j]
            rx[q] = np.interp(v[q], ur, xr)
            ry[q] = np.interp(v[q], ur, yr)
            gv[q] = np.interp(v[q], ur, gr)
            ok[q] = True
    return ys, xs, rx, ry, gv, ok


def sample(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0 = x.astype(int); y0 = y.astype(int)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    a = img[y0, x0].astype(np.float32); b = img[y0, x0 + 1].astype(np.float32)
    c = img[y0 + 1, x0].astype(np.float32); d = img[y0 + 1, x0 + 1].astype(np.float32)
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def wear(ref, tgt_rgba, tgt_mask, tgt_t, tgt_tail, gthresh=0.5, shade=True):
    ys, xs, rx, ry, gv, ok = _lookup(ref, tgt_mask, tgt_t)
    sel = ok & (gv > gthresh) & ~tgt_tail[ys, xs]
    out = tgt_rgba.copy()
    if not sel.any():
        return out, 0
    new = sample(ref["rgba"][..., :3], rx[sel], ry[sel])
    if shade:
        import cv2
        L = cv2.cvtColor(tgt_rgba[..., :3], cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        lv = L[ys[sel], xs[sel]]
        k = np.clip(lv / max(float(np.median(lv)), 1e-6), 0.90, 1.10)[:, None]
        new = np.clip(new * k, 0, 255)
    out[ys[sel], xs[sel], :3] = new.astype(np.uint8)
    return out, int(sel.sum())


def predicted_mask(ref, tgt_mask, tgt_t, tgt_tail, gthresh=0.5):
    ys, xs, _, _, gv, ok = _lookup(ref, tgt_mask, tgt_t)
    out = np.zeros_like(tgt_mask)
    sel = ok & (gv > gthresh) & ~tgt_tail[ys, xs]
    out[ys[sel], xs[sel]] = True
    return out
