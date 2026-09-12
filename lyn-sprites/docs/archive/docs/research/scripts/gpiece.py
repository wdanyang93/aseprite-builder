"""7판 — 옷을 통째로 옮긴다. 잘라 붙이지 않는다.

무엇이 틀렸었나
    1~6판은 옷을 **행 단위로 해체해** 되샘플했다. 두 가지가 따라왔다.
      · 무늬가 뭉개진다 — 한 벌의 천이 수백 개의 가로줄로 쪼개진다
      · **중복** — 대상의 두 덩어리가 레퍼런스의 같은 덩어리를 각각 가져가
        옷이 두 번 찍힌다. 화면이 엉망이 된 주범이다.

실측 — 해체할 필요가 없었다
    옷을 t 띠로 **조각**만 내고 조각마다 평행이동만 준 상한:

        상의묶음(망토·셔츠·하네스·벨트·장갑)   IoU 0.826 ± 0.084
        행별 워프 전체                        IoU 0.825

    **몸통 옷은 워프가 필요 없다.** 정면 걷기에서 상체는 사실상 강체다.
    해체는 순손실이었다.

    다리는 다르다 (레깅스 0.46, 부츠 0.45 — 두 다리를 한 조각으로 묶었을 때).
    다리만 따로 다뤄야 한다.

이 판
    상의묶음   통짜 + 평행이동          (자르지 않음, 중복 없음)
    다리부     덩어리 **1:1** 매칭 워프  (같은 레퍼런스 덩어리를 두 번 쓰지 않는다)
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/claude/handoff/code")
sys.path.insert(0, "/home/claude/handoff/work")

import gwarp

TORSO_END = 0.62          # 이 위는 통짜, 아래는 다리


# ── 몸 기준점 ────────────────────────────────────────────────────────────
def sagittal_x(mask, t, lo=0.27, hi=0.48):
    band = mask & (t >= lo) & (t <= hi)
    ys, xs = np.nonzero(band)
    if len(xs) < 20:
        return float(np.nonzero(mask)[1].mean())
    best = (0.0, float(xs.mean()))
    W = mask.shape[1]
    g = np.zeros_like(band); g[ys, xs] = True
    for cx in np.arange(xs.mean() - 8, xs.mean() + 8.01, 0.5):
        fl = np.roll(g[:, ::-1], int(round(2 * cx)) - (W - 1), axis=1)
        s = (g & fl).sum() / max((g | fl).sum(), 1)
        if s > best[0]:
            best = (s, float(cx))
    return best[1]


def row_at_t(mask, t, tv=0.40):
    ys, xs = np.nonzero(mask)
    sel = np.abs(t[ys, xs] - tv) < 0.01
    return float(ys[sel].mean()) if sel.sum() >= 10 else float(ys.mean())


def anchors(mask, t):
    return sagittal_x(mask, t), row_at_t(mask, t, 0.40)


# ── 조각 ────────────────────────────────────────────────────────────────
def torso_piece(rgba, garment, t):
    return garment & (t < TORSO_END)


def paste(dst_rgba, src_rgba, src_mask, dx, dy):
    """통짜 조각을 평행이동해 얹는다. 화소는 한 번씩만 찍힌다."""
    H, W = dst_rgba.shape[:2]
    ys, xs = np.nonzero(src_mask)
    Y = np.round(ys + dy).astype(int); X = np.round(xs + dx).astype(int)
    ok = (Y >= 0) & (Y < H) & (X >= 0) & (X < W)
    out = dst_rgba.copy()
    out[Y[ok], X[ok], :3] = src_rgba[ys[ok], xs[ok], :3]
    out[Y[ok], X[ok], 3] = 255
    m = np.zeros((H, W), bool); m[Y[ok], X[ok]] = True
    return out, m


# ── 다리부: 중복 없는 1:1 덩어리 매칭 ─────────────────────────────────────
def assign_runs(tgt_centers, ref_centers):
    """대상 덩어리 ↔ 레퍼런스 덩어리를 **일대일**로 짝짓는다.

    기존에는 대상 덩어리마다 가장 가까운 레퍼런스 덩어리를 독립으로 골랐다.
    두 다리가 같은 레퍼런스 다리를 가져가 **옷이 두 번 찍혔다.**
    수가 다르면 레퍼런스 덩어리를 대상 수만큼 **비례로 쪼개** 나눠 준다.
    """
    nt, nr = len(tgt_centers), len(ref_centers)
    ti = np.argsort(tgt_centers)
    ri = np.argsort(ref_centers)
    out = {}
    if nt <= nr:
        for a, b in zip(ti, ri[:nt]):
            out[int(a)] = (int(b), 0.0, 1.0)
        return out
    # 대상이 더 많다 — 레퍼런스를 쪼갠다
    per = nt / nr
    for n, a in enumerate(ti):
        b = min(int(n / per), nr - 1)
        k = n - int(b * per)
        cnt = max(1, int(round(per)))
        lo = min(k / cnt, 1.0); hi = min((k + 1) / cnt, 1.0)
        out[int(a)] = (int(ri[b]), lo, hi)
    return out


def wear_legs(ref, tgt_rgba, tgt_mask, tgt_t, tgt_tail, gthresh=0.5):
    rows = ref["rows"]
    ys, xs, ti, run, v, cen, _ = gwarp._rows_runs(tgt_mask & (tgt_t >= TORSO_END),
                                                  tgt_t, rows)
    out = tgt_rgba.copy()
    painted = np.zeros_like(tgt_mask)
    if len(ys) == 0:
        return out, painted
    px, py, sx, sy = [], [], [], []
    for k in np.unique(ti):
        ent = ref["table"][ref["near"][int(k)]]
        runs, rcen = ent["runs"], ent["cen"]
        avail = [i for i, r in enumerate(runs) if r is not None]
        if not avail:
            continue
        s = ti == k
        tr = sorted(set(run[s].tolist()))
        tcen = [float(cen[s & (run == r)][0]) for r in tr]
        rcen_a = [rcen[i] if i < len(rcen) else 9.0 for i in avail]
        amap = assign_runs(tcen, rcen_a)
        for idx, r in enumerate(tr):
            if idx not in amap:
                continue
            j, lo, hi = amap[idx]
            ur, xr, yr, gr = runs[avail[j]]
            q = s & (run == r)
            vv = lo + v[q] * (hi - lo)          # 쪼갠 구간
            g = np.interp(vv, ur, gr)
            m = (g > gthresh) & ~tgt_tail[ys[q], xs[q]]
            if not m.any():
                continue
            px.append(xs[q][m]); py.append(ys[q][m])
            sx.append(np.interp(vv[m], ur, xr)); sy.append(np.interp(vv[m], ur, yr))
    if not px:
        return out, painted
    px = np.concatenate(px); py = np.concatenate(py)
    sx = np.concatenate(sx); sy = np.concatenate(sy)
    out[py, px, :3] = gwarp.sample(ref["rgba"][..., :3], sx, sy).astype(np.uint8)
    painted[py, px] = True
    return out, painted


# ── 하의: 좌/우 두 조각 (바지는 원래 가랑이에서 둘로 갈린다) ─────────────────
def half_masks(mask, t, sx, lo=TORSO_END, hi=1.01):
    band = mask & (t >= lo) & (t < hi)
    X = np.arange(mask.shape[1])[None, :]
    return band & (X < sx), band & (X >= sx)


def centroid(m):
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def wear_lower_whole(dst_rgba, ref_rgba, ref_g, ref_body, ref_t, ref_sx,
                     tgt_mask, tgt_t, tgt_sx, tgt_tail):
    """하의를 좌/우 **통짜 두 조각**으로 옮긴다. 화소 재사용 0%.

    측정: 하의를 통짜 하나로 두면 IoU 0.469 가 천장이다 — 다리는 관절이 있어
    한 덩어리로 못 옮긴다. 좌/우로 가르면 0.583. 행별 워프는 0.749 지만
    원화 화소의 24.2% 를 두 번 이상 쓰고 최대 21회까지 늘여 쓴다.
    """
    rgL, rgR = half_masks(ref_g, ref_t, ref_sx)
    rbL, rbR = half_masks(ref_body, ref_t, ref_sx)
    tbL, tbR = half_masks(tgt_mask & ~tgt_tail, tgt_t, tgt_sx)
    out = dst_rgba.copy()
    painted = np.zeros(dst_rgba.shape[:2], bool)
    for rg, rb, tb in [(rgL, rbL, tbL), (rgR, rbR, tbR)]:
        c1, c2 = centroid(rb), centroid(tb)
        if c1 is None or c2 is None or tb.sum() < 200:
            continue
        out, m = paste(out, ref_rgba, rg, c2[0] - c1[0], c2[1] - c1[1])
        painted |= m
    return out, painted


# ── 8판: 통짜 조각 다섯 ───────────────────────────────────────────────────
# 이음매는 옷의 실제 경계에서 고른다. 실측으로 t=0.68 (셔츠단·허벅지주머니 아래),
# t=0.88 (부츠 목) 이 가장 좋았다.
#   이음매 0.55 → 0.724   0.62 → 0.744   0.68 → 0.751   0.72 → 0.748   0.85 → 0.730
SEAM_HIP, SEAM_BOOT = 0.68, 0.88


def split_pieces(garment, body, t, sx):
    """의상을 통짜 다섯 조각으로. 자르는 곳은 옷의 이음매뿐이다.

    t 로만 자르면 **망토 자락이 허리선에서 잘려** 다리와 함께 움직인다.
    실제로 그렇게 했더니 허벅지 옆에 망토 끈 조각이 따로 떠다녔다.
    그래서 자른 뒤, 상의 띠에 걸쳐 있는 덩어리는 통째로 상의에 돌려준다 —
    **한 장의 천은 끊지 않는다.**
    """
    import cv2
    X = np.arange(garment.shape[1])[None, :]
    L, R = X < sx, X >= sx
    k = np.ones((21, 21), np.uint8)
    out = []
    torso = garment & (t < SEAM_HIP)
    rest = garment & (t >= SEAM_HIP)
    pieces = []
    for lo, hi, nm in [(SEAM_HIP, SEAM_BOOT, "leg"), (SEAM_BOOT, 1.01, "boot")]:
        band = (t >= lo) & (t < hi)
        for side, s in (("L", L), ("R", R)):
            bb = body & band & s
            near = cv2.dilate(bb.astype(np.uint8), k) > 0      # 그 다리에 붙어 있는 천만
            pieces.append((f"{nm}{side}", rest & band & s & near, bb))
    taken = np.zeros_like(garment)
    for _, g, _ in pieces:
        taken |= g
    torso = torso | (rest & ~taken)      # 다리에 안 붙은 자락은 상의가 데려간다
    out.append(("torso", torso, body & (t < SEAM_HIP)))
    out.extend(pieces)
    return out


class Garment:
    """의상 한 벌 = 원화 한 장 + 통짜 조각 다섯 + 조각마다의 부착 기준점."""

    def __init__(self, rgba, mask, t, tail, garment):
        self.rgba = rgba
        body = mask & ~tail
        self.sx = sagittal_x(body, t)
        self.anchor_t = anchors(body, t)
        self.pieces = []
        for name, g, b in split_pieces(garment, body, t, self.sx):
            c = centroid(b)
            if g.sum() < 80 or c is None:
                continue
            self.pieces.append(dict(name=name, mask=g, ref=c))

    def _ref_of(self, name):
        for p in self.pieces:
            if p["name"] == name:
                return p["ref"]
        return None

    def wear(self, dst_rgba, tgt_mask, tgt_t, tgt_tail):
        body = tgt_mask & ~tgt_tail
        sx = sagittal_x(body, tgt_t)
        anc = anchors(body, tgt_t)
        by_name = {name: b for name, _, b in
                   split_pieces(np.zeros_like(body), body, tgt_t, sx)}
        out = dst_rgba.copy()
        painted = np.zeros(dst_rgba.shape[:2], bool)
        for p in self.pieces:
            if p["name"] == "torso":
                dx = anc[0] - self.anchor_t[0]; dy = anc[1] - self.anchor_t[1]
            else:
                c = centroid(by_name.get(p["name"]))
                if c is None:          # 발이 겹쳐 그쪽 띠가 비었다 — 같은 쪽 다리를 따른다
                    alt = "leg" + p["name"][-1] if p["name"].startswith("boot") else None
                    c = centroid(by_name.get(alt)) if alt else None
                    ref = self._ref_of(alt) if alt else None
                    if c is None or ref is None:
                        continue
                    dx = c[0] - ref[0]; dy = c[1] - ref[1]
                else:
                    dx = c[0] - p["ref"][0]; dy = c[1] - p["ref"][1]
            out, m = paste(out, self.rgba, p["mask"], dx, dy)
            painted |= m
        return out, painted
