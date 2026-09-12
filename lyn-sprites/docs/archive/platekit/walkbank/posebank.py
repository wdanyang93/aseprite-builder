"""posebank — 정규화된 측면 프레임 전부의 자세를 잰다 (옷과 무관한 신호만).

정규화 캔버스는 머리 중심 x=330, 발바닥 y=680, H=600 이므로 좌표를 그대로 비교할 수 있다.

프레임마다
  legs   spread(앞발−뒷발)/H, xf, xb (머리 기준), lift_f, lift_b, together
  arms   hand_f = 손 띠(t .46~.62) 앞 끝 − 엉덩이 띠(t .40~.46) 앞 끝 (앞으로 나온 손)
         hand_b = 엉덩이 띠 뒤 끝 − 손 띠 뒤 끝 (뒤로 간 손)   ※ 착의는 망토 때문에 hand_b 가 약하다
         arm_spread = hand_f + hand_b
  lean   머리 중심 − 발 중점
  NF     앞다리−뒷다리 밝기 (나체 피부 / 착의 레깅스) → 봉우리에서 N(가까운 다리 앞)/F
  theta  걸음 안 진행도 (sync.py 규칙), phi = theta/2 (N 뒤) 또는 .5+theta/2 (F 뒤)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/claude/handoff/work")
import bank
import pose

B = Path("/home/claude/handoff/work/bank")


def load_norm(fid):
    a = np.array(Image.open(B / "norm" / f"{fid}.png").convert("RGBA"))
    m = a[..., 3] > 128
    tl, xc, H0, side = bank.tail_mask_side(a, m)
    body = m & ~tl
    ys, xs = np.nonzero(body)
    top, bot = ys.min(), ys.max(); H = bot - top + 1
    t = (np.arange(a.shape[0]) - top) / H
    return dict(a=a, m=m, tail=tl, body=body, t=t, xc=float(xc), top=int(top), bot=int(bot), H=int(H))


def describe(fr, clothed):
    body, t, xc, H, bot, a = fr["body"], fr["t"], fr["xc"], fr["H"], fr["bot"], fr["a"]
    ft = pose.feet(body, t, xc, H, bot)
    def ext(lo, hi, side):
        band = body & (t[:, None] > lo) & (t[:, None] < hi)
        xs = np.nonzero(band)[1]
        if not len(xs): return np.nan
        return ((np.percentile(xs, 99) if side > 0 else np.percentile(xs, 1)) - xc) / H
    hand_f = ext(0.46, 0.62, +1) - ext(0.40, 0.46, +1)
    hand_b = ext(0.40, 0.46, -1) - ext(0.46, 0.62, -1)
    head = body & (t[:, None] < 0.2); hx = np.nonzero(head)[1].mean()
    d = dict(spread=ft["spread"] if ft else np.nan, xf=ft["xf"] if ft else np.nan, xb=ft["xb"] if ft else np.nan,
             lift_f=ft["lift_f"] if ft else 0.0, lift_b=ft["lift_b"] if ft else 0.0,
             together=bool(ft["together"]) if ft else True,
             hand_f=float(hand_f), hand_b=float(hand_b), arm_spread=float(hand_f + hand_b),
             lean=float((hx - xc) / H))
    # NF shade
    if ft is None or ft["together"]:
        d["shade"] = np.nan
    else:
        band = body & (t[:, None] > 0.62) & (t[:, None] < 0.86)
        rgb = a[..., :3].astype(int); v = rgb.max(2); s = rgb.max(2) - rgb.min(2)
        band &= (v < 140) if clothed else ((s >= 45) & (v > 150))
        ys, xs = np.nonzero(band)
        mid = xc + (ft["xf"] + ft["xb"]) / 2 * H
        fr_, bk = v[ys, xs][xs > mid], v[ys, xs][xs <= mid]
        d["shade"] = float(np.median(fr_) - np.median(bk)) if len(fr_) > 30 and len(bk) > 30 else np.nan
    return d


def phases_for_set(rows):
    """rows: 같은 시트의 프레임(순서대로). sync.phases 와 같은 규칙."""
    n = len(rows)
    spread = np.array([r["spread"] for r in rows]); together = np.array([r["together"] for r in rows])
    lf = np.array([r["lift_f"] for r in rows]); lb = np.array([r["lift_b"] for r in rows])
    shade = np.array([r["shade"] for r in rows])
    smax = np.nanpercentile(spread, 95); sn = np.clip(spread / smax, 0, 1)
    dsp = np.gradient(np.nan_to_num(spread))
    theta = np.zeros(n); peak = (spread >= 0.75 * smax) & ~together
    for k in range(n):
        if together[k]: theta[k] = 0.5; continue
        if peak[k]: theta[k] = 0.0; continue
        if lb[k] > lf[k] + 0.015: d = -1
        elif lf[k] > lb[k] + 0.015: d = +1
        else: d = int(np.sign(dsp[k])) or -1
        theta[k] = 0.5 - 0.5 * sn[k] if d < 0 else 0.5 + 0.5 * sn[k]
    # 봉우리 run 단위 라벨 + N/F 교대 강제 (한 프레임의 약한 밝기 오판이 뒤 교차 프레임까지 뒤집는 것을 막는다)
    runs_ = []; k = 0
    while k < n:
        if peak[k]:
            j = k
            while j + 1 < n and peak[j + 1]: j += 1
            vals = [shade[i] for i in range(k, j + 1) if not np.isnan(shade[i])]
            runs_.append([k, j, float(np.mean(vals)) if vals else np.nan]); k = j + 1
        else:
            k += 1
    means = np.array([r[2] for r in runs_ if not np.isnan(r[2])])
    thr = float((means.min() + means.max()) / 2) if len(means) >= 2 else 0.0
    rl = ["N" if (not np.isnan(r[2]) and r[2] > thr) else "F" for r in runs_]
    for i in range(1, len(rl)):                    # 교대 강제: 같은 라벨이 이어지면 임계에 더 가까운 쪽을 뒤집는다
        if rl[i] == rl[i - 1]:
            di = abs(runs_[i][2] - thr) if not np.isnan(runs_[i][2]) else 0
            dp = abs(runs_[i - 1][2] - thr) if not np.isnan(runs_[i - 1][2]) else 0
            flip = i if di <= dp else i - 1
            rl[flip] = "F" if rl[flip] == "N" else "N"
    lab = np.array(["?"] * n, dtype=object)
    for (k0, k1, _), L in zip(runs_, rl):
        lab[k0:k1 + 1] = L
    cur = None
    for k in range(n):
        if lab[k] != "?": cur = lab[k]
        elif cur is not None: lab[k] = cur
    first = next((i for i in range(n) if lab[i] != "?"), None)
    if first is not None:
        for k in range(first): lab[k] = "F" if lab[first] == "N" else "N"
    phi = np.where(lab == "N", theta / 2, 0.5 + theta / 2) % 1.0
    order = float(np.corrcoef(np.nan_to_num(spread[:-1]), np.nan_to_num(spread[1:]))[0, 1]) if n > 2 else 0
    return theta, lab, phi, order, float(smax), float(thr)


def main():
    labels = json.load(open(B / "labels.json"))
    side = [l for l in labels if l["view"] == "side"]
    out = {}
    for sid in sorted({l["set"] for l in side}):
        ids = [l["id"] for l in side if l["set"] == sid]
        clothed = [l for l in side if l["set"] == sid][0]["clothed"]
        rows = []
        for fid in ids:
            fr = load_norm(fid); d = describe(fr, clothed); d["id"] = fid; rows.append(d)
        theta, lab, phi, order, smax, thr = phases_for_set(rows)
        for r, th, lb, ph in zip(rows, theta, lab, phi):
            r.update(theta=float(th), NF=str(lb), phi=float(ph), set=sid, clothed=clothed)
            out[r["id"]] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        print(f"{sid}: n={len(rows)} order_lag1={order:.2f} smax={smax:.3f} NF={''.join(lab)}")
    json.dump(out, open(B / "pose.json", "w"), indent=1, ensure_ascii=False)
    print("frames:", len(out))


if __name__ == "__main__":
    main()
