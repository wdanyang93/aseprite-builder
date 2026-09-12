"""pose — 측면 걸음 프레임 한 장의 '자세'를 숫자로 잰다 (옷과 무관한 신호만).

세트끼리 동기화하려면 프레임 번호가 아니라 자세로 맞춰야 한다. 그러려면 옷을 입었든
아니든 같은 뜻으로 읽히는 신호가 필요하다.

  feet   발 띠(t>0.88) 의 덩어리 → 앞발/뒷발 각각 x 중심, 발끝, 발바닥 y
         spread = (앞발 x − 뒷발 x)/H            걸음의 '벌어짐'
         lift_f, lift_b = 발바닥이 바닥에서 뜬 높이/H  어느 발이 스윙 중인가
  hand   손 띠(t 0.46~0.62) 앞쪽 실루엣 끝 − 엉덩이 띠 앞쪽 끝  → 앞으로 나온 손
         (나체 = 손, 착의 = 장갑; 둘 다 앞으로 흔들면 실루엣 앞으로 튀어나온다)
  near   화면 쪽(가까운) 다리가 앞인가 뒤인가 — 앞다리·뒷다리의 밝기 차.
         AI 원화는 먼 다리를 어둡게 칠한다. 이 신호가 spread 봉우리마다 부호를
         바꾸면 신뢰하고, 아니면 '모름' 으로 둔다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import label

import debg
import garmentstudy as G

FOOT_T = 0.88
HAND_T = (0.46, 0.62)
HIP_T = (0.40, 0.46)
LEG_T = (0.66, 0.84)


def load(p):
    a = np.array(Image.open(p).convert("RGBA")); a, _ = debg.clean(a)
    m = a[..., 3] > 128
    tl, xc, top, H = G.tail_mask(a, m)
    body = m & ~tl
    ys, xs = np.nonzero(body)
    top, bot = ys.min(), ys.max(); H = bot - top + 1
    t = (np.arange(a.shape[0]) - top) / H
    return a, body, t, float(xc), int(top), int(H), int(bot)


def feet(body, t, xc, H, bot):
    band = body & (t[:, None] > FOOT_T)
    lab, n = label(band)
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 0.0005 * H * H:
            continue
        sole = ys.max()
        base = ys > sole - 0.03 * H                   # 발바닥 근처 3% 만으로 x 를 잰다
        blobs.append(dict(x=float(xs[base].mean()), xmin=int(xs.min()), xmax=int(xs.max()),
                          sole=int(sole), n=int(len(xs))))
    blobs.sort(key=lambda b: b["x"])
    if not blobs:
        return None
    if len(blobs) == 1:                               # 두 발이 겹침 → 하나의 덩어리
        b = blobs[0]
        w = (b["xmax"] - b["xmin"]) / H
        return dict(spread=w, xf=(b["xmax"] - xc) / H, xb=(b["xmin"] - xc) / H,
                    lift_f=(bot - b["sole"]) / H, lift_b=(bot - b["sole"]) / H, together=True)
    fb, ff = blobs[0], blobs[-1]
    return dict(spread=(ff["x"] - fb["x"]) / H, xf=(ff["xmax"] - xc) / H, xb=(fb["xmin"] - xc) / H,
                lift_f=(bot - ff["sole"]) / H, lift_b=(bot - fb["sole"]) / H, together=False)


def hand(body, t, xc, H):
    def front_ext(lo, hi):
        band = body & (t[:, None] > lo) & (t[:, None] < hi)
        xs = np.nonzero(band)[1]
        return (np.percentile(xs, 99) - xc) / H if len(xs) else np.nan
    return front_ext(*HAND_T) - front_ext(*HIP_T)


def near_leg(a, body, t, xc, H, ft):
    """앞다리 − 뒷다리 밝기 (V). 양수 = 앞다리가 밝다 = 앞다리가 가까운(화면 쪽) 다리일 가능성."""
    if ft is None or ft["together"]:
        return np.nan
    band = body & (t[:, None] > LEG_T[0]) & (t[:, None] < LEG_T[1])
    ys, xs = np.nonzero(band)
    if len(xs) < 50:
        return np.nan
    v = a[..., :3].max(2)[ys, xs].astype(float)
    mid = xc + (ft["xf"] + ft["xb"]) / 2 * H
    fr, bk = v[xs > mid], v[xs <= mid]
    if len(fr) < 30 or len(bk) < 30:
        return np.nan
    return float(np.median(fr) - np.median(bk))


def describe(p):
    a, body, t, xc, top, H, bot = load(p)
    ft = feet(body, t, xc, H, bot)
    return dict(path=p, H=H, xc=xc, top=top, bot=bot,
                spread=ft["spread"] if ft else np.nan,
                xf=ft["xf"] if ft else np.nan, xb=ft["xb"] if ft else np.nan,
                lift_f=ft["lift_f"] if ft else np.nan, lift_b=ft["lift_b"] if ft else np.nan,
                together=bool(ft["together"]) if ft else True,
                hand=float(hand(body, t, xc, H)),
                near=near_leg(a, body, t, xc, H, ft))
