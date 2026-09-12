"""pairs8 — 전체 측면 프레임(착의 143 / 나체 149)에서 팔·다리 벌어짐이 거의 일치하는 짝을 찾고,
한 바퀴 8위상에 맞는 짝 세트를 만든다.

자세 거리 (비율·배치 차이에 둔감한 것만; 전체 은행 표준편차로 표준화)
  spread(다리 벌어짐), lift_f, lift_b, hand_f, hand_b, arm_spread, theta
  + N/F 가 다르면 제외.
위상 i 의 짝 = |phi − i/8| ≤ 0.07 인 착의×나체 후보 중 거리 최소. 같은 프레임 재사용 금지.
그리고 위상마다 '다리가 가장 비슷한 착의 프레임'과 '팔이 가장 비슷한 착의 프레임'을 따로 기록해
조립(assemble.py) 이 팔·다리를 전체 프레임에서 가져오게 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

B = Path("/home/claude/handoff/work/bank")
NPH = 8
FEAT = ["spread", "lift_f", "lift_b", "hand_f", "hand_b", "arm_spread"]
LEG_FEAT = ["spread", "lift_f", "lift_b"]
ARM_FEAT = ["hand_f", "hand_b", "arm_spread"]


def load():
    P = json.load(open(B / "pose.json"))
    ids = sorted(P)
    X = np.array([[P[i][k] if P[i][k] is not None else np.nan for k in FEAT] for i in ids])
    sd = np.nanstd(X, 0) + 1e-6
    return P, ids, X / sd, sd


def cdist(P, ids, Xs, a, b, feats=FEAT):
    ia, ib = ids.index(a), ids.index(b)
    cols = [FEAT.index(f) for f in feats]
    d = np.nanmean(np.abs(Xs[ia, cols] - Xs[ib, cols]))
    dth = min(abs(P[a]["theta"] - P[b]["theta"]), 1 - abs(P[a]["theta"] - P[b]["theta"]))
    return float(d + 2.0 * dth)


def circ(a, b):
    d = abs(a - b); return min(d, 1 - d)


def main():
    P, ids, Xs, sd = load()
    clothed = [i for i in ids if P[i]["clothed"]]; nude = [i for i in ids if not P[i]["clothed"]]

    # 3. 거의 일치하는 짝 표: 착의 프레임마다 최근접 나체 3개
    near = {}
    for c in clothed:
        cand = [(cdist(P, ids, Xs, c, n), n) for n in nude if P[n]["NF"] == P[c]["NF"]]
        cand.sort(); near[c] = [(round(d, 3), n) for d, n in cand[:3]]

    # 4. 8위상 짝 (사용 프레임 중복 금지, 위상 순서대로 탐욕 + 최소)
    used_c, used_n = set(), set()
    pairs = []
    for i in range(NPH):
        tgt = i / NPH
        cc = [c for c in clothed if circ(P[c]["phi"], tgt) <= 0.07 and c not in used_c]
        nn = [n for n in nude if circ(P[n]["phi"], tgt) <= 0.07 and n not in used_n]
        best = None
        for c in cc:
            for n in nn:
                if P[n]["NF"] != P[c]["NF"]: continue
                d = cdist(P, ids, Xs, c, n)
                if best is None or d < best[0]: best = (d, c, n)
        if best is None:                                   # 완화: 위상 창 0.12
            for c in [c for c in clothed if circ(P[c]["phi"], tgt) <= 0.12 and c not in used_c]:
                for n in [n for n in nude if circ(P[n]["phi"], tgt) <= 0.12 and n not in used_n]:
                    if P[n]["NF"] != P[c]["NF"]: continue
                    d = cdist(P, ids, Xs, c, n)
                    if best is None or d < best[0]: best = (d, c, n)
        d, c, n = best; used_c.add(c); used_n.add(n)
        # 팔·다리 따로: 이 위상의 '기준 자세' = 짝의 나체 프레임 (몸 자체가 캐논) 에 가장 가까운 착의 프레임
        legs = sorted((cdist(P, ids, Xs, x, n, LEG_FEAT), x) for x in clothed if P[x]["NF"] == P[n]["NF"] and circ(P[x]["phi"], tgt) <= 0.10)
        arms = sorted((cdist(P, ids, Xs, x, n, ARM_FEAT), x) for x in clothed if P[x]["NF"] == P[n]["NF"] and circ(P[x]["phi"], tgt) <= 0.10)
        pairs.append(dict(phase=i, clothed=c, nude=n, dist=round(d, 3),
                          d_spread_px=round((P[c]["spread"] - P[n]["spread"]) * 600, 1),
                          d_arm_px=round((P[c]["arm_spread"] - P[n]["arm_spread"]) * 600, 1),
                          NF=P[c]["NF"], phi_c=round(P[c]["phi"], 3), phi_n=round(P[n]["phi"], 3),
                          leg_src=[(round(dd, 3), x) for dd, x in legs[:3]],
                          arm_src=[(round(dd, 3), x) for dd, x in arms[:3]]))
    json.dump(dict(pairs=pairs, nearest=near, feat_sd={k: float(v) for k, v in zip(FEAT, sd)}),
              open(B / "pairs8.json", "w"), indent=1, ensure_ascii=False)

    # 시트
    cw, ch = 200, 250
    im = Image.new("RGB", (20 + NPH * cw + 20, 60 + 2 * ch + 60), (250, 249, 246)); d = ImageDraw.Draw(im)
    d.text((14, 8), "8-PHASE PAIRS from the whole bank (clothed 143 / nude 149) — matched on leg spread + arm spread + lifts + theta, same near-leg",
           fill=(20, 20, 20))
    d.text((14, 24), "under each pair: spread diff px (H=600), arm-spread diff px, pose distance. Leg/arm donor frames listed in pairs8.json", fill=(90, 90, 90))
    for i, p in enumerate(pairs):
        x0 = 20 + i * cw
        for row, fid in enumerate((p["clothed"], p["nude"])):
            a = Image.open(B / "norm" / f"{fid}.png"); sc = (ch - 20) / a.height
            fr = a.resize((int(a.width * sc), int(a.height * sc)))
            y0 = 44 + row * ch
            im.paste(fr, (x0, y0 + 14), fr)
            d.text((x0, y0), f"phi {i}/8 {fid} {P[fid]['NF']} th={P[fid]['theta']:.2f}", fill=(20, 20, 20))
        d.text((x0, 44 + 2 * ch + 4), f"legs {p['d_spread_px']:+.0f}px arms {p['d_arm_px']:+.0f}px dist {p['dist']:.2f}", fill=(20, 20, 20))
        d.text((x0, 44 + 2 * ch + 18), f"leg-donor {p['leg_src'][0][1]}  arm-donor {p['arm_src'][0][1]}", fill=(90, 90, 90))
    im.save("/mnt/user-data/outputs/PAIRS8-bank.png")
    for p in pairs:
        print(f"phi {p['phase']}/8  {p['clothed']} <-> {p['nude']}  NF={p['NF']}  legs {p['d_spread_px']:+.0f}px  arms {p['d_arm_px']:+.0f}px  dist {p['dist']:.2f}  leg-donor {p['leg_src'][0][1]} arm-donor {p['arm_src'][0][1]}")
    ds = [p["dist"] for p in pairs]; print("mean pair dist", round(float(np.mean(ds)), 3))


if __name__ == "__main__":
    main()
