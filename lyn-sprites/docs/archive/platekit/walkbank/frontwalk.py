"""frontwalk — 정면(남쪽) 은행 (cS1/cS2 착의 72, nS1/nS2 나체 72) 에 같은 파이프라인을 적용한다.

정면 자세 신호 (lr36 에서 검증: 앞으로 나온 발이 더 넓고, 더 아래)
  fw   = footw_R − footw_L (H 정규화)      부호 = 어느 발이 앞 (화면 오른쪽 = 해부학적 왼발)
  dy   = sole_R − sole_L
  asym = 골반 아래 좌/우 질량 비대칭
  hand_r/hand_l = 손 띠(t .46~.62) 끝 − 엉덩이 띠 끝 (좌우 각각)
위상: 세트 순서(검증됨)에서 fw 의 기본파 → phi. phi=0 = 화면 오른발이 가장 앞. 두 세트에 같은 뜻.
짝: |phi−i/8|≤0.07 안에서 (fw, dy, asym, hand_r, hand_l) 거리 최소.
조립: 고정 상부(t<0.42, 착의 중립 프레임) + 기증 하부 통째 + 꼬리 통째, 허리 띠 x 정렬, 페더.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/handoff/work")
import assemble as A
import bank

B = A.B; OUT = A.OUT
NPH = 8


def load(fid):
    """id 끝에 'm' 이 붙으면 좌우반전본 (정면에서 반전 = 다리가 바뀐 반주기 뒤 자세)."""
    mirror = fid.endswith("m"); base = fid[:-1] if mirror else fid
    a = np.array(Image.open(B / "norm" / f"{base}.png").convert("RGBA"))
    if mirror:
        a = a[:, ::-1].copy()
        hx = 640 - 330; a = np.roll(a, 330 - hx, axis=1)      # 머리 중심을 다시 x=330 으로
    m = a[..., 3] > 128
    tl, xc, H0, side = bank.tail_mask_side(a, m)          # 정면도 꼬리는 몸 중심에서 떨어진 털 덩어리
    body = m & ~tl
    ys, xs = np.nonzero(body); top, bot = ys.min(), ys.max(); H = bot - top + 1
    t = (np.arange(a.shape[0]) - top) / H
    return dict(id=fid, a=a, m=m, tail=tl, body=body, t=np.repeat(t[:, None], a.shape[1], 1), xc=float(xc), top=int(top), bot=int(bot), H=int(H), tail_side=int(side))


def describe(fr):
    body, T, H, bot = fr["body"], fr["t"], fr["H"], fr["bot"]
    ys, xs = np.nonzero(body)
    t = T[ys, xs]
    torso = (t > 0.27) & (t < 0.48); sx = np.median(xs[torso])
    low = t > 0.70
    d = {}
    for side, sel in (("L", xs < sx), ("R", xs >= sx)):
        k = low & sel
        if k.sum() < 10: d[f"sole_{side}"] = np.nan; d[f"footw_{side}"] = 0; continue
        sole = ys[k].max(); foot = k & (ys > sole - 0.03 * H)
        d[f"sole_{side}"] = float(sole); d[f"footw_{side}"] = float(np.ptp(xs[foot])) / H
    lower = t > 0.62
    l = (lower & (xs < sx)).sum(); r = (lower & (xs >= sx)).sum()
    def ext(lo, hi, side):
        band = (t > lo) & (t < hi)
        if not band.any(): return np.nan
        return ((xs[band].max() if side > 0 else -xs[band].min()) - (sx if side > 0 else -sx)) / H
    return dict(fw=d["footw_R"] - d["footw_L"], dy=(d["sole_R"] - d["sole_L"]) / H,
                asym=(r - l) / max(r + l, 1),
                hand_r=ext(0.46, 0.62, +1) - ext(0.40, 0.46, +1), hand_l=ext(0.46, 0.62, -1) - ext(0.40, 0.46, -1),
                sx=float(sx), tail_side=fr["tail_side"])


FEAT = ["fw", "dy", "hand_r", "hand_l"]


def phases(rows):
    """서 있는 구간(|fw| 가 ±2 프레임 안에서 8px 미만) 은 idle 로 빼고, 걷는 구간만으로 기본파를 맞춘다."""
    fw = np.array([r["fw"] for r in rows]) * 600
    act = np.array([np.abs(fw[max(0, i - 2):i + 3]).max() >= 8 for i in range(len(fw))])
    for r, a in zip(rows, act): r["idle"] = bool(not a)
    idx = np.nonzero(act)[0]
    x = fw[idx] / 600; x = x - x.mean(); n = len(x)
    ac = np.array([1.0] + [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, 18)])
    P = None
    for k in range(6, 16):
        if ac[k] > 0.2 and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]:
            den = ac[k - 1] - 2 * ac[k] + ac[k + 1]; P = k + (0.5 * (ac[k - 1] - ac[k + 1]) / den if den else 0); break
    if P is None: P = 6 + int(np.argmax(ac[6:16]))
    # 위상 = (fw, −dfw/dt) 평면의 각도: 비정현파여도 fw 최대 = 0, 감소 중 = 0~0.5, 최소 = 0.5. 세트에 무관하게 같은 뜻.
    xs_ = np.convolve(x, [0.25, 0.5, 0.25], mode="same"); dx_ = np.gradient(xs_)
    fn = xs_ / np.percentile(np.abs(xs_), 95); dn = dx_ / max(np.percentile(np.abs(dx_), 95), 1e-9)
    phi_act = (np.arctan2(-dn, fn) / (2 * np.pi)) % 1.0
    phi = np.full(len(fw), np.nan); phi[idx] = phi_act
    order = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return P, phi, order


def main():
    labels = json.load(open(B / "labels.json"))
    front = [l for l in labels if l["view"] == "front"]
    P = {}
    for sid in ("cS1", "cS2", "nS1", "nS2"):
        ids = [l["id"] for l in front if l["set"] == sid]
        rows = [describe(load(i)) for i in ids]
        per, phi, order = phases(rows)
        for i, r, ph in zip(ids, rows, phi):
            r.update(id=i, set=sid, clothed=sid.startswith("c"), phi=None if np.isnan(ph) else float(ph)); P[i] = r
        print(f"{sid}: period {per:.2f} order_lag1 {order:.2f}  fw: " + " ".join(f"{r['fw']*100:+3.0f}" for r in rows[:18]))
    # 정면 착의 은행을 반전본으로 두 배로: 반전 = 위상 +0.5, fw/dy 부호 반전, 손 좌우 교환, 꼬리 쪽 반전
    for i in [k for k in P if P[k]["clothed"]]:
        r = P[i]; m = dict(r); m.update(id=i + "m", fw=-r["fw"], dy=-r["dy"], hand_r=r["hand_l"], hand_l=r["hand_r"],
                                        tail_side=-r["tail_side"], phi=None if r["phi"] is None else (r["phi"] + 0.5) % 1.0)
        P[i + "m"] = m
    ids = sorted(P); X = np.array([[P[i][k] for k in FEAT] for i in ids]); sd = np.nanstd(X, 0) + 1e-6; Xs = X / sd
    def dist(a, b): return float(np.nanmean(np.abs(Xs[ids.index(a)] - Xs[ids.index(b)])))
    def circ(a, b): d = abs(a - b); return min(d, 1 - d)
    clothed = [i for i in ids if P[i]["clothed"] and not P[i]["idle"]]; nude = [i for i in ids if not P[i]["clothed"] and not P[i]["idle"]]
    # 착의 정면 원화는 앞발 쪽으로 꼬리가 바뀌고(R when right foot fwd), 나체 캐논은 항상 오른쪽.
    # → 꼬리는 기증에서 떼고, 고정 꼬리 레이어를 맨 뒤에 깐다. 기증 후보는 원본+반전 전부(반전 벌점 0.05).
    print("active clothed", len(clothed), "nude", len(nude))
    used_c, used_n, pairs = set(), set(), []
    for i in range(NPH):
        tgt = i / NPH; best = None
        for win in (0.07, 0.12, 0.2, 0.5):
            for c in clothed:
                if c in used_c or P[c]["phi"] is None or circ(P[c]["phi"], tgt) > win: continue
                for n in nude:
                    if n in used_n or P[n]["phi"] is None or circ(P[n]["phi"], tgt) > win: continue
                    d = dist(c, n) + (0.05 if c.endswith("m") else 0)
                    if best is None or d < best[0]: best = (d, c, n)
            if best: break
        d, c, n = best; used_c.add(c); used_n.add(n)
        pairs.append(dict(phase=i, clothed=c, nude=n, dist=round(d, 3), fw_c=round(P[c]["fw"] * 600, 1), fw_n=round(P[n]["fw"] * 600, 1)))
        print(f"phi {i}/8  {c} <-> {n}  dist {d:.2f}  fw {P[c]['fw']*600:+.0f}/{P[n]['fw']*600:+.0f}px")
    json.dump(dict(pose=P, pairs=pairs), open(B / "front_pairs8.json", "w"), indent=1)

    # 기증 재선택 (정면 전용): 나체 짝은 진폭이 달라(±20 vs ±38px) 기준으로 부적합 → 착의 자체 진폭으로
    # 목표 fw_i = A·cos(2πi/8), A = 착의 |fw| 95% 값. 위상 창 ±0.10 안에서 |fw−목표| 최소 (반전 벌점, 중복 금지).
    Afw = float(np.percentile([abs(P[c]["fw"]) for c in clothed], 95))
    used = set()
    for p in pairs:
        i = p["phase"]; tgt_fw = Afw * np.cos(2 * np.pi * i / NPH); best = None
        for c in clothed:
            if c in used or P[c]["phi"] is None or circ(P[c]["phi"], i / NPH) > 0.10: continue
            sc = abs(P[c]["fw"] - tgt_fw) / Afw + 0.3 * circ(P[c]["phi"], i / NPH) * NPH + (0.05 if c.endswith("m") else 0)
            if best is None or sc < best[0]: best = (sc, c)
        if best is None:
            best = min(((abs(P[c]["fw"] - tgt_fw) / Afw + (0.05 if c.endswith("m") else 0), c) for c in clothed if c not in used))
        used.add(best[1]); p["clothed_nudepair"] = p["clothed"]; p["clothed"] = best[1]; p["fw_target"] = round(tgt_fw * 600, 1); p["fw_c"] = round(P[best[1]]["fw"] * 600, 1)
        print(f"phi {i}/8 donor {best[1]}  fw {p['fw_c']:+.0f} (target {p['fw_target']:+.0f})")
    # 조립 (정면): T = 착의 프레임 중 |fw| 최소 (두 발 나란히)
    Tid = min([c for c in ids if c.startswith("cS1")], key=lambda c: abs(P[c]["fw"]) + abs(P[c]["dy"]))   # idle 포함: 두 발 나란히 선 프레임
    Tfr = load(Tid); waistT = A.band_x(Tfr, 0.40, 0.44)
    # 고정 꼬리: 착의 원본 중 꼬리가 가장 큰(가장 잘 보이는) 프레임의 꼬리, 캐논 쪽(오른쪽)이면 그대로, 아니면 반전본
    tail_src = max([i for i in ids if i.startswith("cS1") and not i.endswith("m")], key=lambda i: int(load(i)["tail"].sum()))
    Tl = load(tail_src) if P[tail_src]["tail_side"] > 0 else load(tail_src + "m")
    tail_layer = A.layer(Tl["a"], Tl["tail"])
    frames, fC = [], []
    for p in pairs:
        D = load(p["clothed"]); dx = int(round(waistT - A.band_x(D, 0.40, 0.44)))
        # 기증 꼬리 제거 (검출 누락 대비): 털색 & t .48~.88 & 정중선에서 0.10H 밖 (다리·장화 바깥)
        rgb = D["a"][..., :3].astype(int); v = rgb.max(2); fur = (v > 125)          # 장갑(어두움) 만 남기고 밝은 것은 전부 꼬리로 간주
        sx = P[p["clothed"]]["sx"]; X = np.arange(640)[None, :]
        erase = D["tail"] | (D["m"] & fur & (D["t"] > 0.50) & (D["t"] < 0.86) & (np.abs(X - sx) > 0.10 * 600))
        from scipy.ndimage import binary_dilation
        erase = binary_dilation(erase, iterations=2) & D["m"] & (D["t"] > 0.46)
        Dm = dict(D); Dm["m"] = D["m"] & ~erase; Dm["tail"] = np.zeros_like(D["tail"])
        f = A.compose_A(Tfr, Dm, dx)
        out = tail_layer.copy(); A.over(out, f, (f[..., 3] > 0).astype(float) * (f[..., 3] / 255.0) ** 0)
        frames.append(out); fC.append(A.compose_C(Tfr, D, dx))
    print("fixed tail from", tail_src, "side", P[tail_src]["tail_side"])
    (OUT / "frontA").mkdir(exist_ok=True)
    for i, f in enumerate(frames): Image.fromarray(f).save(OUT / "frontA" / f"frontA_{i}.png")
    A.strip_and_gif(frames, "FRONT-A-fixed-upper", f"FRONT A: fixed upper {Tid} + fixed tail layer behind + lower body per phase from pose-matched donors (mirrors allowed)")
    A.strip_and_gif(fC, "FRONT-C-headlock", "FRONT C: donor frames + head lock")
    # 짝 시트
    cw, ch = 170, 250
    im = Image.new("RGB", (20 + NPH * cw, 60 + 2 * ch + 30), (250, 249, 246)); d = ImageDraw.Draw(im)
    d.text((14, 8), "FRONT 8-phase pairs (clothed 72 / nude 72): phi from forward-foot signal; matched on foot width diff, sole dy, leg asym, hands", fill=(20, 20, 20))
    for i, p in enumerate(pairs):
        for row, fid in enumerate((p["clothed"], p["nude"])):
            a = Image.fromarray(load(fid)["a"]); sc = (ch - 20) / a.height
            fr = a.resize((int(a.width * sc), int(a.height * sc))); x0 = 20 + i * cw; y0 = 40 + row * ch
            im.paste(fr, (x0, y0 + 14), fr); d.text((x0, y0), f"phi {i}/8 {fid} fw{P[fid]['fw']*600:+.0f}", fill=(20, 20, 20))
        d.text((20 + i * cw, 40 + 2 * ch + 4), f"dist {p['dist']:.2f}", fill=(20, 20, 20))
    im.save(OUT / "FRONT-PAIRS8.png")
    json.dump(dict(torso=Tid, pairs=pairs), open(OUT / "front_assembly.json", "w"), indent=1)


if __name__ == "__main__":
    main()
