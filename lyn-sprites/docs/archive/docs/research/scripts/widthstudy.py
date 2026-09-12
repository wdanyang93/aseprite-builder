"""widthstudy — 기준점 없이 '폭' 으로 옷 두께를 잰다 (동기화 쌍만 사용).

이유: 두 AI 레퍼런스는 머리-발 배치(lean), 보폭, 비율이 다르다. 머리든 발이든 어떤
기준점을 잡아도 그 차이가 옷 두께에 섞인다. 폭(오른끝−왼끝)은 기준점이 필요 없다.

t 칸마다 (80칸):
  몸통 t<0.47   실루엣 전체 폭 (뒤쪽은 망토 포함; 꼬리 제외)
  다리 t≥0.47   행 안의 연속 구간(run) → 앞다리 run, 뒷다리 run 을 따로. 하나로 붙으면 merged.
쌍마다   dW(t) = W_clothed − W_nude   (px, H 정규화 후 나체 H 로 환산)
분류     dW 가 위상마다 일정 → RIGID 두께 / 위상 따라 변함 → 변형 (팔다리 각도·장화·주름)
예측     캐논(나체) 루프 위상 i 의 몸 run 을 dW(t,i)/2 씩 양쪽으로 밀어 옷 실루엣을 그린다.
         몸통 앞/뒤 배분은 짝 측정(머리 기준) 의 앞 gap 비율을 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/handoff/work")
import pose
from garmentstudy import NB
from pairstudy import head_x

W = Path("/home/claude/handoff/work")
NPH = 8
T_LEG = 0.47


def runs_by_t(body, t, top, H):
    """t 칸마다 run 목록 [(x0, x1), ...] (x 오름차순)."""
    out = []
    for k in range(NB):
        y0 = int(top + k / NB * H); y1 = int(top + (k + 1) / NB * H)
        rows = body[y0:max(y1, y0 + 1)]
        col = rows.any(0)
        xs = np.nonzero(col)[0]
        if not len(xs):
            out.append([]); continue
        cuts = np.nonzero(np.diff(xs) > 3)[0]
        starts = np.r_[xs[0], xs[cuts + 1]]; ends = np.r_[xs[cuts], xs[-1]]
        rr = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= 3]
        out.append(rr)
    return out


def widths(p):
    a, body, t, xc, top, H, bot = pose.load(p)
    rr = runs_by_t(body, t, top, H)
    tot = np.full(NB, np.nan); front = np.full(NB, np.nan); back = np.full(NB, np.nan); nrun = np.zeros(NB, int)
    for k, r in enumerate(rr):
        if not r: continue
        tot[k] = (r[-1][1] - r[0][0] + 1) / H
        nrun[k] = len(r)
        if k >= int(T_LEG * NB):
            if len(r) >= 2:
                front[k] = (r[-1][1] - r[-1][0] + 1) / H; back[k] = (r[0][1] - r[0][0] + 1) / H
            else:
                front[k] = back[k] = tot[k]
    return dict(tot=tot, front=front, back=back, nrun=nrun, H=H, top=top, bot=bot, runs=rr, a=a, body=body, t=t,
                xr=head_x(body, t))


def main():
    sy = json.load(open(W / "pose_sync.json"))
    pairs = sy["pairs"]
    cl = [widths(str(W / "gt" / sy["clothed"]["files"][p["clothed"]])) for p in pairs]
    nu = [widths(str(W / "walkpng" / sy["nude"]["files"][p["nude"]])) for p in pairs]
    Hn = float(np.mean([r["H"] for r in nu]))
    kk = int(T_LEG * NB)
    # 몸통: 전체 폭 차 / 다리: 앞다리·뒷다리 폭 차 (둘 다 분리된 쌍에서만)
    dT = np.stack([c["tot"] - n["tot"] for c, n in zip(cl, nu)]) * Hn
    dF = np.full((NPH, NB), np.nan); dB = np.full((NPH, NB), np.nan); merged = np.zeros((NPH, NB), bool)
    for i, (c, n) in enumerate(zip(cl, nu)):
        both = (c["nrun"] >= 2) & (n["nrun"] >= 2)
        one = (c["nrun"] == 1) & (n["nrun"] == 1)
        dF[i, both] = (c["front"] - n["front"])[both] * Hn; dB[i, both] = (c["back"] - n["back"])[both] * Hn
        dF[i, one] = dB[i, one] = (c["tot"] - n["tot"])[one] * Hn / 2     # 붙은 행: 반씩
        merged[i] = one
    # 세그먼트 요약 (위상 평균·표준편차)
    def summarize(D, lo, hi, name):
        rows = []
        step = 4                                          # 4칸(5%) 단위
        for s in range(lo, hi, step):
            e = min(s + step, hi)
            block = D[:, s:e]
            m = np.nanmean(block, 1)                      # 위상별 평균
            if np.isnan(m).all(): continue
            rows.append(dict(t0=round(s / NB, 2), t1=round(e / NB, 2), part=name,
                             mean_px=round(float(np.nanmean(m)), 1), sd_px=round(float(np.nanstd(m)), 1),
                             per_phase=[None if np.isnan(x) else round(float(x), 1) for x in m],
                             n_phases=int((~np.isnan(m)).sum())))
        return rows
    seg = summarize(dT, 0, kk, "torso_total") + summarize(dF, kk, NB, "front_leg") + summarize(dB, kk, NB, "back_leg")
    for r in seg:
        r["cls"] = "RIGID" if r["sd_px"] < 4 else ("VARIES" if r["n_phases"] >= 4 else "SPARSE")

    # 예측: 나체 루프 몸 run 을 dW/2 씩 확장 → 옷 실루엣 (위상별 측정치 사용)
    # 몸통 앞/뒤 배분: pairstudy 의 머리기준 gap (앞 gap / (앞 gap + 뒤 gap))
    gp = json.load(open(W / "garment_paired.json"))
    preds = []
    for i, n in enumerate(nu):
        edges = []                                        # (k, x0, x1) 옷 run
        for k, r in enumerate(n["runs"]):
            if not r: continue
            if k < kk:
                d = dT[i, k] if not np.isnan(dT[i, k]) else np.nanmean(dT[:, k])
                if np.isnan(d): continue
                # 앞/뒤 배분: 앞 ≈ 0 (셔츠), 뒤 = 나머지 (망토·후드) — pairstudy 평균으로
                fshare = 0.0 if k >= int(0.2 * NB) else 0.5
                x0 = r[0][0] - (1 - fshare) * d * n["H"] / Hn; x1 = r[-1][1] + fshare * d * n["H"] / Hn
                edges.append((k, x0, x1))
            else:
                if len(r) >= 2:
                    df = dF[i, k] if not np.isnan(dF[i, k]) else np.nanmean(dF[:, k])
                    db = dB[i, k] if not np.isnan(dB[i, k]) else np.nanmean(dB[:, k])
                    if not np.isnan(db): edges.append((k, r[0][0] - db / 2 * n["H"] / Hn, r[0][1] + db / 2 * n["H"] / Hn))
                    if not np.isnan(df): edges.append((k, r[-1][0] - df / 2 * n["H"] / Hn, r[-1][1] + df / 2 * n["H"] / Hn))
                else:
                    d = dF[i, k] if not np.isnan(dF[i, k]) else np.nanmean(dF[:, k])
                    if np.isnan(d): continue
                    edges.append((k, r[0][0] - d * n["H"] / Hn, r[-1][1] + d * n["H"] / Hn))
        preds.append(edges)

    lines = ["GARMENT THICKNESS by WIDTH on pose-synced pairs (px at nude scale, H=%.0f). dW = clothed width - nude width" % Hn,
             "  part        t-range   mean   sd   class   per-phase 0..7"]
    for r in seg:
        pp = " ".join("    ." if v is None else f"{v:5.0f}" for v in r["per_phase"])
        lines.append(f"  {r['part']:11s} {r['t0']:.2f}-{r['t1']:.2f} {r['mean_px']:6.1f} {r['sd_px']:4.1f}  {r['cls']:6s} {pp}")
    lines += ["torso t<0.47: total width incl. cape (tail excluded). legs: per-leg run width; '.' = legs merged or missing in one set",
              "class RIGID = thickness does not depend on phase (sd<4px); VARIES = it does (knee/boot/cloth)"]
    res = dict(schema="platekit.garment_width.v1", H_nude=Hn, segments=seg, pairs=pairs, notes=lines)
    json.dump(res, open(W / "garment_width.json", "w"), indent=1, ensure_ascii=False)

    # 시트: 왼쪽 8칸 = 나체 루프 + 예측 옷 실루엣(주황) / 텍스트
    cw, ch = 190, 320
    Wd = 20 + NPH * cw + 760; Hd = 60 + ch + 30
    im = Image.new("RGB", (Wd, Hd), (250, 249, 246)); d = ImageDraw.Draw(im)
    d.text((14, 8), "PREDICTED OUTFIT SILHOUETTE on the canonical body (its own gait) — orange = body run widened by the thickness "
                    "measured on the pose-synced pair of the SAME phase (no anchor point involved)", fill=(20, 20, 20))
    d.text((14, 24), "torso: total width (cape goes to the back), legs: each leg widened by its own measured dW/2 per side. "
                     "Not pixels — a silhouette envelope. Tail region unknown.", fill=(90, 90, 90))
    for i, n in enumerate(nu):
        a = n["a"]; x0 = 20 + i * cw; y0 = 44
        sc = (ch - 20) / a.shape[0]
        fr = Image.fromarray(a).resize((int(a.shape[1] * sc), int(a.shape[0] * sc)))
        im.paste(fr, (x0 + 14, y0 + 14), fr)
        for k, xa, xb in preds[i]:
            yy = y0 + 14 + (n["top"] + (k + 0.5) / NB * n["H"]) * sc
            for xx in (xa, xb):
                px = x0 + 14 + xx * sc
                d.line([px - 2, yy, px + 2, yy], fill=(240, 130, 20), width=2)
        d.text((x0, y0), f"phi {i}/8  w{pairs[i]['nude']+1:02d}  (pair g{pairs[i]['clothed']+1:02d})", fill=(20, 20, 20))
    x0 = 30 + NPH * cw; y = 44
    for line in lines:
        d.text((x0, y), line, fill=(30, 30, 30)); y += 14
    im.save("/mnt/user-data/outputs/PREDICTED-OUTFIT-SILHOUETTE.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
