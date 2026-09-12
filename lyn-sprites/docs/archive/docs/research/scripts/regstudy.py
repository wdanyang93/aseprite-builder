"""regstudy — 옷 실루엣을 '자세 변수' 로 설명한다 (걸음 진폭 차이를 떼어낸다).

문제: 착의 레퍼런스와 나체 레퍼런스는 걸음이 다르다 (보폭 1.13배, 발 들림 0.62배 …).
      같은 위상끼리 빼면 '옷 두께' 와 '걸음 차이' 가 섞인다.
해법: 두 세트가 공유하는 자세 변수 = 앞발 x, 뒷발 x (머리 중심 기준, H 정규화).
      세트마다 모든 프레임(35/36장)으로 t 칸별 회귀
          edge(t) = a(t) + bf(t)·xf + bb(t)·xb
      착의 세트 → 옷 실루엣 규칙, 나체 세트 → 몸 실루엣 규칙.
      같은 (xf, xb) 를 넣으면 같은 자세에서의 옷/몸이 나오므로 차이 = 순수 옷.
분류 (착의 회귀로)
      RIGID   bf,bb ≈ 0 이고 잔차 작음 — 발이 어디 있든 옷끝이 같다 (상체)
      FOLLOW  R² ≥ 0.5 — 발 위치로 설명된다 (바지·장화)
      LOOSE   잔차가 크고 발로 설명 안 됨 (망토·무릎)
적용   캐논 걸음 = 나체 8루프의 (xf, xb). 거기에 착의 규칙을 넣어 '이 몸이 이 옷을 입고
      이 걸음으로 걸으면 실루엣이 이렇게 된다' 를 8위상으로 그린다. 외삽 여부를 같이 적는다.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/handoff/work")
import pose
from garmentstudy import NB
from pairstudy import head_x, profile_head

W = Path("/home/claude/handoff/work")
NPH = 8


def frame_row(p):
    a, body, t, xc, top, H, bot = pose.load(p)
    xr = head_x(body, t)
    ft = pose.feet(body, t, xc, H, bot)
    pr = profile_head(p)
    xf = (xc + ft["xf"] * H - xr) / H; xb = (xc + ft["xb"] * H - xr) / H
    xm = xr + (xf + xb) / 2 * H                       # 발 중점 (px)
    s = float(xf - xb)                                # 보폭
    # 다리 구간용: 발 중점 기준 실루엣
    front_m = pr["front"] + (xr - xm) / H; back_m = pr["back"] + (xr - xm) / H
    return dict(path=p, xf=float(xf), xb=float(xb), s=s, lean=float((xr - xm) / H),
                front=pr["front"], back=pr["back"], front_m=front_m, back_m=back_m,
                H=H, top=top, bot=bot, xr=xr, xm=float(xm), a=a, body=body, t=t)


T_LEG = 0.47


def fit(rows):
    """t < T_LEG: 머리 기준, edge = a + b·s ;  t ≥ T_LEG: 발 중점 기준, edge = a + b·s."""
    X = np.array([[1, r["s"]] for r in rows])
    out = {}
    for side in ("front", "back"):
        Yh = np.stack([r[side] for r in rows]); Ym = np.stack([r[side + "_m"] for r in rows])
        kk = int(T_LEG * NB)
        Y = np.concatenate([Yh[:, :kk], Ym[:, kk:]], 1)      # (n, NB)
        coef = np.full((NB, 2), np.nan); r2 = np.full(NB, np.nan); rsd = np.full(NB, np.nan)
        for k in range(NB):
            y = Y[:, k]; ok = ~np.isnan(y)
            if ok.sum() < 8:
                continue
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            res = y[ok] - X[ok] @ beta
            coef[k] = beta; rsd[k] = res.std()
            tot = ((y[ok] - y[ok].mean()) ** 2).sum()
            r2[k] = 1 - (res ** 2).sum() / max(tot, 1e-9)
        out[side] = dict(coef=coef, r2=r2, rsd=rsd)
    return out


def classify(fitc, Hpx):
    cls = {}
    for side in ("front", "back"):
        c = fitc[side]; lab = []
        for k in range(NB):
            if np.isnan(c["r2"][k]):
                lab.append("NA"); continue
            slope = abs(c["coef"][k, 1])
            if slope < 0.12 and c["rsd"][k] * Hpx < 5:
                lab.append("RIGID")
            elif c["r2"][k] >= 0.5:
                lab.append("FOLLOW")
            else:
                lab.append("LOOSE")
        cls[side] = lab
    return cls


def predict(fitc, s):
    X = np.array([1, s])
    return {side: fitc[side]["coef"] @ X for side in ("front", "back")}


def to_head_frame(edge, r):
    """예측 edge(혼합 기준) 를 머리 기준 좌표로 통일."""
    kk = int(T_LEG * NB); e = edge.copy()
    e[kk:] = e[kk:] - (r["xr"] - r["xm"]) / r["H"]
    return e


def main():
    sy = json.load(open(W / "pose_sync.json"))
    cl = [frame_row(str(W / "gt" / f)) for f in sy["clothed"]["files"]]
    nu = [frame_row(str(W / "walkpng" / f)) for f in sy["nude"]["files"]]
    Hc = float(np.mean([r["H"] for r in cl])); Hn = float(np.mean([r["H"] for r in nu]))
    fc = fit(cl); fn = fit(nu)
    cls = classify(fc, Hc)

    # 자세 변수 범위 (외삽 확인)
    rng = {k: dict(clothed=(round(min(r[k] for r in cl), 3), round(max(r[k] for r in cl), 3)),
                   nude=(round(min(r[k] for r in nu), 3), round(max(r[k] for r in nu), 3)))
           for k in ("s", "lean", "xf", "xb")}

    # 몸통용 gap: 자세 동기화 쌍(머리 기준) 의 위상 평균 — sd 가 작아 RIGID
    pairs = sy["pairs"]
    Cf = np.stack([cl[p["clothed"]]["front"] for p in pairs]); Nf_ = np.stack([nu[p["nude"]]["front"] for p in pairs])
    Cb = np.stack([cl[p["clothed"]]["back"] for p in pairs]); Nb_ = np.stack([nu[p["nude"]]["back"] for p in pairs])
    torso_gap_f = np.nanmean(Cf - Nf_, 0); torso_gap_b = np.nanmean(Nb_ - Cb, 0)
    torso_gap_f = np.nan_to_num(torso_gap_f); torso_gap_b = np.nan_to_num(torso_gap_b)
    # 캐논 걸음 = 나체 8루프 자세
    loop = sy["nude"]["loop8"]
    preds = []
    for i, j in enumerate(loop):
        r = nu[j]
        pc = predict(fc, r["s"]); pn = predict(fn, r["s"])
        gf = to_head_frame(pc["front"], r); gb = to_head_frame(pc["back"], r)
        bf_ = to_head_frame(pn["front"], r)
        # 몸통(t<T_LEG) 은 회귀 대신 '몸 실루엣 + 짝 측정 평균 gap' (RIGID) — 두 캐릭터 비율 차이를 피한다
        kk = int(T_LEG * NB)
        gf[:kk] = r["front"][:kk] + torso_gap_f[:kk]; gb[:kk] = r["back"][:kk] - torso_gap_b[:kk]
        extrap = (r["s"] > rng["s"]["clothed"][1] + 0.02) or (r["s"] < rng["s"]["clothed"][0] - 0.02)
        preds.append(dict(phase=i, nude_frame=j, xf=r["xf"], xb=r["xb"], s=r["s"], extrapolated=bool(extrap),
                          garment_front=gf, garment_back=gb,
                          body_front_fit=bf_, body_front_actual=r["front"], body_back_actual=r["back"]))

    # 순수 옷 두께 (자세를 고정한 뒤): gap(t, i) = 옷(t|pose_i) − 몸(t, i)
    gap_f = np.stack([p["garment_front"] - p["body_front_actual"] for p in preds]) * Hn
    gap_b = np.stack([p["body_back_actual"] - p["garment_back"] for p in preds]) * Hn
    # 몸 회귀의 검증: 나체 규칙이 나체 자신을 얼마나 맞히나 (px)
    body_fit_err = float(np.nanmean(np.abs(np.stack([p["body_front_fit"] - p["body_front_actual"] for p in preds])))) * Hn

    def seg(side, gap):
        lab = cls[side]; out = []; s = 0
        for k in range(1, NB + 1):
            if k == NB or lab[k] != lab[s]:
                if (k - s) / NB >= 0.03:
                    g = np.nanmean(gap[:, s:k], 1)
                    c = fc[side]["coef"][s:k]
                    out.append(dict(t0=round(s / NB, 2), t1=round(k / NB, 2), cls=lab[s],
                                    bf=round(float(np.nanmean(c[:, 1])), 2), bb=round(float(np.nanmean(fn[side]["coef"][s:k, 1])), 2),
                                    r2=round(float(np.nanmean(fc[side]["r2"][s:k])), 2),
                                    resid_px=round(float(np.nanmean(fc[side]["rsd"][s:k])) * Hc, 1),
                                    gap_px=[round(float(x), 1) for x in g]))
                s = k
        return out
    seg_f = seg("front", gap_f); seg_b = seg("back", gap_b)

    lines = ["GARMENT RULE by STRIDE s (all 35 clothed frames; legs referenced to foot midpoint, torso to head), on the canonical (nude) loop:",
             "  FRONT  t-range  class  b_garment b_body  R2  resid | gap(px) per phase 0..7 = garment beyond body at the SAME stride"]
    for s in seg_f:
        lines.append(f"  t {s['t0']:.2f}-{s['t1']:.2f} {s['cls']:6s} {s['bf']:+.2f}     {s['bb']:+.2f}  {s['r2']:.2f} {s['resid_px']:4.1f} | " + " ".join(f"{g:5.0f}" for g in s["gap_px"]))
    lines.append("  BACK")
    for s in seg_b:
        lines.append(f"  t {s['t0']:.2f}-{s['t1']:.2f} {s['cls']:6s} {s['bf']:+.2f}     {s['bb']:+.2f}  {s['r2']:.2f} {s['resid_px']:4.1f} | " + " ".join(f"{g:5.0f}" for g in s["gap_px"]))
    lines += [f"stride s range: clothed {rng['s']['clothed']}  nude {rng['s']['nude']} | lean (head over foot-mid): clothed {rng['lean']['clothed']} nude {rng['lean']['nude']}",
              "  torso (t<0.47) uses the pose-synced pair gap (RIGID), legs use the stride regression",
              f"extrapolated phases: {[p['phase'] for p in preds if p['extrapolated']]}",
              f"body-rule self-check on nude loop: {body_fit_err:.1f}px mean |err| (how well feet predict the body edge)"]

    res = dict(schema="platekit.garment_rule_by_pose.v1", H_clothed=Hc, H_nude=Hn, ranges=rng,
               front=seg_f, back=seg_b, classes=cls, body_fit_err_px=body_fit_err,
               loop=[dict(phase=p["phase"], nude_frame=p["nude_frame"], xf=p["xf"], xb=p["xb"], extrapolated=p["extrapolated"]) for p in preds],
               notes=lines)
    json.dump(res, open(W / "garment_rule_by_pose.json", "w"), indent=1, ensure_ascii=False)
    np.savez(W / "garment_rule_by_pose.npz", coef_front=fc["front"]["coef"], coef_back=fc["back"]["coef"],
             body_coef_front=fn["front"]["coef"], body_coef_back=fn["back"]["coef"],
             pred_front=np.stack([p["garment_front"] for p in preds]), pred_back=np.stack([p["garment_back"] for p in preds]))

    # ── 시트: 캐논(나체) 8루프 위에 예측 옷 실루엣 ──────────────────────────
    cw, ch = 190, 320
    Wd = 20 + NPH * cw + 700; Hd = 60 + ch + 30
    im = Image.new("RGB", (Wd, Hd), (250, 249, 246)); d = ImageDraw.Draw(im)
    d.text((14, 8), "PREDICTED OUTFIT SILHOUETTE on the canonical body walking its own gait — orange = garment edge predicted "
                    "from foot positions (rule learned on all 35 clothed frames), cyan = body edge", fill=(20, 20, 20))
    d.text((14, 24), "back side below t=0.47 is UNKNOWN (tail/cape) and not drawn. '*' = pose outside clothed data range (extrapolated)", fill=(90, 90, 90))
    colmap = {"RIGID": (40, 160, 70), "FOLLOW": (40, 90, 220), "LOOSE": (220, 50, 40), "NA": (170, 170, 170)}
    for i, p in enumerate(preds):
        r = nu[p["nude_frame"]]; a = r["a"]
        x0 = 20 + i * cw; y0 = 44
        sc = (ch - 20) / a.shape[0]
        fr = Image.fromarray(a).resize((int(a.shape[1] * sc), int(a.shape[0] * sc)))
        im.paste(fr, (x0 + 14, y0 + 14), fr)
        for k in range(NB):
            yy = y0 + 14 + (r["top"] + (k + 0.5) / NB * r["H"]) * sc
            for arr, col, w in ((p["body_front_actual"], (0, 170, 190), 1), (p["body_back_actual"], (0, 170, 190), 1),
                                (p["garment_front"], (240, 130, 20), 2), (p["garment_back"], (240, 130, 20), 2)):
                if np.isnan(arr[k]): continue
                xx = x0 + 14 + (r["xr"] + arr[k] * r["H"]) * sc
                d.line([xx - 2, yy, xx + 2, yy], fill=col, width=w)
            d.rectangle([x0 + 14 + a.shape[1] * sc + 2, yy - 1, x0 + 14 + a.shape[1] * sc + 8, yy + 1], fill=colmap[cls["front"][k]])
        d.text((x0, y0), f"phi {i}/8 w{p['nude_frame']+1:02d}{' *' if p['extrapolated'] else ''}", fill=(20, 20, 20))
    x0 = 30 + NPH * cw; y = 44
    for line in lines:
        d.text((x0, y), line, fill=(30, 30, 30)); y += 15
    im.save("/mnt/user-data/outputs/PREDICTED-OUTFIT-SILHOUETTE.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
