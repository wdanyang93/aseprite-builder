"""sync — 두 걸음 세트(착의 35, 나체 36)를 프레임 번호가 아니라 '자세'로 동기화한다.

지적: 8프레임을 1~8번끼리 비교하면 안 된다. 36장에서 먼저 시작점을 맞추고, 각 세트에서
한 루프가 되는 8장을 자세 기준으로 고른 뒤, 손·발·팔·다리가 가장 비슷한 것끼리 짝지어
비교해야 한다. 순서는 검증하고 쓴다.

보행 위상 phi (한 바퀴 = 두 걸음) 를 물리적으로 정의한다
  theta  걸음 안 진행도. 디딤발→스윙발 벡터 (dx, lift):
         스윙발이 뒤(dx<0, 바닥) = 0, 발 교차(dx=0, 최고 들림) = 0.5, 앞에 착지 = 1
  half   어느 다리가 앞인가. 두 세트 모두 AI 가 먼 다리를 어둡게 칠한다:
         앞다리가 더 밝다 = 가까운 다리가 앞 (N),  아니면 먼 다리가 앞 (F)
         (나체 피부 V 차 −2 vs +33, 착의 레깅스 V 차 −3 vs +5 — 둘 다 보폭 봉우리마다 번갈음,
          실루엣 IoU 군집(g02≈g14≈g25 / g08≈g20≈g30) 과 일치)
  phi = theta/2            N 봉우리 뒤 (먼 다리가 스윙)
      = 0.5 + theta/2      F 봉우리 뒤 (가까운 다리가 스윙)
  phi=0 은 '가까운 다리가 앞, 보폭 최대' — 두 세트에 같은 뜻이므로 짝은 phi 로 바로 맞는다.
  8위상 대표 = phi 가 i/8 에 가장 가까운 관측 프레임 (새 화소 없음).
  검증: 자세 특징(발·손·기울기) 거리의 순환 이동 8개 중 s=0 이 최소여야 한다.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/handoff/work")
import debg
import pose

W = Path("/home/claude/handoff/work")
NPH = 8


def load_set(files):
    out = []
    for p in files:
        a, body, t, xc, top, H, bot = pose.load(p)
        out.append(dict(m=body, xc=xc, bot=bot, H=H, top=top, a=a, t=t, path=p))
    return out


# ── 앞다리 밝기 (가까운 다리 판별) ──────────────────────────────────────
def front_minus_back_V(m, dark_only):
    a, body, t, xc, H, bot = m["a"], m["m"], m["t"], m["xc"], m["H"], m["bot"]
    ft = pose.feet(body, t, xc, H, bot)
    if ft is None or ft["together"]:
        return np.nan
    band = body & (t[:, None] > 0.62) & (t[:, None] < 0.86)
    rgb = a[..., :3].astype(int); v = rgb.max(2); s = rgb.max(2) - rgb.min(2)
    band &= (v < 140) if dark_only else ((s >= 45) & (v > 150))
    ys, xs = np.nonzero(band)
    if len(xs) < 50:
        return np.nan
    mid = xc + (ft["xf"] + ft["xb"]) / 2 * H
    fr = v[ys, xs][xs > mid]; bk = v[ys, xs][xs <= mid]
    if len(fr) < 30 or len(bk) < 30:
        return np.nan
    return float(np.median(fr) - np.median(bk))


# ── 걸음 안 진행도 theta ────────────────────────────────────────────────
def phases(ms, dark_only):
    """theta: 보폭 크기 + 닫힘/열림 방향. 방향은 발 들림이 말해 주면 그것을, 아니면 보폭의
    시간 미분(순서)을 쓴다 — AI 는 발을 미끄러뜨려 그리므로 들림만으로는 부족하다.
    봉우리(보폭 ≥ 0.75 smax) 는 theta=0, 라벨은 앞다리 밝기로."""
    n = len(ms)
    fts = [pose.feet(m["m"], m["t"], m["xc"], m["H"], m["bot"]) for m in ms]
    spread = np.array([f["spread"] if f else np.nan for f in fts])
    together = np.array([f["together"] if f else True for f in fts])
    lf = np.array([f["lift_f"] if f else 0 for f in fts]); lb = np.array([f["lift_b"] if f else 0 for f in fts])
    smax = np.nanpercentile(spread, 95)
    sn = np.clip(spread / smax, 0, 1)
    dsp = np.gradient(np.nan_to_num(spread))
    theta = np.zeros(n); dirn = np.zeros(n, int)
    peak = (spread >= 0.75 * smax) & ~together
    for k in range(n):
        if together[k]:
            theta[k] = 0.5; continue
        if peak[k]:
            theta[k] = 0.0; continue
        if lb[k] > lf[k] + 0.015: d = -1          # 뒷발이 확실히 떠 있다(1.5%H) → 닫히는 중
        elif lf[k] > lb[k] + 0.015: d = +1        # 앞발이 확실히 떠 있다 → 열리는 중(착지 직전)
        else: d = int(np.sign(dsp[k])) or -1
        dirn[k] = d
        theta[k] = 0.5 - 0.5 * sn[k] if d < 0 else 0.5 + 0.5 * sn[k]
    shade = np.array([front_minus_back_V(m, dark_only) for m in ms])
    thr = np.nanmedian(shade[peak & ~np.isnan(shade)])
    lab = np.array(["?"] * n, dtype=object)
    lab[peak & ~np.isnan(shade)] = np.where(shade[peak & ~np.isnan(shade)] > thr, "N", "F")
    cur = None
    for k in range(n):
        if lab[k] != "?": cur = lab[k]
        elif cur is not None: lab[k] = cur
    first = next(i for i in range(n) if lab[i] != "?")
    for k in range(first):
        lab[k] = "F" if lab[first] == "N" else "N"
    phi = np.where(lab == "N", theta / 2, 0.5 + theta / 2) % 1.0
    order_ok = float(np.corrcoef(np.nan_to_num(spread[:-1]), np.nan_to_num(spread[1:]))[0, 1])
    return dict(theta=theta, shade=shade, label=lab, phi=phi, spread=spread, lift=np.maximum(lf, lb),
                smax=float(smax), shade_thr=float(thr), order_lag1=order_ok, peak=peak, dirn=dirn)


def pick8(phi):
    """8위상에 서로 다른 프레임을 배정 (헝가리안). 같은 프레임을 두 위상에 쓰지 않는다 —
    그 위상의 관측이 없으면 없다고 오차로 드러나야 한다."""
    from scipy.optimize import linear_sum_assignment
    P = np.array([i / NPH for i in range(NPH)])
    D = np.minimum(np.abs(phi[None, :] - P[:, None]), 1 - np.abs(phi[None, :] - P[:, None]))
    D = np.nan_to_num(D, nan=9.0)
    r, c = linear_sum_assignment(D)
    sel = [0] * NPH; err = [0.0] * NPH
    for i, j in zip(r, c):
        sel[i] = int(j); err[i] = float(D[i, j])
    return sel, err


# ── 동기화 검증용 자세 특징 ───────────────────────────────────────────────
FEAT = ["spread", "xf", "xb", "lift_f", "lift_b", "hand_f", "hand_b", "lean"]


def features(m):
    ft = pose.feet(m["m"], m["t"], m["xc"], m["H"], m["bot"])
    body, t, xc, H = m["m"], m["t"], m["xc"], m["H"]
    def ext(lo, hi, side):
        band = body & (t[:, None] > lo) & (t[:, None] < hi)
        xs = np.nonzero(band)[1]
        if not len(xs): return np.nan
        return ((np.percentile(xs, 99) if side > 0 else np.percentile(xs, 1)) - xc) / H
    hand_f = ext(0.46, 0.62, +1) - ext(0.40, 0.46, +1)
    hand_b = ext(0.40, 0.46, -1) - ext(0.46, 0.62, -1)
    head = body & (t[:, None] < 0.2); hx = np.nonzero(head)[1].mean()
    return dict(spread=ft["spread"], xf=ft["xf"], xb=ft["xb"], lift_f=ft["lift_f"], lift_b=ft["lift_b"],
                hand_f=float(hand_f), hand_b=float(hand_b), lean=float((hx - xc) / H))


def shift_costs(Fc, Fn):
    A = np.array([[f[k] for k in FEAT] for f in Fc]); B = np.array([[f[k] for k in FEAT] for f in Fn])
    sd = np.nanstd(np.vstack([A, B]), 0) + 1e-6
    A = A / sd; B = B / sd
    costs = []; per = None
    for s in range(NPH):
        d = np.array([np.nanmean(np.abs(A[i] - B[(i + s) % NPH])) for i in range(NPH)])
        costs.append(float(d.sum()))
        if s == 0: per = d
    return costs, per


# ── 시트 ───────────────────────────────────────────────────────────────
def sheet(cl, nu, selc, seln, per, res, path):
    cw, ch = 170, 250
    Wd = 20 + NPH * cw + 470; Hd = 60 + 2 * ch + 40
    im = Image.new("RGB", (Wd, Hd), (250, 249, 246)); d = ImageDraw.Draw(im)
    d.text((14, 8), "POSE-SYNC  row1 clothed loop, row2 nude loop — same phi means same pose: "
                    "phi 0 = near leg forward at widest stride, 0.5 = far leg forward", fill=(20, 20, 20))
    d.text((14, 24), "red ring = front foot, blue ring = back foot (ring height = lift), grey tick = torso center; "
                     "N/F = which leg is forward (near/far, from leg shading)", fill=(90, 90, 90))

    def draw(m, x0, y0, label, lab, theta):
        a = m["a"]
        sc = (ch - 24) / a.shape[0]
        fr = Image.fromarray(a).resize((int(a.shape[1] * sc), int(a.shape[0] * sc)))
        im.paste(fr, (x0, y0 + 14), fr)
        f = features(m); H, xc, bot, top = m["H"], m["xc"], m["bot"], m["top"]
        def P(xn, yn): return (x0 + (xc + xn * H) * sc, y0 + 14 + yn * sc)
        for xn, lift, col in ((f["xf"], f["lift_f"], (220, 50, 40)), (f["xb"], f["lift_b"], (40, 90, 220))):
            px, py = P(xn, bot - lift * H)
            d.ellipse([px - 5, py - 5, px + 5, py + 5], outline=col, width=3)
        px, py = P(0, top + 0.43 * H); d.line([px, py - 8, px, py + 8], fill=(120, 120, 120), width=2)
        d.text((x0, y0), f"{label}  {lab} th={theta:.2f}", fill=(20, 20, 20))

    for i in range(NPH):
        x0 = 20 + i * cw
        jc, jn = selc[i], seln[i]
        draw(cl[jc], x0, 44, f"phi {i}/8 g{jc+1:02d}", res["clothed"]["label"][jc], res["clothed"]["theta"][jc])
        draw(nu[jn], x0, 44 + ch, f"w{jn+1:02d}", res["nude"]["label"][jn], res["nude"]["theta"][jn])
        d.text((x0, 44 + 2 * ch + 2), f"pose dist {per[i]:.2f}", fill=(20, 20, 20))
    x0 = 30 + NPH * cw; y = 44
    for line in res["notes"]:
        d.text((x0, y), line, fill=(30, 30, 30)); y += 15
    im.save(path)


if __name__ == "__main__":
    cl_files = sorted(glob.glob(str(W / "gt/g*.png")))
    nu_files = sorted(glob.glob(str(W / "walkpng/w*.png")))
    cl = load_set(cl_files); nu = load_set(nu_files)
    Pc = phases(cl, dark_only=True); Pn = phases(nu, dark_only=False)
    selc, errc = pick8(Pc["phi"]); seln, errn = pick8(Pn["phi"])
    Fc = [features(cl[j]) for j in selc]; Fn = [features(nu[j]) for j in seln]
    costs, per = shift_costs(Fc, Fn)

    def amp(F, k): return float(np.nanmax([f[k] for f in F]) - np.nanmin([f[k] for f in F]))
    gait = {k: dict(clothed=round(amp(Fc, k), 3), nude=round(amp(Fn, k), 3)) for k in ("spread", "lift_b", "hand_f", "hand_b", "lean")}
    lab_c = "".join(Pc["label"]); lab_n = "".join(Pn["label"])
    notes = [
        "WITHIN-SET:",
        f"  clothed labels {lab_c}",
        f"  nude    labels {lab_n}",
        f"  order check (spread lag-1 corr): clothed {Pc['order_lag1']:.2f}, nude {Pn['order_lag1']:.2f}",
        f"  clothed loop g{[j+1 for j in selc]}  phase err {[round(e,2) for e in errc]}",
        f"  nude    loop w{[j+1 for j in seln]}  phase err {[round(e,2) for e in errn]}",
        "  (phase err = how far the nearest distinct observation is from i/8; 0.12 = one eighth missing)",
        "CROSS-SET: shift cost s=0..7 = " + " ".join(f"{c:.1f}" for c in costs),
        f"  s=0 is {'the minimum -> physical anchor confirmed' if int(np.argmin(costs)) == 0 else 'NOT the minimum: anchor disagrees, see s=' + str(int(np.argmin(costs)))}",
        f"  mean pair pose distance {per.mean():.2f} (std units)",
        "GAIT DIFFERENCE THAT REMAINS AFTER SYNC (loop amplitude / H):",
    ] + [f"  {k:7s} clothed {v['clothed']:.3f}  nude {v['nude']:.3f}  ratio {v['clothed']/max(v['nude'],1e-6):.2f}" for k, v in gait.items()] + [
        f"  step timing: clothed ~5.6 frames/step, nude ~8.3 -> removed by phi-resampling",
        "-> amplitude ratios are the AI-to-AI gait mismatch; selection cannot fix them,",
        "   only a canonical gait + deformation can.",
    ]
    res = dict(schema="platekit.pose_sync.v2",
               clothed=dict(files=[Path(p).name for p in cl_files], phi=[float(x) for x in Pc["phi"]],
                            theta=[float(x) for x in Pc["theta"]], label=list(Pc["label"]),
                            shade=[None if np.isnan(x) else float(x) for x in Pc["shade"]],
                            spread=[float(x) for x in Pc["spread"]], loop8=selc, order_lag1=Pc["order_lag1"]),
               nude=dict(files=[Path(p).name for p in nu_files], phi=[float(x) for x in Pn["phi"]],
                         theta=[float(x) for x in Pn["theta"]], label=list(Pn["label"]),
                         shade=[None if np.isnan(x) else float(x) for x in Pn["shade"]],
                         spread=[float(x) for x in Pn["spread"]], loop8=seln, order_lag1=Pn["order_lag1"]),
               shift_costs=costs, pair_distance=[float(x) for x in per],
               phase_err=dict(clothed=errc, nude=errn),
               pairs=[dict(phase=i, clothed=int(selc[i]), nude=int(seln[i]), clothed_feat=Fc[i], nude_feat=Fn[i]) for i in range(NPH)],
               gait_difference=gait, notes=notes)
    json.dump(res, open(W / "pose_sync.json", "w"), indent=1, ensure_ascii=False, default=float)
    sheet(cl, nu, selc, seln, per, res, "/mnt/user-data/outputs/POSE-SYNC-8x2.png")
    print("\n".join(notes))
    print("clothed phi:", " ".join(f"{p:.2f}" for p in Pc["phi"]))
    print("nude    phi:", " ".join(f"{p:.2f}" for p in Pn["phi"]))
