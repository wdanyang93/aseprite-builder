"""garmentstudy — 옷을 입은 채 걷는 프레임에서 '옷 형태가 어떻게 변하는가'를 실측한다.

지시: 옷을 자르지도, 첫 장을 옮기지도 말 것. 걷는 동안 옷의 어느 부분이 그대로이고
어느 부분이 팔·다리 때문에 비율·모양이 바뀌는지를 찾아서, 그 규칙으로
"나체 원화가 이 옷을 입으면 이렇게 될 것이다"를 예측한다.

자료  gt/      옷 입은 측면 걸음 35장 (동쪽)
      walkpng/ 나체 측면 걸음 36장 (동쪽, 다른 보행 레퍼런스)

측정  몸축 t (0 정수리, 1 발끝) 80칸. 칸마다 실루엣의 뒤쪽 끝·앞쪽 끝을
      몸통 중심선 x_c 기준으로 잰다 (신장 H 로 정규화). 꼬리는 제거.
      걸음 위상 psi 는 발 벌어짐(stride) 의 기본파 — 측면은 stride 가 정보량이 크다.
      한 걸음(반주기) 을 8위상으로 나눈다.

분류  t 칸마다 위상에 따른 옷 실루엣 변화를 세 가지로 나눈다
      RIGID   옷 실루엣이 위상과 무관하게 일정          (몸통·조끼·배낭)
      FOLLOW  옷 실루엣 = a + b · 나체 실루엣(같은 위상) (바지·소매·장화)
      LOOSE   위상 따라 변하지만 나체로 설명되지 않음   (망토·꼬리 근처)

예측  FOLLOW 칸은 a(t)+b(t)·나체(t,psi), RIGID 칸은 평균, LOOSE 칸은 '모름' 표시.
      leave-one-phase-out 으로 실제 옷 실루엣과 비교. 비교 기준선:
      (a) 캐논 1장 그대로 복사 (지적받은 '억지 캐논' 방식)  (b) 예측 규칙.
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

W = Path("/home/claude/handoff/work")
NB = 80          # t 칸 수
NPH = 8          # 위상 수


# ── 프레임 → 실루엣 프로파일 ──────────────────────────────────────────────
def load(p):
    a = np.array(Image.open(p).convert("RGBA"))
    a, _ = debg.clean(a)
    return a, a[..., 3] > 128


def tail_mask(a, m):
    """측면 꼬리: 털색 씨앗(밝고 r-b 작음: 살색·바지·장화 제외) & 몸통 중심선보다 뒤 & 허리 아래.
    가장 큰 덩어리를 취한 뒤, 같은 행에서 털 화소 사이에 낀 화소(그림자·망토가 겹친 부분)를
    행 단위로 채운다 — 실루엣 끝을 재는 목적이라 꼬리 안쪽이 무엇이든 상관없다."""
    from scipy.ndimage import binary_dilation, label, binary_fill_holes
    rgb = a[..., :3].astype(int); r, b = rgb[..., 0], rgb[..., 2]
    v = rgb.max(2)
    ys, xs = np.nonzero(m)
    top, bot = ys.min(), ys.max(); H = bot - top + 1
    t = (np.arange(a.shape[0]) - top) / H
    X = np.arange(a.shape[1])[None, :]
    torso = m & (t[:, None] > 0.28) & (t[:, None] < 0.45)
    xc = np.median(np.nonzero(torso)[1])
    fur = (v > 140) & ((r - b) < 34)
    tl = m & fur & (X < xc - 0.03 * H) & (t[:, None] > 0.36) & (t[:, None] < 0.95)
    tl = binary_dilation(tl, iterations=3) & m & (X < xc - 0.02 * H) & (t[:, None] > 0.36)
    lab, n = label(tl)
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        tl = lab == sizes.argmax()
    tl = binary_fill_holes(tl)
    lim = int(xc - 0.02 * H)
    for y in np.nonzero(tl.any(1))[0]:          # 행 단위 채움
        xx = np.nonzero(tl[y])[0]
        tl[y, xx.min():min(xx.max(), lim) + 1] = m[y, xx.min():min(xx.max(), lim) + 1]
    return tl, xc, top, H


def profile(p):
    a, m = load(p)
    tl, xc, top, H = tail_mask(a, m)
    body = m & ~tl
    ys, xs = np.nonzero(body)
    top, bot = ys.min(), ys.max(); H = bot - top + 1
    t = (ys - top) / H
    back = np.full(NB, np.nan); front = np.full(NB, np.nan)
    b = np.minimum((t * NB).astype(int), NB - 1)
    for k in range(NB):
        sel = b == k
        if sel.sum() < 3:
            continue
        back[k] = (xs[sel].min() - xc) / H
        front[k] = (xs[sel].max() - xc) / H
    # 뒤쪽 t>0.48 은 꼬리·망토가 겹쳐 옷 실루엣이 아니다 → 측정 제외 (UNKNOWN)
    back[int(0.48 * NB):] = np.nan
    foot = t > 0.92
    stride = (np.ptp(xs[foot]) / H) if foot.sum() > 10 else 0.0
    return dict(back=back, front=front, stride=float(stride), H=int(H), xc=float(xc), top=int(top))


# ── 위상 ────────────────────────────────────────────────────────────────
def step_phase(strides):
    x = np.asarray(strides, float); x = x - x.mean(); n = len(x)
    ac = np.array([1.0] + [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, 16)])
    # 첫 국소 봉우리 (두 번째 봉우리 = 두 걸음이라 argmax 는 틀린다), 포물선으로 소수 보정
    P = None
    for k in range(3, 14):
        if ac[k] > 0.3 and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]:
            den = ac[k - 1] - 2 * ac[k] + ac[k + 1]
            P = k + (0.5 * (ac[k - 1] - ac[k + 1]) / den if den != 0 else 0); break
    if P is None:
        P = 3 + int(np.argmax(ac[3:15]))
    k = np.arange(n); w = 2 * np.pi / P
    c = complex((x * np.cos(w * k)).sum(), -(x * np.sin(w * k)).sum())
    phi = ((w * k - np.angle(c)) / (2 * np.pi)) % 1.0      # phi=0 : stride 최대(발 벌림)
    return P, phi


def bin_mean(profiles, phi, key):
    """위상 8칸별 평균 프로파일 (nan 무시)."""
    out = np.full((NPH, NB), np.nan)
    members = {i: [] for i in range(NPH)}
    for j, pr in enumerate(profiles):
        i = int(phi[j] * NPH) % NPH
        members[i].append(j)
    for i in range(NPH):
        if members[i]:
            out[i] = np.nanmean(np.stack([profiles[j][key] for j in members[i]]), 0)
    return out, members


# ── 분류·예측 ───────────────────────────────────────────────────────────
def classify(C, N):
    """C,N: (NPH, NB) 옷/나체 프로파일. t 칸마다 RIGID / FOLLOW / LOOSE."""
    cls = []; A = np.zeros(NB); B = np.zeros(NB); R2 = np.zeros(NB); SD = np.zeros(NB)
    for k in range(NB):
        c, n = C[:, k], N[:, k]
        ok = ~np.isnan(c) & ~np.isnan(n)
        if ok.sum() < 5:
            cls.append("NA"); A[k] = np.nan; B[k] = np.nan; continue
        c, n = c[ok], n[ok]
        sd = c.std(); SD[k] = sd
        if sd < 0.006:                        # 신장의 0.6% 미만 → 고정
            cls.append("RIGID"); A[k] = c.mean(); B[k] = 0; R2[k] = 1; continue
        X = np.stack([np.ones_like(n), n], 1)
        beta, *_ = np.linalg.lstsq(X, c, rcond=None)
        pred = X @ beta
        r2 = 1 - ((c - pred) ** 2).sum() / max(((c - c.mean()) ** 2).sum(), 1e-9)
        A[k], B[k], R2[k] = beta[0], beta[1], r2
        cls.append("FOLLOW" if r2 > 0.5 else "LOOSE")
    return cls, A, B, R2, SD


def predict_loo(C, N):
    """leave-one-phase-out: 위상 i 를 빼고 규칙을 배운 뒤 i 를 예측."""
    pred = np.full_like(C, np.nan); canon = np.full_like(C, np.nan)
    for i in range(NPH):
        keep = [j for j in range(NPH) if j != i]
        cls, A, B, R2, SD = classify(C[keep], N[keep])
        for k in range(NB):
            if cls[k] == "NA" or np.isnan(N[i, k]) or np.isnan(A[k]):
                continue
            pred[i, k] = A[k] + B[k] * N[i, k]
        canon[i] = C[keep[0]]                   # 캐논 1장 복사 기준선
    return pred, canon


def err(P, C):
    d = np.abs(P - C); return float(np.nanmean(d))


# ── 시트 ────────────────────────────────────────────────────────────────
def sheet(cl_files, nu_files, cl_prof, nu_prof, cl_phi, nu_phi, C, N, cls, pred, canon, res, path):
    from PIL import ImageFont
    cellw, cellh = 150, 210
    Wd = 20 + NPH * cellw + 340; Hd = 60 + cellh * 2 + 320
    im = Image.new("RGB", (Wd, Hd), (250, 249, 246)); d = ImageDraw.Draw(im)
    d.text((14, 10), f"GARMENT CHANGE STUDY  {res.get('view','side (east)')} view  clothed vs nude  8 phases  "
                     f"| rule-predict err {res['pred_err_px']:.1f}px  vs  copy-canon err {res['canon_err_px']:.1f}px",
           fill=(20, 20, 20))
    d.text((14, 26), "row1 clothed medoid per phase (back/front silhouette: green=RIGID  blue=FOLLOW body  red=LOOSE) | "
                     "row2 nude medoid per phase, orange = predicted clothed silhouette from rule", fill=(90, 90, 90))
    colmap = {"RIGID": (40, 160, 70), "FOLLOW": (40, 90, 220), "LOOSE": (220, 50, 40), "NA": (150, 150, 150)}

    def draw_frame(fp, pr, x0, y0, color_by_cls, extra=None):
        a = np.array(Image.open(fp).convert("RGBA")); a, _ = debg.clean(a)
        sc = (cellh - 20) / a.shape[0]
        fr = Image.fromarray(a).resize((int(a.shape[1] * sc), int(a.shape[0] * sc)))
        im.paste(fr, (x0, y0), fr)
        H, xc, top = pr["H"], pr["xc"], pr["top"]
        for k in range(NB):
            if np.isnan(pr["back"][k]):
                continue
            y = y0 + (top + (k + 0.5) / NB * H) * sc
            col = colmap[cls[k]] if color_by_cls else (120, 120, 120)
            for key in ("back", "front"):
                x = x0 + (xc + pr[key][k] * H) * sc
                d.line([x - 2, y, x + 2, y], fill=col, width=2)
            if extra is not None and not np.isnan(extra[0][k]):
                for e in extra:
                    x = x0 + (xc + e[k] * H) * sc
                    d.line([x - 3, y, x + 3, y], fill=(240, 140, 20), width=2)

    for i in range(NPH):
        x0 = 20 + i * cellw
        jc = res["clothed_medoid"][i]; jn = res["nude_medoid"][i]
        d.text((x0, 44), f"phase {i}/8  clothed f{jc+1:02d}", fill=(20, 20, 20))
        draw_frame(cl_files[jc], cl_prof[jc], x0, 58, True)
        d.text((x0, 58 + cellh), f"nude w{jn+1:02d}  -> predicted", fill=(20, 20, 20))
        draw_frame(nu_files[jn], nu_prof[jn], x0, 72 + cellh, False, extra=(pred_back[i], pred_front[i]))

    # 오른쪽: t 별 분류 막대 + 계수
    x0 = 30 + NPH * cellw; y0 = 58
    d.text((x0, 44), "t (0 crown .. 1 sole)  class  b(back) b(front)", fill=(20, 20, 20))
    for k in range(NB):
        y = y0 + k * 5
        col = colmap[cls[k]]
        d.rectangle([x0, y, x0 + 40, y + 4], fill=col)
        if k % 8 == 0:
            d.text((x0 + 46, y - 4), f"t={k/NB:.2f} {cls[k]:6s} b={res['b_back'][k]:+.2f}/{res['b_front'][k]:+.2f}", fill=(40, 40, 40))
    # 하단: 규칙 요약
    y = 60 + cellh * 2 + 20
    for line in res["rules_text"]:
        d.text((20, y), line, fill=(30, 30, 30)); y += 16
    im.save(path)


if __name__ == "__main__":
    cl_files = sorted(glob.glob(str(W / "gt/g*.png")))
    nu_files = sorted(glob.glob(str(W / "walkpng/w*.png")))
    cl_prof = [profile(p) for p in cl_files]
    nu_prof = [profile(p) for p in nu_files]
    Pc, cl_phi = step_phase([p["stride"] for p in cl_prof])
    Pn, nu_phi = step_phase([p["stride"] for p in nu_prof])

    Cb, mc = bin_mean(cl_prof, cl_phi, "back"); Cf, _ = bin_mean(cl_prof, cl_phi, "front")
    Nb, mn = bin_mean(nu_prof, nu_phi, "back"); Nf, _ = bin_mean(nu_prof, nu_phi, "front")

    cls_b, Ab, Bb, R2b, SDb = classify(Cb, Nb)
    cls_f, Af, Bf, R2f, SDf = classify(Cf, Nf)
    # 칸의 최종 분류 = 앞/뒤 중 더 '움직이는' 쪽
    order = {"NA": 0, "RIGID": 1, "FOLLOW": 2, "LOOSE": 3}
    cls = [cls_b[k] if order[cls_b[k]] >= order[cls_f[k]] else cls_f[k] for k in range(NB)]
    cls_back_only = cls_b; cls_front_only = cls_f

    pred_back, canon_back = predict_loo(Cb, Nb)
    pred_front, canon_front = predict_loo(Cf, Nf)
    # 평균 신장(px) 로 환산
    Hpx = np.mean([p["H"] for p in cl_prof])
    pe = (err(pred_back, Cb) + err(pred_front, Cf)) / 2 * Hpx
    ce = (err(canon_back, Cb) + err(canon_front, Cf)) / 2 * Hpx

    # medoid (위상 칸 대표: 칸 평균에 가장 가까운 프레임)
    def medoid(members, prof, Pb, Pf):
        out = []
        for i in range(NPH):
            js = members[i]
            if not js: out.append(0); continue
            dd = [np.nanmean(np.abs(prof[j]["back"] - Pb[i])) + np.nanmean(np.abs(prof[j]["front"] - Pf[i])) for j in js]
            out.append(js[int(np.argmin(dd))])
        return out

    # 규칙 요약 텍스트: t 구간별로 연속 분류 묶기
    rules_text = ["RULES FOUND per side (t range, class, slope vs nude silhouette, garment sd px over phases):"]
    for side, cl_, B_, SD_ in (("FRONT", cls_f, Bf, SDf), ("BACK ", cls_b, Bb, SDb)):
        s = 0
        for k in range(1, NB + 1):
            if k == NB or cl_[k] != cl_[s]:
                if (k - s) / NB >= 0.03:
                    bb = float(np.nanmean(B_[s:k])) if cl_[s] != "NA" else float("nan")
                    rules_text.append(f"  {side} t {s/NB:.2f}-{k/NB:.2f}  {cl_[s]:6s}  b={bb:+.2f}  sd={float(np.nanmean(SD_[s:k]))*Hpx:.1f}px"
                                      + ("   (tail+cape overlap: not garment, UNKNOWN)" if cl_[s] == "NA" else ""))
                s = k
    rules_text.append(f"step period: clothed {Pc:.2f} frames, nude {Pn:.2f} frames (different references, phase-matched by stride)")
    rules_text.append(f"leave-one-phase-out: rule prediction {pe:.1f}px mean |error|  vs  copy-one-canon {ce:.1f}px  (H={Hpx:.0f}px)")

    res = dict(schema="platekit.garment_change_study.v1",
               step_period_clothed=round(float(Pc),2), step_period_nude=round(float(Pn),2),
               clothed_medoid=medoid(mc, cl_prof, Cb, Cf), nude_medoid=medoid(mn, nu_prof, Nb, Nf),
               classes=cls, b_back=[round(float(x), 3) for x in Bb], b_front=[round(float(x), 3) for x in Bf],
               r2_back=[round(float(x), 3) for x in R2b], r2_front=[round(float(x), 3) for x in R2f],
               sd_back_px=[round(float(x * Hpx), 2) for x in SDb], sd_front_px=[round(float(x * Hpx), 2) for x in SDf],
               pred_err_px=round(pe, 2), canon_err_px=round(ce, 2), H_px=round(float(Hpx), 1),
               class_counts={c: cls.count(c) for c in ("RIGID", "FOLLOW", "LOOSE", "NA")},
               rules_text=rules_text)
    json.dump(res, open(W / "garment_study.json", "w"), indent=1, ensure_ascii=False)
    sheet(cl_files, nu_files, cl_prof, nu_prof, cl_phi, nu_phi, Cb, Nb, cls, pred_back, canon_back, res,
          "/mnt/user-data/outputs/GARMENT-CHANGE-STUDY.png")
    print("\n".join(rules_text))
    print("class counts:", res["class_counts"])
