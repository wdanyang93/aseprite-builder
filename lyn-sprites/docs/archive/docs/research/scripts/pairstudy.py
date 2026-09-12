"""pairstudy — 자세로 동기화된 8쌍(착의 g / 나체 w)에서만 옷 형태 변화를 잰다.

이전 실수: 위상 칸 평균끼리 비교 → 손발이 안 맞는 프레임이 섞였다.
이제는 sync.py 가 고른 쌍(같은 phi, 같은 앞다리) 만 쓴다.

잰 것 (몸축 t 80칸, 앞쪽/뒤쪽 실루엣 끝, H 정규화, 꼬리 제외)
  gap_front(t, i) = 옷 앞끝 − 몸 앞끝     옷이 몸보다 얼마나 나와 있나 (위상 i)
  gap_back (t, i) = 몸 뒤끝 − 옷 뒤끝     (뒤 t>0.47 은 꼬리·망토 → UNKNOWN)
  RIGID  : gap 이 위상과 무관 (sd 작음) — 옷이 몸에 고정된 두께
  FOLLOW : 옷끝 = a + b·몸끝 로 설명 (R² 높음) — 팔다리를 따라 늘어나거나 줄어듦
  LOOSE  : 위상 따라 변하지만 몸끝으로 설명 안 됨
그림: 위상마다 착의 프레임 위에 '같은 자세 나체 실루엣' 을 정렬해 겹친다 → 옷이 몸에서
  어디가 얼마나 벗어나며 위상마다 어떻게 달라지는지 눈으로 본다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/handoff/work")
import garmentstudy as G
import pose
from garmentstudy import NB

W = Path("/home/claude/handoff/work")
NPH = 8


def head_x(body, t):
    ys, xs = np.nonzero(body & (t[:, None] < 0.2))
    return float(xs.mean())


def profile_head(p):
    """garmentstudy.profile 과 같되 x 기준 = 머리 중심 (옷에 둔감)."""
    a, body, t, xc, top, H, bot = pose.load(p)
    xr = head_x(body, t)
    ys, xs = np.nonzero(body)
    tt = (ys - top) / H
    back = np.full(NB, np.nan); front = np.full(NB, np.nan)
    b = np.minimum((tt * NB).astype(int), NB - 1)
    for k in range(NB):
        sel = b == k
        if sel.sum() < 3:
            continue
        back[k] = (xs[sel].min() - xr) / H
        front[k] = (xs[sel].max() - xr) / H
    back[int(0.47 * NB):] = np.nan
    return dict(back=back, front=front, H=int(H), xr=xr, top=int(top))


def main():
    sy = json.load(open(W / "pose_sync.json"))
    pairs = sy["pairs"]
    cl_files = [W / "gt" / sy["clothed"]["files"][p["clothed"]] for p in pairs]
    nu_files = [W / "walkpng" / sy["nude"]["files"][p["nude"]] for p in pairs]
    cp = [profile_head(str(f)) for f in cl_files]; npf = [profile_head(str(f)) for f in nu_files]
    Cb = np.stack([p["back"] for p in cp]); Cf = np.stack([p["front"] for p in cp])
    Nb = np.stack([p["back"] for p in npf]); Nf = np.stack([p["front"] for p in npf])
    Hpx = float(np.mean([p["H"] for p in cp]))

    cls_b, Ab, Bb, R2b, SDb = G.classify(Cb, Nb)
    cls_f, Af, Bf, R2f, SDf = G.classify(Cf, Nf)
    pb, cb = G.predict_loo(Cb, Nb); pf, cf = G.predict_loo(Cf, Nf)
    pe = (G.err(pb, Cb) + G.err(pf, Cf)) / 2 * Hpx
    ce = (G.err(cb, Cb) + G.err(cf, Cf)) / 2 * Hpx
    gap_f = (Cf - Nf) * Hpx; gap_b = (Nb - Cb) * Hpx          # px, 양수 = 옷이 몸 밖

    def segments(cl_, B_, SD_, gap):
        out = []; s = 0
        for k in range(1, NB + 1):
            if k == NB or cl_[k] != cl_[s]:
                if (k - s) / NB >= 0.03:
                    g = gap[:, s:k]
                    out.append(dict(t0=round(s / NB, 2), t1=round(k / NB, 2), cls=cl_[s],
                                    b=None if cl_[s] == "NA" else round(float(np.nanmean(B_[s:k])), 2),
                                    sd_px=round(float(np.nanmean(SD_[s:k])) * Hpx, 1),
                                    gap_mean_px=None if cl_[s] == "NA" else round(float(np.nanmean(g)), 1),
                                    gap_min_px=None if cl_[s] == "NA" else round(float(np.nanmin(np.nanmean(g, 1))), 1),
                                    gap_max_px=None if cl_[s] == "NA" else round(float(np.nanmax(np.nanmean(g, 1))), 1)))
                s = k
        return out
    seg_f = segments(cls_f, Bf, SDf, gap_f); seg_b = segments(cls_b, Bb, SDb, gap_b)

    res = dict(schema="platekit.garment_change_paired.v1", pairs=pairs, H_px=Hpx,
               pred_err_px=round(pe, 2), canon_err_px=round(ce, 2),
               front=seg_f, back=seg_b, classes_front=cls_f, classes_back=cls_b)
    json.dump(res, open(W / "garment_paired.json", "w"), indent=1, ensure_ascii=False)

    # ── 시트: 착의 프레임 + 정렬된 나체 실루엣 윤곽 ───────────────────────
    cw, ch = 190, 300
    Wd = 20 + NPH * cw + 520; Hd = 60 + ch + 40
    im = Image.new("RGB", (Wd, Hd), (250, 249, 246)); d = ImageDraw.Draw(im)
    d.text((14, 8), f"GARMENT vs BODY on POSE-SYNCED pairs — cyan outline = nude body silhouette of the SAME pose, "
                    f"aligned on sole + HEAD center (garment-free anchor) | rule err {pe:.1f}px vs copy-one-canon {ce:.1f}px", fill=(20, 20, 20))
    d.text((14, 24), "right strip per frame: front-side class per t (green RIGID / blue FOLLOW / red LOOSE), "
                     "left strip: back side (grey = tail/cape UNKNOWN)", fill=(90, 90, 90))
    colmap = {"RIGID": (40, 160, 70), "FOLLOW": (40, 90, 220), "LOOSE": (220, 50, 40), "NA": (170, 170, 170)}
    for i in range(NPH):
        x0 = 20 + i * cw; y0 = 44
        a, body, t, xc, top, H, bot = pose.load(str(cl_files[i]))
        b, nbody, t2, xd, top2, H2, bot2 = pose.load(str(nu_files[i]))
        sc = (ch - 20) / a.shape[0]
        fr = Image.fromarray(a).resize((int(a.shape[1] * sc), int(a.shape[0] * sc)))
        im.paste(fr, (x0 + 14, y0 + 14), fr)
        # 나체 실루엣을 착의 좌표로: 발바닥·몸통중심 정렬 + 신장 비율
        k = H / H2
        hx1 = head_x(body, t); hx2 = head_x(nbody, t2)
        from scipy.ndimage import binary_erosion
        edge = nbody & ~binary_erosion(nbody, iterations=2)
        ey, ex = np.nonzero(edge)
        X = x0 + 14 + (hx1 + (ex - hx2) * k) * sc; Y = y0 + 14 + (bot + (ey - bot2) * k) * sc
        for px, py in zip(X[::2], Y[::2]):
            d.point((px, py), fill=(0, 170, 190))
        d.text((x0, y0), f"phi {i}/8  {cl_files[i].stem} + {nu_files[i].stem}", fill=(20, 20, 20))
        # 분류 띠
        for kk in range(NB):
            yy = y0 + 14 + (top + (kk + 0.5) / NB * H) * sc
            d.rectangle([x0 + 14 + a.shape[1] * sc + 2, yy - 1, x0 + 14 + a.shape[1] * sc + 8, yy + 1], fill=colmap[cls_f[kk]])
            d.rectangle([x0 + 4, yy - 1, x0 + 10, yy + 1], fill=colmap[cls_b[kk]])
    x0 = 30 + NPH * cw; y = 44
    lines = ["FRONT side (t range, class, b, gap of garment beyond body: mean [min..max over phases] px):"]
    for s in seg_f:
        lines.append(f"  t {s['t0']:.2f}-{s['t1']:.2f} {s['cls']:6s} b={s['b']}  gap {s['gap_mean_px']} [{s['gap_min_px']}..{s['gap_max_px']}]  sd {s['sd_px']}")
    lines.append("BACK side:")
    for s in seg_b:
        lines.append(f"  t {s['t0']:.2f}-{s['t1']:.2f} {s['cls']:6s} b={s['b']}  gap {s['gap_mean_px']} [{s['gap_min_px']}..{s['gap_max_px']}]  sd {s['sd_px']}")
    lines += [f"H = {Hpx:.0f}px. pairs: " + ", ".join(f"{c.stem}/{n.stem}" for c, n in zip(cl_files, nu_files))]
    for line in lines:
        d.text((x0, y), line, fill=(30, 30, 30)); y += 15
    im.save("/mnt/user-data/outputs/GARMENT-ON-SYNCED-PAIRS.png")
    print("\n".join(lines))
    print(f"rule err {pe:.1f}px  copy-canon err {ce:.1f}px")


if __name__ == "__main__":
    main()
