"""lr36 — 해부학적 왼발/오른발과 '앞으로 나온 다리'를 전신 맥락으로 판단한다.

지적받은 실수: 발만 보고 좌우를 골랐다. 근거의 크기 순서로 다시 짠다.

  1. 얼굴  — 두 눈이 다 보이고 좌우 대칭이면 관찰자를 보고 있다.
             관찰자를 보는 캐릭터의 해부학적 왼쪽 = 화면 오른쪽.
             (사람 확인: 화면 오른쪽(파랑) = 왼발 — 이 규칙과 일치)
  2. 진행  — 남쪽 걸음 = 관찰자 쪽으로 온다. 앞으로 나온 다리는 카메라에
             가까워지므로 화면상 더 아래까지 내려오고(길어지고), 발등·발가락이
             더 크게 보인다. 뒤로 간 다리는 짧아지고 발이 가려진다.
  3. 발    — 마지막 보조 근거. 발끝 y(더 낮은 쪽), 발 폭(더 넓은 쪽).
  4. 패턴  — 위 판단이 36장에 걸쳐 주기 P 로 번갈아야 한다. 번갈지 않으면
             개별 프레임이 아니라 규칙이 틀린 것이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/handoff/code")
sys.path.insert(0, "/home/claude/handoff/work")
import debg
import southkit as S
from gpiece import sagittal_x

M = Path("/home/claude/handoff/work/mocap36")
rep = json.load(open(M / "phase_report.json"))
ank = {a["i"]: a for a in json.load(open(M / "ankles.json"))}
P = rep["fit"]["gait_period"]
phi = np.array(rep["phase_aligned"])

HUMAN = {"canonical_frame": 5,
         "screen_right_is_anatomical_left": True,   # 사람 라벨: 파랑(화면 R) = 왼발
         "walk_direction": "south (toward viewer)"}


def load(i):
    a = np.array(Image.open(M / f"frames/f{i:02d}.png"))
    a, _ = debg.clean(a)
    m = a[..., 3] > 128
    r = S.tail_mask(a, m)
    if r is None:
        t = S.front_t(a, m); tl = S.tail_bright(a, m, t)
    else:
        tl, _z, t, _l = r
    return a, m & ~tl, t


def face_facing(a, body, t):
    """규칙 1: 홍채(어두운 주황) 두 개가 머리 띠 안에서 정중선 양쪽에 있는가."""
    head = body & (t < 0.24)
    rgb = a[..., :3].astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    iris = head & (r > 120) & (r < 220) & (g > 60) & (g < 160) & (b < 90) & (r - b > 70)
    ys, xs = np.nonzero(iris)
    if len(xs) < 20:
        return None, 0
    hx = np.nonzero(head)[1]
    mid = (hx.min() + hx.max()) / 2
    left, right = (xs < mid).sum(), (xs >= mid).sum()
    sym = min(left, right) / max(left, right)
    return bool(sym > 0.45), float(sym)          # 양쪽에 비슷하게 → 정면


def leg_evidence(a, body, t, sx):
    """규칙 2·3: 화면 좌/우 다리의 '앞으로 나옴' 근거."""
    out = {}
    for side, sel in (("L", lambda x: x < sx), ("R", lambda x: x >= sx)):
        low = body & (t > 0.70)
        ys, xs = np.nonzero(low)
        k = sel(xs)
        if k.sum() < 10:
            out[side] = dict(sole=np.nan, foot_w=0, leg_len=0); continue
        sole = ys[k].max()
        foot = (ys[k] > sole - 18)
        foot_w = np.ptp(xs[k][foot]) if foot.sum() else 0
        hip_y = np.nonzero(body & (t > 0.55) & (t < 0.57))[0].mean()
        out[side] = dict(sole=float(sole), foot_w=int(foot_w), leg_len=float(sole - hip_y))
    return out


def judge(i):
    a, body, t = load(i)
    sx = sagittal_x(body, t)
    facing, sym = face_facing(a, body, t)
    ev = leg_evidence(a, body, t, sx)
    dy = ev["R"]["sole"] - ev["L"]["sole"]          # >0 : 화면 오른 다리가 더 아래
    dw = ev["R"]["foot_w"] - ev["L"]["foot_w"]
    # 앞으로 나온 화면쪽: 더 아래(길다) 가 1순위, 발 폭이 2순위 동률 깨기
    if abs(dy) >= 2:
        fwd_screen = "R" if dy > 0 else "L"; basis = "lower_sole"
    elif abs(dw) >= 4:
        fwd_screen = "R" if dw > 0 else "L"; basis = "foot_width"
    else:
        fwd_screen = "?"; basis = "tie"
    # 화면 → 해부학 (규칙 1)
    to_anat = {"R": "LEFT", "L": "RIGHT", "?": "?"} if facing else {"R": "?", "L": "?", "?": "?"}
    return dict(index=i, facing_viewer=facing, iris_symmetry=round(sym, 2),
                sole_L=ev["L"]["sole"], sole_R=ev["R"]["sole"], dy=float(dy),
                footw_L=ev["L"]["foot_w"], footw_R=ev["R"]["foot_w"],
                forward_screen=fwd_screen, basis=basis,
                forward_anatomical=to_anat[fwd_screen])


def pattern_check(rows):
    """규칙 4: dy 부호가 주기 P 로 번갈아야 한다."""
    dy = np.array([r["dy"] for r in rows])
    k = np.arange(len(dy)); w = 2 * np.pi / P
    X = np.stack([np.cos(w * k), np.sin(w * k)], 1)
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    pred = X @ beta
    agree = float(((np.sign(pred) == np.sign(dy)) | (np.abs(dy) < 2)).mean())
    corr = float(np.corrcoef(pred, dy)[0, 1])
    # 위상 모델(asym 기반)과의 정렬: dy 가 asym 위상에 대해 어떤 각도로 서는가
    return dict(sinusoid_r=round(corr, 3), sign_agreement=round(agree, 3),
                amp_px=round(float(np.hypot(*beta)), 1))


def sheet(rows, path):
    cols = 12; cw, ch = 96, 262; sc = 0.42
    W = cols * cw + 20; H = 3 * ch + 60
    im = Image.new("RGB", (W, H), (250, 249, 246))
    d = ImageDraw.Draw(im)
    d.text((10, 8), "front-36: forward leg by (1) face facing viewer -> screen R = anatomical LEFT, "
                    "(2) south walk -> forward leg reaches LOWER, (3) foot width as tie-break", fill=(20, 20, 20))
    d.text((10, 22), "label = anatomical forward leg; red bar = lower sole side. '?' = tie (<2px)", fill=(90, 90, 90))
    for r in rows:
        i = r["index"]
        a = np.array(Image.open(M / f"frames/f{i:02d}.png"))
        fr = Image.fromarray(a).resize((int(a.shape[1] * sc), int(a.shape[0] * sc)))
        x = 10 + (i % cols) * cw; y = 40 + (i // cols) * ch
        im.paste(fr, (x, y), fr)
        col = (200, 40, 40) if r["forward_anatomical"] == "LEFT" else (30, 90, 200) if r["forward_anatomical"] == "RIGHT" else (120, 120, 120)
        d.text((x, y + int(a.shape[0] * sc) + 2), f"{i:02d} {r['forward_anatomical'][:1]} dy{r['dy']:+.0f}", fill=col)
        if r["forward_screen"] in "LR":
            bx = x + (int(a.shape[1] * sc) * (0.72 if r["forward_screen"] == "R" else 0.28))
            d.line([bx, y + int(a.shape[0] * sc) - 4, bx, y + int(a.shape[0] * sc)], fill=(220, 40, 40), width=4)
    im.save(path)


if __name__ == "__main__":
    rows = [judge(i) for i in range(36)]
    pat = pattern_check(rows)
    fac = sum(1 for r in rows if r["facing_viewer"])
    out = dict(schema="platekit.anatomical_lr.v1", human=HUMAN, period=P,
               facing_viewer_frames=fac, pattern=pat, frames=rows)
    json.dump(out, open(M / "anatomical_lr.json", "w"), indent=1, ensure_ascii=False)
    sheet(rows, "/mnt/user-data/outputs/ANSWER-1-forward-leg-36.png")
    seq = "".join(r["forward_anatomical"][:1] for r in rows)
    print("facing viewer:", fac, "/36")
    print("forward leg (anatomical) sequence:", seq)
    print("pattern:", pat)
    print("basis:", {b: sum(r["basis"] == b for r in rows) for b in ("lower_sole", "foot_width", "tie")})
