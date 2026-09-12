"""assemble — 8위상 걷기 루프를 착의 프레임 은행에서 조립한다 (동쪽, 정규화 좌표).

팔 분리는 이 옷에서 불가능했다 (장갑·조끼·파우치·레깅스가 전부 같은 갈색, 셔츠 무늬가 살색).
그래서 절단선을 팔꿈치 위로 올린다 — 어떤 팔다리도 두 프레임에 걸쳐 잘리지 않게.

A. 고정 상부 + 위상 기증
   T        몸통 원판 (동쪽 원본, 발 교차, 팔이 가장 내려간 프레임): t < CUT_A(0.42) 만 사용
            = 머리·머리카락·망토 윗부분·어깨·가슴·소매·배낭 윗부분 (걸어도 안 변하는 부분)
   D_i      위상 i 기증 프레임 (pairs8 의 착의 짝 = 나체 기준 자세에 팔·다리가 함께 가장 가까운 프레임)
            t ≥ CUT_A 전부 = 팔뚝·장갑·벨트·파우치·다리·장화·망토 자락 + 꼬리 전체
   정렬     D_i 를 허리 띠(t .40~.44, 꼬리 제외) 중심 x 가 T 와 같도록 수평 이동. 절단선 ±FEATHER 섞음.
B. 상하체 분리 (캐논 고정 없음)  CUT_B(0.62)
   상체 = 팔 기증 A_i 의 t<0.62 (머리는 T 로 잠금), 하체 = 다리 기증 L_i 의 t≥0.62, 허벅지 띠로 정렬.
C. 기준선 = 기증 프레임 그대로 + 머리 잠금 (절단 없음) — 비교용.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/claude/handoff/work")
import bank

B = Path("/home/claude/handoff/work/bank")
OUT = Path("/mnt/user-data/outputs")
NPH = 8
CUT_A = 0.42
CUT_B = 0.62
FEATHER = 6
HEAD_T = 0.22
MIRROR_PENALTY = 0.06


def load(fid):
    a = np.array(Image.open(B / "norm" / f"{fid}.png").convert("RGBA"))
    m = a[..., 3] > 128
    tl, xc, H0, side = bank.tail_mask_side(a, m)
    body = m & ~tl
    ys, xs = np.nonzero(body)
    top, bot = ys.min(), ys.max(); H = bot - top + 1
    t = (np.arange(a.shape[0]) - top) / H
    return dict(id=fid, a=a, m=m, tail=tl, body=body, t=np.repeat(t[:, None], a.shape[1], 1),
                xc=float(xc), top=int(top), bot=int(bot), H=int(H))


def band_x(fr, lo, hi, use_body=True):
    src = fr["body"] if use_body else fr["m"]
    band = src & (fr["t"] > lo) & (fr["t"] < hi)
    xs = np.nonzero(band)[1]
    return float(xs.mean()) if len(xs) else fr["xc"]


def shift(img, dx):
    out = np.zeros_like(img)
    if dx >= 0: out[:, dx:] = img[:, :img.shape[1] - dx]
    else: out[:, :dx] = img[:, -dx:]
    return out


def over(dst, src, w):
    a = (src[..., 3] / 255.0) * w
    a3 = a[..., None]
    dst[..., :3] = (src[..., :3] * a3 + dst[..., :3] * (1 - a3)).astype(np.uint8)
    dst[..., 3] = np.clip(dst[..., 3] + a * (255 - dst[..., 3]), 0, 255).astype(np.uint8)


def w_upper(T, cut):
    y = T * 600.0; d = (cut * 600.0 - y) / FEATHER
    return np.clip(0.5 + 0.5 * d, 0, 1)


def blend(U, L, w):
    """상부 U 와 하부 L 을 가중 w(상부) 로 교차 섞기. 한쪽만 있는 곳은 그쪽 알파를 그대로 (반투명 띠 방지)."""
    Ua = U[..., 3] / 255.0; La = L[..., 3] / 255.0
    both = (Ua > 0) & (La > 0)
    wa = np.where(both, w, np.where(Ua > 0, 1.0, 0.0))            # 상부 가중
    out = np.zeros_like(U)
    a = wa * Ua + (1 - wa) * La
    rgb = (wa * Ua)[..., None] * U[..., :3] + ((1 - wa) * La)[..., None] * L[..., :3]
    rgb = rgb / np.maximum(a[..., None], 1e-6)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8); out[..., 3] = np.clip(a * 255, 0, 255).astype(np.uint8)
    return out


def layer(src, mask):
    o = np.zeros_like(src); o[mask] = src[mask]; return o


def compose_A(Tfr, Dfr, dx):
    Da = shift(Dfr["a"], dx); Dm = shift(Dfr["m"], dx); Dtail = shift(Dfr["tail"], dx); DT = shift(Dfr["t"], dx)
    lower = np.zeros_like(Da)
    over(lower, Da, Dtail.astype(float))                                     # 기증 꼬리 전체
    over(lower, Da, (Dm & ~Dtail & (DT >= CUT_A - 2 * FEATHER / 600.0)).astype(float))   # 기증 하부 (+페더 여유)
    upper = layer(Tfr["a"], Tfr["body"] & (Tfr["t"] <= CUT_A + 2 * FEATHER / 600.0))     # 고정 상부 (꼬리 제외)
    return blend(upper, lower, w_upper(Tfr["t"], CUT_A))


def head_lock(canvas, Tfr):
    head = layer(Tfr["a"], Tfr["body"] & (Tfr["t"] <= HEAD_T + 12 / 600.0))
    hw = np.clip(((HEAD_T * 600.0) - Tfr["t"] * 600.0) / 6 + 0.5, 0, 1)
    canvas[:] = blend(head, canvas, hw)


def compose_B(Tfr, Ufr, Lfr, dxU, dxL):
    """꼬리는 통째로 하체 기증에서 (두 프레임에 걸쳐 자르지 않는다)."""
    La = shift(Lfr["a"], dxL); Lm = shift(Lfr["m"], dxL); LT = shift(Lfr["t"], dxL); Ltail = shift(Lfr["tail"], dxL)
    Ua = shift(Ufr["a"], dxU); Ub = shift(Ufr["body"], dxU); UT = shift(Ufr["t"], dxU)
    lower = np.zeros_like(La)
    over(lower, La, Ltail.astype(float))
    over(lower, La, (Lm & ~Ltail & (LT >= CUT_B - 2 * FEATHER / 600.0)).astype(float))
    upper = layer(Ua, Ub & (UT <= CUT_B + 2 * FEATHER / 600.0))
    canvas = blend(upper, lower, w_upper(UT, CUT_B))
    head_lock(canvas, Tfr)
    return canvas


def compose_C(Tfr, Dfr, dx):
    canvas = np.zeros_like(Tfr["a"])
    over(canvas, shift(Dfr["a"], dx), shift(Dfr["m"], dx).astype(float))
    head_lock(canvas, Tfr)
    return canvas


def strip_and_gif(frames, name, title):
    sc = 0.5; w, h = int(640 * sc), int(720 * sc)
    im = Image.new("RGB", (NPH * w + 20, h + 40), (250, 249, 246)); d = ImageDraw.Draw(im)
    d.text((10, 6), title, fill=(20, 20, 20))
    for i, f in enumerate(frames):
        fr = Image.fromarray(f).resize((w, h), Image.LANCZOS)
        im.paste(fr, (10 + i * w, 30), fr); d.text((10 + i * w, 18), f"{i}/8", fill=(60, 60, 60))
    im.save(OUT / f"{name}-strip.png")
    gif = []
    for f in frames:
        bg = Image.new("RGBA", (640, 720), (250, 249, 246, 255)); bg.alpha_composite(Image.fromarray(f))
        gif.append(bg.convert("P", palette=Image.ADAPTIVE, colors=255))
    gif[0].save(OUT / f"{name}.gif", save_all=True, append_images=gif[1:], duration=110, loop=0, disposal=2)


def main():
    P = json.load(open(B / "pose.json")); pairs = json.load(open(B / "pairs8.json"))["pairs"]
    east = [i for i in P if P[i]["clothed"] and P[i]["set"] in ("cE1", "cE2")]
    Tid = min([i for i in east if P[i]["together"]], key=lambda i: P[i]["arm_spread"])
    Tfr = load(Tid); waistT = band_x(Tfr, 0.40, 0.44)

    def pick(srcs):
        return min(srcs, key=lambda s: s[0] + (MIRROR_PENALTY if P[s[1]]["set"].startswith("cW") else 0))[1]

    fA, fB, fC, log = [], [], [], []
    for p in pairs:
        # 관절 매칭 기증(짝의 착의 프레임) 을 우선하되 반전 프레임은 벌점
        Did = p["clothed"]
        Dfr = load(Did); dx = int(round(waistT - band_x(Dfr, 0.40, 0.44)))
        fA.append(compose_A(Tfr, Dfr, dx)); fC.append(compose_C(Tfr, Dfr, dx))
        Uid = pick(p["arm_src"]); Lid = pick(p["leg_src"])
        Ufr = load(Uid); Lfr = load(Lid)
        dxU = int(round(waistT - band_x(Ufr, 0.40, 0.44)))
        thighU = band_x(Ufr, 0.60, 0.64) + dxU
        dxL = int(round(thighU - band_x(Lfr, 0.60, 0.64)))
        fB.append(compose_B(Tfr, Ufr, Lfr, dxU, dxL))
        log.append(dict(phase=p["phase"], torso=Tid, donor=Did, dx=dx, upper=Uid, lower=Lid, dxU=dxU, dxL=dxL, nude_ref=p["nude"]))
        print(f"phi {p['phase']}/8  A/C donor {Did} dx {dx:+d} | B upper {Uid} lower {Lid} dx {dxU:+d}/{dxL:+d}")
    for name, fr in (("walkA", fA), ("walkB", fB), ("walkC", fC)):
        (OUT / name).mkdir(exist_ok=True)
        for i, f in enumerate(fr): Image.fromarray(f).save(OUT / name / f"{name}_{i}.png")
    strip_and_gif(fA, "WALK-A-fixed-upper", f"A: fixed upper body (t<0.42) from {Tid} + forearm/glove/legs/boots/tail per phase from the joint-matched donor")
    strip_and_gif(fB, "WALK-B-split", "B: upper (t<0.62, arm donor, head locked) + lower (t>=0.62, leg donor), aligned at the thigh")
    strip_and_gif(fC, "WALK-C-headlock", "C: donor frames as-is + head lock (no cut) — baseline")
    json.dump(dict(torso=Tid, cut_A=CUT_A, cut_B=CUT_B, feather=FEATHER, frames=log), open(OUT / "walk_assembly.json", "w"), indent=1)
    # 리뷰 GIF: A | B | C 나란히, 배경 위
    gif = []
    for i in range(NPH):
        bg = Image.new("RGBA", (3 * 420 + 40, 500), (250, 249, 246, 255)); d = ImageDraw.Draw(bg)
        for k, (nm, fr) in enumerate((("A fixed upper", fA), ("B upper/lower split", fB), ("C head-lock only", fC))):
            im = Image.fromarray(fr[i]).crop((40, 80, 640, 720)).resize((394, 420), Image.LANCZOS)
            bg.alpha_composite(im, (10 + k * 420, 60)); d.text((10 + k * 420, 40), f"{nm}   phase {i}/8", fill=(20, 20, 20, 255))
        d.text((10, 10), "8-frame walk loop assembled from the clothed bank (east). A: torso above the elbow fixed; B: upper/lower body from different donors; C: donor frames + head lock", fill=(20, 20, 20, 255))
        gif.append(bg.convert("P", palette=Image.ADAPTIVE, colors=255))
    gif[0].save(OUT / "WALK-review-ABC.gif", save_all=True, append_images=gif[1:], duration=120, loop=0, disposal=2)


if __name__ == "__main__":
    main()
