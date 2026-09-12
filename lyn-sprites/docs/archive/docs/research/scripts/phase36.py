"""phase36 — 36 관측 프레임에서 보행 위상을 추정해 8위상으로 재표본한다.

인계서 Milestone A(입고·측정) + C(위상·재표본)의 정면 최소 절편.
관측을 고치지 않는다. 순서를 믿지 않고 신호로 검증한다. 낮은 신뢰는 낮다고 적는다.

신호 (전부 기하 — 실패 기록 D/F/G 의 교훈대로 겉모습을 쓰지 않는다)
    stride(k)   발 띠(t>0.94)의 좌우 퍼짐 / 신장      — 걸음마다 두 번 진동 (2f)
    asym(k)     골반 아래 질량의 좌우 부호 비대칭       — 어느 다리가 앞인가 (f)
    sole(k)     몸 최하단 y                          — 접지
    bob(k)      정수리 y                             — 몸통 상하 진동 (2f)

위상 모델
    gait 주기 P 는 asym 의 자기상관 첫 봉우리. stride 는 P/2 에서 진동해야 하며
    이 관계가 성립하지 않으면 주기 추정을 신뢰하지 않는다.
    phi_k = atan2 를 기본 주파수 성분에서 취하고, 이상치는 2조화 적합 잔차의
    Hampel 규칙으로 표시만 한다 (자동 수정 금지).

재표본
    phi_i = i/8. phi=0 은 '화면 왼쪽 다리 앞 + 접지' 프레임에 정렬.
    각 위상의 대표는 **관측 medoid** — 새 화소를 만들지 않는다 (렌더러는 다음 단계).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/claude/handoff/code")
sys.path.insert(0, "/home/claude/handoff/work")

import debg
import southkit as S

WORK = Path("/home/claude/handoff/work")
FRAMES = WORK / "mocap36/frames"
BP36 = WORK / "bp36/body_parts_v1_36"
OUT = WORK / "mocap36"


# ── 입고 ────────────────────────────────────────────────────────────────
def ingest_manifest():
    man = {"schema": "platekit.motion_ingest.v1", "frames": []}
    for i in range(36):
        e = {"index": i, "composite": f"frames/f{i:02d}.png"}
        e["composite_sha256"] = hashlib.sha256(
            (FRAMES / f"f{i:02d}.png").read_bytes()).hexdigest()
        parts = {}
        d = BP36 / f"frame_{i:03d}"
        for p in ["head", "torso", "arm_l", "arm_r", "leg_l", "leg_r", "tail"]:
            parts[p] = hashlib.sha256((d / f"{p}.png").read_bytes()).hexdigest()[:16]
        e["part_sha256_16"] = parts
        man["frames"].append(e)
    return man


# ── 프레임 특징 ──────────────────────────────────────────────────────────
def features():
    rows = []
    for i in range(36):
        a = np.array(Image.open(FRAMES / f"f{i:02d}.png"))
        a, removed = debg.clean(a)
        m = a[..., 3] > 128
        r = S.tail_mask(a, m)
        if r is None:
            t = S.front_t(a, m)
            tl = S.tail_bright(a, m, t)
            method = "bright_fallback"
        else:
            tl, _z, t, _l = r
            method = "lineart"
        body = m & ~tl
        ys, xs = np.nonzero(body)
        H = float(ys.max() - ys.min() + 1)
        sx = S.sagittal_x(body, t) if hasattr(S, "sagittal_x") else float(xs.mean())
        foot = body & (t > 0.94)
        fys, fxs = np.nonzero(foot)
        stride = float((np.percentile(fxs, 98) - np.percentile(fxs, 2)) / H) if len(fxs) > 10 else 0.0
        low = body & (t > 0.62)
        lys, lxs = np.nonzero(low)
        left = (lxs < sx).sum(); right = (lxs >= sx).sum()
        asym = float((left - right) / max(left + right, 1))
        soleL = float(lys[lxs < sx].max()) if (lxs < sx).any() else np.nan
        soleR = float(lys[lxs >= sx].max()) if (lxs >= sx).any() else np.nan
        rows.append(dict(index=i, height=H, sagittal_x=float(sx),
                         stride=stride, asym=asym,
                         sole=float(ys.max()), crown=float(ys.min()),
                         sole_left=soleL, sole_right=soleR,
                         tail_method=method, black_residue_px=int(removed)))
    return rows


# ── 주기·위상 ────────────────────────────────────────────────────────────
def autocorr_period(x, lo=4, hi=18):
    x = np.asarray(x, float); x = x - x.mean()
    ac = [1.0] + [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, hi + 1)]
    ac = np.array(ac)
    k = lo + int(np.argmax(ac[lo:hi + 1]))
    return k, ac


def fit_phase(rows):
    n = len(rows)
    asym = np.array([r["asym"] for r in rows])
    stride = np.array([r["stride"] for r in rows])
    Pg, ac_a = autocorr_period(asym, 8, 20)          # gait 주기 (한 왕복)
    Ps, ac_s = autocorr_period(stride, 4, 12)        # 걸음(반주기)
    ratio = Pg / max(Ps, 1)

    # 기본 주파수 성분으로 위상
    k = np.arange(n)
    w = 2 * np.pi / Pg
    def comp(sig):
        c = (sig - sig.mean())
        return complex((c * np.cos(w * k)).sum(), -(c * np.sin(w * k)).sum())
    # asym 이 f, bob/stride 가 2f. 위상은 asym 기준.
    a0 = np.angle(comp(asym))
    phi = ((w * k + (-a0)) / (2 * np.pi)) % 1.0

    # 2조화 적합 잔차로 이상치 표시 (Hampel)
    X = np.stack([np.ones(n), np.cos(w * k), np.sin(w * k),
                  np.cos(2 * w * k), np.sin(2 * w * k)], 1)
    out_flags = {}
    resid_total = np.zeros(n)
    for name, sig in [("asym", asym), ("stride", stride)]:
        beta, *_ = np.linalg.lstsq(X, sig, rcond=None)
        r = sig - X @ beta
        med = np.median(r); mad = np.median(np.abs(r - med)) + 1e-9
        z = np.abs(r - med) / (1.4826 * mad)
        out_flags[name] = z
        resid_total += z
    outliers = sorted(int(i) for i in np.nonzero(resid_total > 7.0)[0])

    # 순서 유효성: 인접 프레임 신호 매끄러움 vs 무작위 순서
    def lag1(sig):
        return float(np.corrcoef(sig[:-1], sig[1:])[0, 1])
    rng = np.random.default_rng(0)
    sh = [lag1(rng.permutation(asym)) for _ in range(200)]
    seq_score = lag1(asym)
    seq_p = float((np.array(sh) >= seq_score).mean())

    return dict(gait_period=int(Pg), step_period=int(Ps),
                period_ratio=float(ratio),
                period_relation_ok=bool(1.6 <= ratio <= 2.4),
                phase=[float(p) for p in phi],
                outliers=outliers,
                order_lag1=seq_score, order_shuffle_p=seq_p,
                cycles_in_36=float(n / Pg))


def resample8(rows, fit):
    """위상 0 = '화면 왼쪽 절반이 무겁고(asym>0) 접지'에 정렬한 medoid 8장."""
    phi = np.array(fit["phase"])
    asym = np.array([r["asym"] for r in rows])
    # phi=0 후보: asym 최대 근처
    shift = phi[int(np.argmax(asym))]
    phi = (phi - shift) % 1.0
    sel = {}
    for i in range(8):
        target = i / 8
        d = np.minimum(np.abs(phi - target), 1 - np.abs(phi - target))
        d[fit["outliers"]] = 9.9
        sel[i] = int(np.argmin(d))
    return sel, [float(p) for p in phi]


def loop_qa(sel):
    imgs = {i: np.array(Image.open(FRAMES / f"f{sel[i]:02d}.png")).astype(int)
            for i in range(8)}
    diffs = []
    for i in range(8):
        j = (i + 1) % 8
        a, b = imgs[i], imgs[j]
        m = (a[..., 3] > 0) | (b[..., 3] > 0)
        diffs.append(float(np.abs(a[..., :3] - b[..., :3])[m].mean()))
    internal = diffs[:-1]
    return dict(neighbor_rgb_diff=[round(d, 2) for d in diffs],
                wrap_7_to_0=round(diffs[-1], 2),
                internal_mean=round(float(np.mean(internal)), 2),
                internal_max=round(float(np.max(internal)), 2),
                wrap_within_internal_range=bool(diffs[-1] <= np.max(internal) * 1.15))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    man = ingest_manifest()
    json.dump(man, open(OUT / "ingest_manifest.json", "w"), indent=1)
    rows = features()
    fit = fit_phase(rows)
    sel, phi0 = resample8(rows, fit)
    qa = loop_qa(sel)
    rep = dict(schema="platekit.phase_report.v1",
               features=rows, fit=fit,
               phase_aligned=phi0,
               loop8_medoid={str(i): sel[i] for i in range(8)},
               loop_qa=qa,
               notes=["medoid 는 관측 프레임 그대로다 — 새 화소를 만들지 않았다",
                      "이상치는 표시만 하고 수정하지 않았다 (인계서 Stage 4)"])
    json.dump(rep, open(OUT / "phase_report.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(fit=fit, loop8=sel, qa=qa), ensure_ascii=False, indent=1))
