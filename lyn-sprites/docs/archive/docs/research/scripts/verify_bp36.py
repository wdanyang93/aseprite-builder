"""body_parts_v1_36 검증 — 상대 리포트를 믿지 않고 화소 단위로 재계산한다.

검증 항목
  1. 구조     36 프레임 x 7 파츠 = 252 PNG, 전부 같은 캔버스
  2. 소유권   어떤 화소도 두 파츠에 속하지 않는다 (multiply-owned = 0)
  3. 합집합   파츠 합집합 화소수 = manifest 의 foreground_pixels (unowned = 0 의 재계산)
  4. 왕복     파츠를 다시 합쳤을 때 RGB/알파가 자기모순 없이 하나의 그림이 되는가
             (원본 frame_XXX.png 는 이 기계에 없다 — 대신 재조립본을 내 south 세트와
              대조해 출처 동일성을 판정한다)
  5. 경계     head/hip 절단이 수평 직선인가 (상대 문서가 스스로 인정한 한계의 실측)
  6. 꼬리     상대 tail 레이어 vs 내 검증된 꼬리 검출(southkit)의 IoU
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

ROOT = Path("/home/claude/handoff/work/bp36/body_parts_v1_36")
PARTS = ["head", "torso", "arm_l", "arm_r", "leg_l", "leg_r", "tail"]


def load_frame(i: int):
    d = ROOT / f"frame_{i:03d}"
    layers = {}
    for p in PARTS:
        layers[p] = np.array(Image.open(d / f"{p}.png").convert("RGBA"))
    return layers


def recombine(layers):
    """소유권이 배타적이라면 순서와 무관하게 단순 합이 성립한다."""
    canvas = None
    count = None
    for p in PARTS:
        a = layers[p]
        m = a[..., 3] > 0
        if canvas is None:
            canvas = np.zeros_like(a)
            count = np.zeros(m.shape, np.int32)
        canvas[m] = a[m]
        count += m
    return canvas, count


def tight(a):
    m = a[..., 3] > 0
    ys, xs = np.nonzero(m)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    manifest = json.load(open(ROOT / "parts_manifest.json"))
    report = {
        "frames": 0, "part_pngs": 0,
        "multiply_owned_total": 0,
        "union_mismatch_frames": [],
        "manifest_count_mismatch": [],
        "canvas": None,
        "composite_sha256": {},
        "straight_cut": {"head": 0, "hip": 0},
        "frames_checked_for_cut": 0,
    }
    canvases = set()
    comps = {}
    for i in range(36):
        layers = load_frame(i)
        report["part_pngs"] += len(layers)
        shp = {layers[p].shape for p in PARTS}
        canvases |= {s[:2] for s in shp}
        comp, cnt = recombine(layers)
        comps[i] = comp
        multi = int((cnt > 1).sum())
        report["multiply_owned_total"] += multi
        union = int((cnt > 0).sum())
        mf = manifest["frames"][i]
        if union != mf["foreground_pixels"]:
            report["union_mismatch_frames"].append((i, union, mf["foreground_pixels"]))
        for p in PARTS:
            if int((layers[p][..., 3] > 0).sum()) != mf["part_pixels"][p]:
                report["manifest_count_mismatch"].append((i, p))
        report["composite_sha256"][f"frame_{i:03d}"] = hashlib.sha256(
            comp.tobytes()).hexdigest()[:16]
        report["frames"] += 1

        # 경계 직선성: head 레이어의 최하단 행이 하나의 수평선인가
        hm = layers["head"][..., 3] > 0
        ys, xs = np.nonzero(hm)
        bottom = ys.max()
        frac = (ys == bottom).sum() / max((np.abs(ys - bottom) <= 1).sum(), 1)
        # head 최하단이 그 행 전체 폭을 차지하면 직선 절단
        row_w = (ys == bottom).sum()
        below = layers["torso"][..., 3] > 0
        bys, bxs = np.nonzero(below)
        if len(bys) and abs(int(bys.min()) - (bottom + 1)) <= 1 and row_w > 30:
            report["straight_cut"]["head"] += 1
        lm = (layers["leg_l"][..., 3] > 0) | (layers["leg_r"][..., 3] > 0)
        lys, lxs = np.nonzero(lm)
        tm = layers["torso"][..., 3] > 0
        tys, txs = np.nonzero(tm)
        if len(lys) and len(tys) and abs(int(lys.min()) - int(tys.max()) - 1) <= 1:
            top_w = (lys == lys.min()).sum()
            if top_w > 30:
                report["straight_cut"]["hip"] += 1
        report["frames_checked_for_cut"] += 1

    report["canvas"] = sorted(canvases)
    return report, comps


if __name__ == "__main__":
    rep, comps = main()
    out = {k: v for k, v in rep.items() if k != "composite_sha256"}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    np.save("/tmp/claude-0/-home-claude/60a1100f-4d22-5314-b5c3-abe3f28692c8/scratchpad/comp0.npy", comps[0])
    # 재조립본 저장 (위상 분석 입력)
    outdir = Path("/home/claude/handoff/work/mocap36/frames")
    outdir.mkdir(parents=True, exist_ok=True)
    for i, c in comps.items():
        Image.fromarray(c).save(outdir / f"f{i:02d}.png")
    print("재조립 36장 저장:", outdir)
