"""walkbank 단위·회귀 테스트.

- 합성 시트로 격자 슬라이스 (bank.slice_sheet)
- 합성 신호로 위상·N/F 교대 (posebank.phases_for_set)
- 페더 교차 합성이 반투명 띠를 만들지 않는가 (assemble.blend)
- 골든 파일 구조 회귀 (labels 437, pose 293, pairs 8, front pairs 8)
실제 시트가 assets/raw/sheets/ 에 있으면 은행 전체 회귀도 돈다 (없으면 skip).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "platekit" / "walkbank"))

import bank            # noqa: E402
import posebank        # noqa: E402
import assemble        # noqa: E402


def synthetic_sheet(cols=3, rows=2, cell=(120, 160)):
    """셀마다 세로로 긴 불투명 덩어리 하나 (몸) — 알파 투영 빈 간격이 생기도록 여백을 둔다."""
    W, H = cols * cell[0], rows * cell[1]
    a = np.zeros((H, W, 4), np.uint8)
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * cell[1] + 15, c * cell[0] + 30
            a[y0:y0 + 120, x0:x0 + 50] = (200, 180, 160, 255)
    return a


def test_slice_grid_by_gaps():
    a = synthetic_sheet()
    cells = bank.slice_sheet(a, (3, 2))
    assert len(cells) == 6
    assert all(c.shape[0] == 120 and c.shape[1] == 50 for c in cells)


def test_slice_falls_back_to_equal_division_when_cells_touch():
    a = synthetic_sheet()
    a[:, :, 3] = 255                      # 전부 불투명 → 빈 간격 없음 → 기대 격자로 등분
    cells = bank.slice_sheet(a, (3, 2))
    assert len(cells) == 6


def _rows(n=36, P=12):
    """보폭 사인파 + 봉우리마다 번갈아 밝은 앞다리 → N/F 가 교대해야 한다."""
    k = np.arange(n)
    spread = 0.25 + 0.12 * np.cos(2 * np.pi * k / P)              # 봉우리 k=0,12,24
    rows = []
    for i in range(n):
        peak_idx = int(round(i / P))
        shade = 30.0 if peak_idx % 2 == 0 else -2.0               # 짝수 봉우리 = 가까운 다리 앞
        rows.append(dict(spread=float(spread[i]), together=False, lift_f=0.0, lift_b=0.0, shade=shade))
    return rows


def test_phases_alternate_NF_and_cover_circle():
    theta, lab, phi, order, smax, thr = posebank.phases_for_set(_rows())
    s = "".join(lab)
    assert set(s) <= {"N", "F"}
    runs = [s[0]] + [c for a, c in zip(s, s[1:]) if c != a]
    assert all(a != b for a, b in zip(runs, runs[1:])), s      # 교대
    assert order > 0.8                                          # 순서 검증 신호 (주기 12 코사인의 lag-1 = cos(2π/12) ≈ 0.87)
    hist = np.histogram(phi, bins=8, range=(0, 1))[0]
    assert (hist > 0).sum() >= 6                                # 8칸 중 6칸 이상 채움


def test_blend_keeps_full_alpha_where_only_one_layer_exists():
    U = np.zeros((10, 10, 4), np.uint8); L = np.zeros((10, 10, 4), np.uint8)
    U[:5] = (255, 0, 0, 255); L[5:] = (0, 0, 255, 255)          # 겹치지 않는 두 층
    w = np.full((10, 10), 0.5)
    out = assemble.blend(U, L, w)
    assert out[..., 3].min() == 255                             # 반투명 띠 없음
    assert tuple(out[0, 0, :3]) == (255, 0, 0) and tuple(out[9, 9, :3]) == (0, 0, 255)


def test_blend_mixes_where_both_exist():
    U = np.zeros((4, 4, 4), np.uint8); L = np.zeros((4, 4, 4), np.uint8)
    U[:] = (200, 0, 0, 255); L[:] = (0, 0, 200, 255)
    out = assemble.blend(U, L, np.full((4, 4), 0.5))
    assert abs(int(out[0, 0, 0]) - 100) <= 1 and abs(int(out[0, 0, 2]) - 100) <= 1


# ── 골든 회귀 ────────────────────────────────────────────────────────────
G = ROOT / "golden"


def test_golden_shapes():
    labels = json.load(open(G / "labels.json"))
    assert len(labels) == 437
    sets = {l["set"] for l in labels}
    assert sets == {"cE1", "cE2", "cW1", "cW2", "nE1", "nE2", "nE3", "cS1", "cS2", "nS1", "nS2"}
    assert all(l["facing_original"] == "west" for l in labels if l["set"].startswith("cW"))
    pose = json.load(open(G / "pose.json"))
    assert len(pose) == 293
    for sid in ("cE1", "cE2", "cW1", "cW2", "nE1", "nE2", "nE3"):
        s = "".join(pose[k]["NF"] for k in sorted(pose) if pose[k]["set"] == sid)
        runs = [s[0]] + [c for a, c in zip(s, s[1:]) if c != a]
        assert all(a != b for a, b in zip(runs, runs[1:])), (sid, s)
    pairs = json.load(open(G / "pairs8.json"))["pairs"]
    assert [p["phase"] for p in pairs] == list(range(8))
    assert len({p["clothed"] for p in pairs}) == 8 and len({p["nude"] for p in pairs}) == 8
    fp = json.load(open(G / "front_pairs8.json"))["pairs"]
    assert len(fp) == 8


@pytest.mark.skipif(not (ROOT / "assets" / "raw" / "sheets" / "cE1.png").exists(),
                    reason="원화 시트가 없으면 은행 회귀는 건너뜀 (LFS pull 후 실행)")
def test_bank_regression_against_golden(tmp_path):
    """M1 에서 bank.main 이 (sheets_dir, out_dir) 를 받게 바뀌면 이 테스트를 켠다."""
    pytest.skip("bank.main CLI 화(M1) 후 활성화")
