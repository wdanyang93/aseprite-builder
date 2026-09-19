"""Deterministic measurement of a generated frame against the request spec.

The numbers reported here are the ones the REVIEW.md tables are built from, so
they have to match the pipeline's own QA exactly: same rules, same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .ccl import component_areas
from .spec import Spec


@dataclass(frozen=True)
class Check:
    name: str
    required: str
    measured: str
    ok: bool


@dataclass(frozen=True)
class Report:
    path: Path
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(check.ok for check in self.checks)

    def as_markdown(self) -> str:
        head = f"| 항목 | 기준 | 잰 값 | 판정 |\n|---|---:|---:|---|"
        rows = [
            f"| {c.name} | {c.required} | {c.measured} | {'PASS' if c.ok else 'FAIL'} |"
            for c in self.checks
        ]
        return "\n".join([head, *rows])


def load_rgba(path: str | Path) -> np.ndarray:
    """Read a PNG as an (h, w, 4) uint8 array."""
    with Image.open(path) as img:
        return np.array(img.convert("RGBA"), dtype=np.uint8)


def subject_bbox(alpha: np.ndarray, threshold: int = 0) -> tuple[int, int, int, int] | None:
    """Bounding box `(left, top, right, bottom)` of pixels above `threshold`."""
    rows = np.flatnonzero(alpha.any(axis=1) if threshold == 0 else (alpha > threshold).any(axis=1))
    cols = np.flatnonzero(alpha.any(axis=0) if threshold == 0 else (alpha > threshold).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def measure(path: str | Path, spec: Spec) -> Report:
    path = Path(path)
    rgba = load_rgba(path)
    height, width = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    total = height * width

    checks: list[Check] = []

    checks.append(
        Check(
            "캔버스",
            f"{spec.width} × {spec.height}",
            f"{width} × {height}",
            (width, height) == spec.canvas,
        )
    )

    partial = int(np.count_nonzero((alpha > 0) & (alpha < spec.opaque_alpha)))
    partial_pct = partial / total * 100
    checks.append(
        Check(
            "부분 알파",
            f"≤ {spec.partial_alpha_max_pct}%",
            f"{partial:,}px ({partial_pct:.4f}%)",
            partial_pct <= spec.partial_alpha_max_pct,
        )
    )

    if spec.transparent_rgb_must_be_zero:
        nonzero = int(np.count_nonzero((alpha == 0) & rgb.any(axis=2)))
        checks.append(
            Check("투명부 비영 RGB", "0px", f"{nonzero:,}px", nonzero == 0)
        )

    pure_red = int(
        np.count_nonzero(
            (alpha > 0)
            & (rgb[:, :, 0] == 255)
            & (rgb[:, :, 1] == 0)
            & (rgb[:, :, 2] == 0)
        )
    )
    checks.append(
        Check(
            "순수 빨강 RGB(255,0,0)",
            f"≤ {spec.pure_red_max}px",
            f"{pure_red:,}px",
            pure_red <= spec.pure_red_max,
        )
    )

    _, areas = component_areas(alpha > 0)
    specks = int(np.count_nonzero(areas <= spec.speck_max_area))
    checks.append(
        Check(
            f"티끌 덩어리 (≤{spec.speck_max_area}px)",
            f"≤ {spec.speck_components_max}개",
            f"{specks}개 / 전체 {len(areas)}개",
            specks <= spec.speck_components_max,
        )
    )

    box = subject_bbox(alpha)
    if box is None:
        checks.append(Check("피사체", "존재", "없음 (빈 캔버스)", False))
        return Report(path, tuple(checks))

    left, top, right, bottom = box
    subject_height = bottom - top
    checks.append(
        Check(
            "피사체 높이",
            f"≥ {spec.subject_height_px_min}px",
            f"{subject_height}px",
            subject_height >= spec.subject_height_px_min,
        )
    )

    if spec.subject_must_not_touch_border:
        touching = left == 0 or top == 0 or right == width or bottom == height
        checks.append(
            Check(
                "캔버스 경계 접촉",
                "없음",
                f"bbox [{left}, {top}, {right}, {bottom}]" + (" — 접촉" if touching else " — 여백 확보"),
                not touching,
            )
        )

    return Report(path, tuple(checks))
