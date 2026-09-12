"""비율 정렬 — AI 가 그려 준 파츠의 크기를 믿지 않고, 리그의 뼈 길이에 맞춘다.

AI 는 요청할 때마다 조금씩 다른 크기로 그린다(요청을 반복할수록 작아지는 경향도 있다).
그 크기를 사람이 맞추려 하면 끝이 없다. 대신 여기서 **측정해서 맞춘다**:

    scale = 슬롯 기준 길이(rig.target_length) / 파츠에서 잰 앵커→끝 길이

이러면 어느 시트에서 왔든, 몇 번째 재생성이든 파츠는 항상 같은 비율로 들어온다.
파츠 화소는 건드리지 않는다 — 균일 확대·축소뿐이고, 렌더할 때 적용된다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .rig import GROWS_UP, target_length


def opaque_box(img: Image.Image, thr: int = 32) -> tuple[int, int, int, int]:
    a = np.array(img.convert("RGBA"))[..., 3] > thr
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return (0, 0, img.width, img.height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def measured_length(img: Image.Image, anchor: tuple[float, float], slot: str,
                    explicit: float | None = None) -> float:
    """앵커에서 파츠 끝까지의 길이(픽셀). 머리만 위로, 나머지는 아래로 잰다.
    관절선에서 나눈 조각은 겹침이 섞이므로 explicit(=관절 사이 길이)를 우선한다."""
    if explicit:
        return float(explicit)
    x0, y0, x1, y1 = opaque_box(img)
    base = slot.rstrip("_lr").rstrip("_")
    if base in GROWS_UP:
        return max(1.0, anchor[1] - y0)
    return max(1.0, y1 - anchor[1])


def fit_scale(img: Image.Image, anchor: tuple[float, float], slot: str) -> float:
    return target_length(slot) / measured_length(img, anchor, slot)


def fit_parts(parts_dir: str | Path, *, write: bool = True) -> dict[str, dict]:
    """parts.json 의 모든 슬롯에 scale 을 계산해 넣는다. 이미지 파일은 건드리지 않는다."""
    d = Path(parts_dir)
    meta = json.loads((d / "parts.json").read_text())
    report = {}
    for slot, m in meta.items():
        img = Image.open(d / m["file"]).convert("RGBA")
        anchor = tuple(m["anchor"])
        before = measured_length(img, anchor, slot, m.get("length"))
        s = target_length(slot) / before
        m["scale"] = round(s, 5)
        report[slot] = {"measured": round(before, 1),
                        "target": round(target_length(slot), 1),
                        "scale": round(s, 4)}
    if write:
        (d / "parts.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return report


def sheet_scale_hint(parts_dir: str | Path) -> float:
    """여러 슬롯의 배율 중앙값. 1.0 에서 멀수록 그 시트가 통째로 작게/크게 그려진 것이고,
    슬롯마다 값이 크게 흩어지면 시트 안에서 비율이 깨진 것이다(= AI 에게 다시 요구할 근거)."""
    r = fit_parts(parts_dir, write=False)
    vals = sorted(v["scale"] for v in r.values())
    return float(np.median(vals)) if vals else 1.0


def consistency(parts_dir: str | Path) -> dict:
    """시트 안에서 파츠끼리 비율이 맞는지. 중앙값 대비 각 슬롯의 어긋남(%)."""
    r = fit_parts(parts_dir, write=False)
    med = sheet_scale_hint(parts_dir)
    out = {s: round(100 * (v["scale"] / med - 1), 1) for s, v in r.items()}
    return {"median_scale": round(med, 4), "deviation_pct": out,
            "worst": max(out.items(), key=lambda kv: abs(kv[1])) if out else None}
