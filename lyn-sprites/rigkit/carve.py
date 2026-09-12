"""서 있는 전신 그림 → 관절 높이에서 잘라 파츠로.

8방향 턴어라운드처럼 '차렷 자세 전신' 이 방향별로 있으면, 파츠를 따로 그려 받을 필요가 없다.
자세가 곧기 때문에 관절 높이(JOINT_T)에서 가로로 자르면 그게 곧 파츠다.

한계 — 지어내지 않고 그대로 적는다:
- 팔은 가로 자르기로 몸통과 분리되지 않는다. 팔은 파츠 시트에서 따로 받아야 한다.
- 두 다리는 정면·후면에서만 세로로 나눌 수 있다(--split-legs). 측면은 겹쳐 있어 불가능하다.
- 잘린 단면은 원화에 없던 면이다. 회전하면 틈이 보일 수 있어 overlap 으로 여유를 남긴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .fit import opaque_box
from .spec import BODY_H, CANVAS, CENTER_X, JOINT_T, SOLE_Y, t_to_y

def bands(pad: float = 0.02) -> list[tuple[str, float, float, float]]:
    """(슬롯, 시작 t, 끝 t, 앵커 t). 관절 표가 캐릭터에 맞게 바뀌면 밴드도 따라간다."""
    j = JOINT_T
    return [
        ("head",  0.0,                    j["head"] + pad,   j["head"]),
        ("torso", j["head"] - pad,         j["hips"] + pad,   j["chest"]),
        ("thigh", j["hips"] - pad,         j["knee"] + pad,   j["hips"]),
        ("shin",  j["knee"] - pad,         j["ankle"] + pad,  j["knee"]),
        ("foot",  j["ankle"] - pad * 0.5,  1.0,               j["ankle"]),
    ]


def normalize(img: Image.Image, *, body_h: int = BODY_H) -> Image.Image:
    """전신 그림을 규약 캔버스로: 몸 높이 = BODY_H, 발바닥 = SOLE_Y, 중심선 = CENTER_X.
    균일 배율이므로 비율은 변하지 않는다."""
    x0, y0, x1, y1 = opaque_box(img)
    h = max(1, y1 - y0)
    s = body_h / h
    w2, h2 = max(1, int(round(img.width * s))), max(1, int(round(img.height * s)))
    small = img.convert("RGBA").resize((w2, h2), Image.LANCZOS)
    # 원본 bbox 의 발바닥·중심을 캔버스 기준점에 맞춘다
    cx = (x0 + x1) / 2 * s
    sole = y1 * s
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    canvas.alpha_composite(small, (int(round(CENTER_X - cx)), int(round(SOLE_Y - sole))))
    return canvas


def _anchor_x(band: Image.Image, row: int) -> float:
    """그 높이에서 불투명 화소의 가운데 — 관절의 좌우 위치."""
    a = np.array(band)[..., 3] > 32
    row = min(max(row, 0), a.shape[0] - 1)
    xs = np.nonzero(a[row])[0]
    if len(xs) == 0:
        ys = np.nonzero(a.any(axis=1))[0]
        if len(ys) == 0:
            return band.width / 2
        xs = np.nonzero(a[ys[len(ys) // 2]])[0]
    return float(xs.min() + xs.max()) / 2


def carve(body: str | Path, out_dir: str | Path, *, split_legs: bool = False,
          overlap: int = 8, suffix: str = "") -> dict:
    """전신 한 장 → 파츠 PNG + parts.json."""
    img = normalize(Image.open(body).convert("RGBA"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    parts: dict[str, dict] = {}

    for name, t0, t1, ta in bands():
        y0 = int(round(t_to_y(t0))) - (0 if name == "head" else overlap)
        y1 = int(round(t_to_y(t1))) + overlap
        band = img.crop((0, max(0, y0), img.width, min(img.height, y1)))
        anchor_row = int(round(t_to_y(ta))) - max(0, y0)
        ax = _anchor_x(band, anchor_row)

        targets = [(name, band, ax)]
        if split_legs and name in ("thigh", "shin", "foot"):
            left = band.crop((0, 0, int(CENTER_X), band.height))
            right = band.crop((int(CENTER_X), 0, band.width, band.height))
            targets = [(f"{name}_l", left, _anchor_x(left, anchor_row)),
                       (f"{name}_r", right, _anchor_x(right, anchor_row))]
        elif name in ("thigh", "shin", "foot"):
            targets = [(f"{name}_r", band, ax)]     # 측면: 다리 한 벌만 쓴다

        for slot, piece, anchor_x in targets:
            slot = slot + suffix
            f = f"{slot}.png"
            piece.save(out / f)
            parts[slot] = {"file": f, "anchor": [round(anchor_x, 2), round(anchor_row, 2)],
                           "scale": 1.0, "from": "carve"}
    (out / "parts.json").write_text(json.dumps(parts, indent=1, ensure_ascii=False))
    return parts
