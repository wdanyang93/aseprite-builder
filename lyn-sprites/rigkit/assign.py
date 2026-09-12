"""조각 → 슬롯 배정 + 앵커(관절점) 계산.

names.json 예시:
{
  "01": {"slot": "torso"},
  "04": {"slot": "thigh_l", "anchor": "top"},
  "07": {"slot": "foot_r", "anchor": [0.42, 0.12]}
}
anchor 를 안 적으면 슬롯 종류에 맞는 기본값을 쓴다. 팔다리는 '윗변 중앙',
머리는 '아래쪽 목', 몸통은 '가슴 높이', 발은 '발목'.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ANCHOR = {
    "head": (0.50, 0.78),
    "torso": (0.50, 0.06),
    "hips_wear": (0.50, 0.10),
    "backpack": (0.50, 0.20),
    "bag": (0.50, 0.15),
}
_LIMB_TOP = (0.50, 0.04)
_FOOT = (0.45, 0.12)

NAMED = {"top": _LIMB_TOP, "foot": _FOOT, "center": (0.5, 0.5)}


def default_anchor(slot: str) -> tuple[float, float]:
    if slot in DEFAULT_ANCHOR:
        return DEFAULT_ANCHOR[slot]
    if slot.startswith("foot"):
        return _FOOT
    return _LIMB_TOP


def opaque_anchor(img: Image.Image, frac: tuple[float, float]) -> tuple[float, float]:
    """비율 앵커를 '불투명 화소의 bbox' 기준으로 픽셀 좌표로 바꾼다.
    조각 PNG 가 이미 꽉 잘려 있어도, 깃털 같은 반투명 가장자리에 흔들리지 않게 한다."""
    a = np.array(img.convert("RGBA"))[..., 3] > 32
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        w, h = img.size
        return (w * frac[0], h * frac[1])
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return (x0 + (x1 - x0) * frac[0], y0 + (y1 - y0) * frac[1])


def assign(pieces_dir: str | Path, names: str | Path, out_dir: str | Path) -> dict:
    src, out = Path(pieces_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "pieces.json").read_text())
    table = json.loads(Path(names).read_text())
    parts = {}
    for key, spec in table.items():
        if key not in meta["pieces"]:
            raise KeyError(f"조각 {key} 가 pieces.json 에 없다")
        slot = spec["slot"]
        img = Image.open(src / meta["pieces"][key]["file"]).convert("RGBA")
        a = spec.get("anchor")
        frac = NAMED.get(a, a) if a is not None else default_anchor(slot)
        if isinstance(frac, (list, tuple)) and len(frac) == 2 and all(isinstance(v, (int, float)) for v in frac):
            frac = tuple(float(v) for v in frac)
        else:
            frac = default_anchor(slot)
        f = f"{slot}.png"
        img.save(out / f)
        ax, ay = opaque_anchor(img, frac)
        parts[slot] = {"file": f, "anchor": [round(ax, 2), round(ay, 2)],
                       "scale": spec.get("scale", 1.0), "from_piece": key}
    (out / "parts.json").write_text(json.dumps(parts, indent=1, ensure_ascii=False))
    return parts
