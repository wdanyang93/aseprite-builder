"""통짜 팔다리를 관절에서 나눈다.

AI 는 다리를 허벅지+종아리로 나눠 주지 않고 통짜 한 장으로 주는 일이 많다.
통짜로는 무릎을 접을 수 없으므로 여기서 나눈다. 자르는 선은 비율(t)로 지정하고,
양쪽에 겹침 여유를 남겨 회전했을 때 관절이 벌어지지 않게 한다.

화소를 늘리지 않는다 — 같은 그림을 두 장으로 나눠 가질 뿐이다.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .fit import opaque_box


def split_limb(img: Image.Image, at: float = 0.5, overlap: int = 12
               ) -> tuple[Image.Image, Image.Image, float]:
    """세로로 그려진 팔다리를 위/아래 두 장으로 나눈다.

    at      : 불투명 영역 높이에서 자를 위치 (0=위끝, 1=아래끝). 무릎은 보통 0.5 근처.
    overlap : 잘린 선 양쪽에 남길 여유(px). 회전 시 관절 틈을 메운다.
    반환     : (윗조각, 아랫조각, 자른 y 좌표(원본 기준))
    """
    x0, y0, x1, y1 = opaque_box(img)
    cut_y = y0 + (y1 - y0) * at
    top = img.crop((0, 0, img.width, min(img.height, int(cut_y + overlap))))
    bottom = img.crop((0, max(0, int(cut_y - overlap)), img.width, img.height))
    return top, bottom, cut_y


def split_to_files(src: str | Path, out_dir: str | Path, upper: str, lower: str,
                   *, at: float = 0.5, overlap: int = 12) -> dict:
    img = Image.open(src).convert("RGBA")
    top, bottom, cut_y = split_limb(img, at, overlap)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    top.save(out / f"{upper}.png")
    bottom.save(out / f"{lower}.png")
    return {upper: f"{upper}.png", lower: f"{lower}.png",
            "cut_y": round(cut_y, 1), "overlap": overlap}


def cut_chain(img: Image.Image, fracs: list[float], overlap: int = 10
              ) -> list[tuple[Image.Image, tuple[float, float], float]]:
    """통짜 팔·다리를 관절선 여러 개에서 한 번에 나눈다.

    fracs : 불투명 높이에서 자를 위치들 (예: 팔 [0.48, 0.85] → 위팔/아래팔/손).
    반환   : 조각마다 (이미지, 앵커, 길이). 앵커는 그 조각이 붙는 관절선 위에 있고,
             길이는 겹침을 뺀 '관절에서 다음 관절까지'다. 이 길이로 뼈가 만들어지므로
             조각끼리 틈이나 겹침 없이 이어진다.
    """
    import numpy as np

    x0, y0, x1, y1 = opaque_box(img)
    h = y1 - y0
    lines = [y0] + [y0 + h * f for f in fracs] + [y1]
    a = np.array(img.convert("RGBA"))[..., 3] > 32

    def center_x(y: float) -> float:
        row = int(min(max(y, y0), y1 - 1))
        xs = np.nonzero(a[row])[0]
        return float(xs.min() + xs.max()) / 2 if len(xs) else (x0 + x1) / 2

    out = []
    for i in range(len(lines) - 1):
        top, bot = lines[i], lines[i + 1]
        c0 = int(max(0, top - (overlap if i else 0)))
        c1 = int(min(img.height, bot + (overlap if i + 2 < len(lines) else 0)))
        piece = img.crop((0, c0, img.width, c1))
        out.append((piece, (center_x(top), top - c0), bot - top))
    return out
