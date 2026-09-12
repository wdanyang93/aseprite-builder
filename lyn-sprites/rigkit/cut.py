"""원화 파츠 시트 → 개별 파츠 PNG.

AI 가 흰 배경에 파츠를 늘어놓은 시트(망토·레깅스·장갑·부츠 …)를 받아
1) 테두리에서 번진 흰 배경만 지우고 (옷 자체의 흰색은 남긴다),
2) 남은 덩어리를 연결성분으로 분리해 조각 PNG 로 저장하고,
3) 번호를 붙인 대조 시트를 만들어 사람이 슬롯 이름을 정하게 한다.

자동 이름 붙이기는 하지 않는다. 씨앗 연구에서 '색·성분으로 파츠를 알아맞히기'가
16번 실패한 것이 그 이유다 (docs/archive/docs/research/failures.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes, label as cc_label


def background_mask(rgb: np.ndarray, tol: int = 18) -> np.ndarray:
    """테두리와 이어진 '거의 흰색' 화소만 배경으로 본다.
    옷의 흰색(후드·셔츠)은 윤곽선으로 둘러싸여 테두리와 이어지지 않으므로 살아남는다."""
    h, w, _ = rgb.shape
    near_white = (rgb.min(axis=2) >= 255 - tol)
    lab, n = cc_label(near_white)
    border = set(np.unique(np.r_[lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    border.discard(0)
    return np.isin(lab, list(border)) if border else np.zeros((h, w), bool)


def to_alpha(img: Image.Image, tol: int = 18) -> Image.Image:
    """알파가 없는(흰 배경) 시트에 알파를 만든다. 이미 알파가 있으면 그대로 쓴다."""
    im = img.convert("RGBA")
    a = np.array(im)
    if (a[..., 3] < 250).mean() > 0.02:      # 이미 투명 배경
        return im
    bg = background_mask(a[..., :3], tol)
    fg = binary_fill_holes(binary_closing(~bg, np.ones((3, 3), bool)))
    a[..., 3] = np.where(fg, 255, 0)
    return Image.fromarray(a)


def islands(im: Image.Image, min_area: int = 400, gap: int = 5) -> list[tuple[int, int, int, int]]:
    """파츠 덩어리 bbox 목록 (x0, y0, x1, y1). gap 만큼 닫아서 끈·버클이 본체와 붙게 한다."""
    a = np.array(im)[..., 3] > 8
    a = binary_closing(a, np.ones((gap, gap), bool))
    lab, n = cc_label(a)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < min_area:
            continue
        out.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    # 위→아래, 왼→오른 순서 (행 단위로 묶어서)
    out.sort(key=lambda b: (b[1] // 120, b[0]))
    return out


def cut(sheet: str | Path, out_dir: str | Path, *, min_area: int = 400,
        tol: int = 18, gap: int = 5) -> dict:
    sheet, out = Path(sheet), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    im = to_alpha(Image.open(sheet), tol)
    boxes = islands(im, min_area, gap)
    pieces = {}
    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        key = f"{i:02d}"
        crop = im.crop((x0, y0, x1, y1))
        crop.save(out / f"raw_{key}.png")
        pieces[key] = {"file": f"raw_{key}.png", "sheet_box": [x0, y0, x1, y1],
                       "size": [x1 - x0, y1 - y0]}
    meta = {"sheet": sheet.name, "sheet_size": list(im.size), "pieces": pieces}
    (out / "pieces.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    contact_sheet(im, boxes).save(out / "CONTACT.png")
    return meta


def contact_sheet(im: Image.Image, boxes) -> Image.Image:
    """번호가 찍힌 대조 시트 — 이걸 보고 names.json 을 적는다."""
    base = Image.new("RGBA", im.size, (255, 255, 255, 255))
    base.alpha_composite(im)
    d = ImageDraw.Draw(base)
    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(220, 40, 40, 255), width=3)
        d.rectangle([x0, y0, x0 + 54, y0 + 34], fill=(220, 40, 40, 255))
        d.text((x0 + 12, y0 + 9), f"{i:02d}", fill=(255, 255, 255, 255))
    return base
