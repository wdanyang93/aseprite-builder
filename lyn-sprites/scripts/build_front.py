"""몸 파츠 시트(4방향)에서 정면 리그용 파츠 한 벌을 만든다.

조각 번호는 build/cut_2/CONTACT.png 를 보고 사람이 정한 것이다 (자동 추론하지 않는다).
팔·다리는 통짜로 왔으므로 관절에서 나눈다. 좌우는 거울로 만든다 — 원화에 없는 화소를
지어내지 않고, 있는 파츠를 뒤집을 뿐이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigkit.assign import default_anchor, opaque_anchor          # noqa: E402
from rigkit.split import cut_chain                                # noqa: E402

# 조각 번호 → 무엇인가 (CONTACT.png 기준)
PIECES = {"head": "raw_01", "torso": "raw_09", "arm": "raw_07",
          "leg": "raw_13", "tail": "raw_21"}

# 통짜 파츠의 관절선 (그 파츠 전체 높이에 대한 비율)
ARM_CUTS = [0.48, 0.85]          # 어깨→팔꿈치→손목→끝
LEG_CUTS = [0.47, 0.89]          # 골반→무릎→발목→발끝
OVERLAP = 10


def mirror(img: Image.Image, anchor: tuple[float, float]) -> tuple[Image.Image, tuple[float, float]]:
    return img.transpose(Image.FLIP_LEFT_RIGHT), (img.width - anchor[0], anchor[1])


def build(cut_dir: str | Path, out_dir: str | Path) -> dict:
    cut, out = Path(cut_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = {k: Image.open(cut / f"{v}.png").convert("RGBA") for k, v in PIECES.items()}

    arm = cut_chain(src["arm"], ARM_CUTS, OVERLAP)
    leg = cut_chain(src["leg"], LEG_CUTS, OVERLAP)
    right = dict(zip(("upperarm_r", "forearm_r", "hand_r"), arm))
    right |= dict(zip(("thigh_r", "shin_r", "foot_r"), leg))
    center = {"head": src["head"], "torso": src["torso"], "tail1": src["tail"]}

    parts: dict[str, dict] = {}

    def put(slot: str, img: Image.Image, anchor=None, length=None):
        anchor = anchor or opaque_anchor(img, default_anchor(slot))
        f = f"{slot}.png"
        img.save(out / f)
        parts[slot] = {"file": f, "anchor": [round(anchor[0], 2), round(anchor[1], 2)],
                       "scale": 1.0}
        if length:
            parts[slot]["length"] = round(length, 2)

    for slot, img in center.items():
        put(slot, img)
    for slot, (img, anchor, length) in right.items():
        put(slot, img, anchor, length)
        mi, ma = mirror(img, anchor)
        put(slot[:-2] + "_l", mi, ma, length)

    (out / "parts.json").write_text(json.dumps(parts, indent=1, ensure_ascii=False))
    return parts


if __name__ == "__main__":
    p = build(sys.argv[1], sys.argv[2])
    print(f"파츠 {len(p)}개 → {sys.argv[2]}/parts.json")
    print(" ".join(sorted(p)))
