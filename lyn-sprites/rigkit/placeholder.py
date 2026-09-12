"""검증용 임시 파츠 — 진짜 원화가 오기 전에 뼈대가 제대로 도는지 보기 위한 것.
실제 파츠(assets/parts/)가 생기면 이 모듈은 쓰지 않는다."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .rig import build_skeleton, target_length
from .spec import JOINT_T, t_to_y

C = {
    "skin":   (238, 205, 184, 255),
    "hair":   (232, 228, 222, 255),
    "shirt":  (243, 237, 226, 255),
    "cloak":  (88, 99, 64, 255),
    "leather":(150, 100, 62, 255),
    "tights": (74, 62, 58, 255),
    "boot":   (169, 118, 74, 255),
    "fur":    (240, 236, 228, 255),
}


def _limb(w: int, h: int, color, pad: int = 6) -> tuple[Image.Image, tuple[float, float]]:
    """위쪽 중앙이 관절(anchor)인 둥근 막대."""
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([pad, pad, pad + w, pad + h], radius=w // 2, fill=color,
                        outline=(60, 48, 42, 255), width=2)
    return im, (pad + w / 2, pad + 2.0)


def _blob(w: int, h: int, color, anchor_t: float = 0.0) -> tuple[Image.Image, tuple[float, float]]:
    pad = 8
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([pad, pad, pad + w, pad + h], fill=color, outline=(60, 48, 42, 255), width=2)
    return im, (pad + w / 2, pad + h * anchor_t)


def build(out_dir: str | Path, side: str = "east") -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sk = build_skeleton(side)

    def seg(a: str, b: str) -> int:
        return int(round(abs(t_to_y(JOINT_T[b]) - t_to_y(JOINT_T[a]))))

    ua, fa = seg("shoulder", "elbow"), seg("elbow", "wrist")
    th, sh = seg("hips", "knee"), seg("knee", "ankle")

    specs: dict[str, tuple[Image.Image, tuple[float, float]]] = {}
    for s in ("l", "r"):
        sleeve = C["shirt"]
        specs[f"upperarm_{s}"] = _limb(30, ua, sleeve)
        specs[f"forearm_{s}"] = _limb(24, fa, C["skin"])
        specs[f"hand_{s}"] = _limb(24, int(round(target_length("hand"))), C["leather"])
        specs[f"thigh_{s}"] = _limb(46, th, C["tights"])
        specs[f"shin_{s}"] = _limb(36, sh, C["tights"])
        # 부츠: 앵커(발목)에서 바닥까지가 기준 길이여야 한다
        fh = int(round(target_length("foot") / 0.88))
        foot = Image.new("RGBA", (int(fh * 1.6), fh + 10), (0, 0, 0, 0))
        ImageDraw.Draw(foot).rounded_rectangle([6, 6, int(fh * 1.6) - 14, fh + 4], radius=10,
                                               fill=C["boot"], outline=(60, 48, 42, 255), width=2)
        specs[f"foot_{s}"] = (foot, (fh * 0.55, (fh + 10) * 0.12))

    specs["torso"] = _limb(108, int(round(target_length("torso"))), C["shirt"], pad=10)
    specs["hips_wear"] = _limb(96, int(round(target_length("hips_wear"))), C["cloak"], pad=10)
    head_h = int(round(target_length("head") / 0.90))      # 앵커(목)가 아래 90% 지점
    specs["head"] = _blob(int(head_h * 0.95), head_h, C["hair"], anchor_t=0.90)
    for i, (w, h) in enumerate(((54, 0), (48, 0), (40, 0)), start=1):
        specs[f"tail{i}"] = _limb(w, int(round(target_length(f"tail{i}"))), C["fur"])

    meta = {}
    for name, (img, anchor) in specs.items():
        f = f"{name}.png"
        img.save(out / f)
        meta[name] = {"file": f, "anchor": [round(anchor[0], 2), round(anchor[1], 2)], "scale": 1.0}
    (out / "parts.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return out / "parts.json"
