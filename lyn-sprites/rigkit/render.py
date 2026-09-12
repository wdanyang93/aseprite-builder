"""리그 렌더러 — 뼈대 FK + 파츠 합성.

파츠는 통째로 회전·이동만 한다. 늘리거나 자르지 않는다 (씨앗 연구에서 row-warp 가 실패한 이유).
여기서 나오는 그림과 Godot 에서 나오는 그림은 같은 rig/walk 정의를 쓰므로 같아야 한다.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .rig import Bone, SLOT_BONE, build_skeleton, slot_order
from .spec import CANVAS


@dataclass
class Part:
    """그림 파츠 하나. anchor 는 파츠 이미지 안에서 뼈 머리와 겹치는 점(픽셀)."""
    name: str
    image: Image.Image
    anchor: tuple[float, float]
    scale: float = 1.0


def load_parts(parts_dir: str | Path) -> dict[str, Part]:
    d = Path(parts_dir)
    meta = json.loads((d / "parts.json").read_text())
    out: dict[str, Part] = {}
    for name, m in meta.items():
        img = Image.open(d / m["file"]).convert("RGBA")
        out[name] = Part(name, img, tuple(m["anchor"]), m.get("scale", 1.0))
    return out


def solve_fk(skeleton: dict[str, Bone], pose: dict[str, float]) -> dict[str, tuple[float, float, float]]:
    """각 뼈의 월드 (x, y, angle_deg). angle + 는 화면 시계방향."""
    world: dict[str, tuple[float, float, float]] = {}
    root_dx, root_dy = pose.get("root:x", 0.0), pose.get("root:y", 0.0)

    def solve(name: str):
        if name in world:
            return world[name]
        b = skeleton[name]
        ang = pose.get(name, 0.0)
        if b.parent is None:
            world[name] = (b.x + root_dx, b.y + root_dy, ang)
            return world[name]
        px, py, pa = solve(b.parent)
        pb = skeleton[b.parent]
        ox, oy = b.x - pb.x, b.y - pb.y          # 기준 자세에서의 부모 기준 오프셋
        r = math.radians(pa)
        c, s = math.cos(r), math.sin(r)
        world[name] = (px + c * ox - s * oy, py + s * ox + c * oy, pa + ang)
        return world[name]

    for n in skeleton:
        solve(n)
    return world


def draw_pose(parts: dict[str, Part], skeleton: dict[str, Bone], pose: dict[str, float],
              size: tuple[int, int] = CANVAS, bg=(0, 0, 0, 0)) -> Image.Image:
    world = solve_fk(skeleton, pose)
    canvas = Image.new("RGBA", size, bg)
    for slot in slot_order():
        part = parts.get(slot)
        if part is None:
            continue
        bone = SLOT_BONE[slot]
        if bone not in world:
            continue
        px, py, deg = world[bone]
        r = math.radians(deg)
        c, s = math.cos(r), math.sin(r)
        k = 1.0 / part.scale
        ax, ay = part.anchor
        # 출력(px,py) -> 입력(qx,qy) 역사상
        a, b = c * k, s * k
        d, e = -s * k, c * k
        cc = -a * px - b * py + ax
        ff = -d * px - e * py + ay
        layer = part.image.transform(size, Image.AFFINE, (a, b, cc, d, e, ff),
                                     resample=Image.BICUBIC)
        canvas = Image.alpha_composite(canvas, layer)
    return canvas


def render_cycle(parts, skeleton, poses, **kw) -> list[Image.Image]:
    return [draw_pose(parts, skeleton, p, **kw) for p in poses]


def strip(frames: list[Image.Image], cols: int | None = None, bg=(255, 255, 255, 255)) -> Image.Image:
    cols = cols or len(frames)
    rows = (len(frames) + cols - 1) // cols
    w, h = frames[0].size
    out = Image.new("RGBA", (w * cols, h * rows), bg)
    for i, f in enumerate(frames):
        out.alpha_composite(f, ((i % cols) * w, (i // cols) * h))
    return out


def save_gif(frames: list[Image.Image], path, fps: int = 12, bg=(255, 255, 255)):
    flat = []
    for f in frames:
        b = Image.new("RGB", f.size, bg)
        b.paste(f, mask=f.split()[3])
        flat.append(b.convert("P", palette=Image.ADAPTIVE))
    flat[0].save(path, save_all=True, append_images=flat[1:],
                 duration=int(1000 / fps), loop=0, disposal=2)
