"""Godot 4 내보내기 — Skeleton2D + Bone2D + Sprite2D 컷아웃 리그와 걷기 애니메이션.

파이썬 미리보기(render.py)와 같은 rig/walk 정의에서 나오므로 두 결과는 같아야 한다.
Godot 쪽 기존 동작(Background, Dig, Room, Light, Walk)은 건드리지 않는다 — 이 씬은 새로 추가되는 것이다.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .rig import SLOT_BONE, build_skeleton, slot_order
from .spec import CANVAS, WALK_FRAMES
from .walk import walk_cycle

TEX_ROOT = "res://assets/parts"


def _chain(sk, name: str) -> str:
    parts = []
    while name is not None:
        parts.append(name)
        name = sk[name].parent
    return "Skeleton2D/" + "/".join(reversed(parts))


def _bone_len_angle(sk, name: str) -> tuple[float, float]:
    kids = [b for b in sk.values() if b.parent == name]
    if not kids:
        return 24.0, 90.0
    k = kids[0]
    dx, dy = k.x - sk[name].x, k.y - sk[name].y
    return math.hypot(dx, dy) or 24.0, math.degrees(math.atan2(dy, dx))


def export_scene(parts_json: str | Path, out_tscn: str | Path, *, side: str = "east",
                 fps: int = 12, frames: int = WALK_FRAMES, tex_root: str = TEX_ROOT) -> Path:
    parts = json.loads(Path(parts_json).read_text())
    sk = build_skeleton(side)
    slots = [s for s in slot_order() if s in parts]

    ext, lines = [], []
    for i, s in enumerate(slots, start=1):
        ext.append(f'[ext_resource type="Texture2D" '
                   f'path="{tex_root}/{parts[s]["file"]}" id="{i}_{s}"]')

    # ── 애니메이션 트랙
    poses = walk_cycle(frames, side=side)
    times = [round(i / fps, 4) for i in range(frames)] + [round(frames / fps, 4)]
    tracks, ti = [], 0

    def add_track(path: str, values: list[str], kind: str):
        nonlocal ti
        vals = values + [values[0]]                     # 루프 닫기
        tracks.append(
            f'tracks/{ti}/type = "{kind}"\n'
            f'tracks/{ti}/imported = false\n'
            f'tracks/{ti}/enabled = true\n'
            f'tracks/{ti}/path = NodePath("{path}")\n'
            f'tracks/{ti}/interp = 1\n'
            f'tracks/{ti}/loop_wrap = true\n'
            f'tracks/{ti}/keys = {{\n'
            f'"times": PackedFloat32Array({", ".join(str(t) for t in times)}),\n'
            f'"transitions": PackedFloat32Array({", ".join("1" for _ in times)}),\n'
            f'"update": 0,\n'
            f'"values": [{", ".join(vals)}]\n'
            f'}}')
        ti += 1

    rootb = sk["root"]
    add_track(_chain(sk, "root") + ":position",
              [f"Vector2({rootb.x + p.get('root:x', 0):.2f}, {rootb.y + p.get('root:y', 0):.2f})"
               for p in poses], "value")
    for name in sk:
        if all(abs(p.get(name, 0.0)) < 1e-6 for p in poses):
            continue
        add_track(_chain(sk, name) + ":rotation",
                  [f"{math.radians(p.get(name, 0.0)):.5f}" for p in poses], "value")

    anim = (f'[sub_resource type="Animation" id="Animation_walk"]\n'
            f'resource_name = "walk"\n'
            f'length = {frames / fps:.4f}\n'
            f'loop_mode = 1\n' + "\n".join(tracks))
    animlib = ('[sub_resource type="AnimationLibrary" id="AnimationLibrary_lyn"]\n'
               '_data = {\n"walk": SubResource("Animation_walk")\n}')

    # ── 노드
    lines.append('[node name="Lyn" type="Node2D"]')
    lines.append('\n[node name="Skeleton2D" type="Skeleton2D" parent="."]')
    for name, b in sk.items():
        parent_path = _chain(sk, b.parent) if b.parent else "Skeleton2D"
        px, py = (b.x, b.y) if b.parent is None else (b.x - sk[b.parent].x, b.y - sk[b.parent].y)
        ln, ang = _bone_len_angle(sk, name)
        lines.append(
            f'\n[node name="{name}" type="Bone2D" parent="{parent_path}"]\n'
            f'position = Vector2({px:.2f}, {py:.2f})\n'
            f'rest = Transform2D(1, 0, 0, 1, {px:.2f}, {py:.2f})\n'
            f'auto_calculate_length_and_angle = false\n'
            f'length = {ln:.2f}\n'
            f'bone_angle = {math.radians(ang):.5f}')
    for i, s in enumerate(slots, start=1):
        ax, ay = parts[s]["anchor"]
        sc = parts[s].get("scale", 1.0)
        lines.append(
            f'\n[node name="{s}" type="Sprite2D" parent="{_chain(sk, SLOT_BONE[s])}"]\n'
            f'z_index = {_z(s)}\n'
            f'texture = ExtResource("{i}_{s}")\n'
            f'centered = false\n'
            f'offset = Vector2({-ax:.2f}, {-ay:.2f})'
            + (f'\nscale = Vector2({sc}, {sc})' if sc != 1.0 else ''))
    lines.append('\n[node name="AnimationPlayer" type="AnimationPlayer" parent="."]\n'
                 'libraries = {\n"": SubResource("AnimationLibrary_lyn")\n}\n'
                 'autoplay = "walk"')

    load_steps = len(ext) + 3
    text = (f'[gd_scene load_steps={load_steps} format=3]\n\n'
            + "\n".join(ext) + "\n\n" + anim + "\n\n" + animlib + "\n\n"
            + "\n".join(lines) + "\n")
    out = Path(out_tscn); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def _z(slot: str) -> int:
    from .rig import SLOT_Z
    return SLOT_Z[slot]
