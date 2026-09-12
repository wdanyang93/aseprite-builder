import math
import re

from rigkit.godot import export_scene
from rigkit.placeholder import build as build_placeholder
from rigkit.rig import build_skeleton, slot_order
from rigkit.walk import walk_cycle


def test_scene_has_every_bone_and_slot(tmp_path):
    pd = tmp_path / "parts"
    build_placeholder(pd, "east")
    out = export_scene(pd / "parts.json", tmp_path / "lyn.tscn", side="east")
    text = out.read_text()
    for name in build_skeleton("east"):
        assert f'[node name="{name}" type="Bone2D"' in text
    import json
    have = json.loads((pd / "parts.json").read_text())
    for slot in slot_order():
        present = f'[node name="{slot}" type="Sprite2D"' in text
        assert present == (slot in have), f"{slot}: 파츠가 있으면 노드도 있어야 한다"
    assert text.startswith("[gd_scene load_steps=")
    assert 'autoplay = "walk"' in text


def test_animation_values_match_python_walk(tmp_path):
    pd = tmp_path / "parts"
    build_placeholder(pd, "east")
    out = export_scene(pd / "parts.json", tmp_path / "lyn.tscn", side="east", frames=8, fps=12)
    text = out.read_text()
    block = re.search(r'path = NodePath\("[^"]*thigh_r:rotation"\).*?"values": \[([^\]]*)\]',
                      text, re.S).group(1)
    got = [float(v) for v in block.split(",")]
    want = [math.radians(p["thigh_r"]) for p in walk_cycle(8, side="east")]
    assert got[:-1] == [round(w, 5) for w in want]
    assert got[-1] == got[0]        # 루프가 닫혀 있다


def test_sprite_offset_cancels_anchor(tmp_path):
    import json
    pd = tmp_path / "parts"
    build_placeholder(pd, "east")
    parts = json.loads((pd / "parts.json").read_text())
    text = export_scene(pd / "parts.json", tmp_path / "lyn.tscn", side="east").read_text()
    block = re.search(r'\[node name="thigh_r" type="Sprite2D".*?offset = Vector2\(([-\d.]+), ([-\d.]+)\)',
                      text, re.S)
    ax, ay = parts["thigh_r"]["anchor"]
    assert (float(block.group(1)), float(block.group(2))) == (-ax, -ay)
    assert "centered = false" in text
