"""리그 불변식 — '말로만 됐다'를 막는 수치 검증."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from rigkit.assign import opaque_anchor
from rigkit.cut import cut, to_alpha
from rigkit.placeholder import build as build_placeholder
from rigkit.render import draw_pose, load_parts, solve_fk
from rigkit.rig import build_skeleton, slot_order
from rigkit.spec import CENTER_X
from rigkit.walk import idle_pose, walk_cycle, walk_pose


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    d = tmp_path_factory.mktemp("parts")
    build_placeholder(d, "east")
    return load_parts(d)


def test_rest_pose_puts_bones_at_rest_positions():
    sk = build_skeleton("east")
    w = solve_fk(sk, idle_pose("east"))
    for name, b in sk.items():
        assert w[name][0] == pytest.approx(b.x, abs=1e-6)
        assert w[name][1] == pytest.approx(b.y, abs=1e-6)
        assert w[name][2] == pytest.approx(0.0, abs=1e-9)


def test_walk_alternates_legs_and_is_half_cycle_symmetric():
    sk = build_skeleton("east")
    a = solve_fk(sk, walk_pose(0.0, side="east"))
    b = solve_fk(sk, walk_pose(0.5, side="east"))
    # phi 0 = 오른발이 앞(화면 오른쪽), 왼발이 뒤
    assert a["foot_r"][0] > CENTER_X > a["foot_l"][0]
    # 반 주기 뒤에는 역할이 완전히 뒤바뀐다
    assert b["foot_l"][0] > CENTER_X > b["foot_r"][0]


def test_half_cycle_swaps_left_and_right_exactly():
    """반 주기 뒤 왼쪽 뼈의 각도 = 지금 오른쪽 뼈의 각도. (세계좌표는 골반 좌우 오프셋과
    허리 트위스트 때문에 정확히 겹치지 않으므로 각도로 본다.)"""
    for i in range(16):
        phi = i / 16
        a, b = walk_pose(phi, side="east"), walk_pose(phi + 0.5, side="east")
        for r, l in (("thigh_r", "thigh_l"), ("shin_r", "shin_l"),
                     ("foot_r", "foot_l"), ("upperarm_r", "upperarm_l"),
                     ("forearm_r", "forearm_l")):
            assert a[r] == pytest.approx(b[l], abs=1e-9)


def test_knee_never_bends_forward():
    for p in walk_cycle(32, side="east"):
        assert p["shin_r"] >= -1e-9 and p["shin_l"] >= -1e-9   # 렌더 규약에서 뒤로 = +


def test_body_is_lowest_at_contact():
    ys = [walk_pose(i / 32, side="east")["root:y"] for i in range(32)]
    assert ys[0] == pytest.approx(max(ys), abs=1e-6)      # phi 0 = 착지 = 가장 낮다(+y)
    assert ys[8] == pytest.approx(min(ys), abs=1e-6)      # phi .25 = 중간 지지 = 가장 높다


def test_stride_is_a_plausible_fraction_of_height():
    sk = build_skeleton("east")
    w = solve_fk(sk, walk_pose(0.0, side="east"))
    from rigkit.spec import BODY_H
    stride = abs(w["foot_r"][0] - w["foot_l"][0])
    assert 0.2 * BODY_H < stride < 0.40 * BODY_H


def test_parts_are_moved_whole_not_stretched(parts):
    """리깅의 절대 규칙: 파츠는 통째로 회전·이동만 한다. 화소 수가 보존돼야 한다."""
    sk = build_skeleton("east")
    for pose in (idle_pose("east"), walk_pose(0.3, side="east")):
        for slot in ("thigh_r", "torso", "foot_r"):
            single = {slot: parts[slot]}
            img = draw_pose(single, sk, pose)
            drawn = (np.array(img)[..., 3] > 128).sum()
            src = (np.array(parts[slot].image)[..., 3] > 128).sum()
            assert drawn == pytest.approx(src, rel=0.06)


def test_every_slot_has_a_known_bone():
    sk = build_skeleton("east")
    from rigkit.rig import SLOT_BONE
    for s in slot_order():
        assert SLOT_BONE[s] in sk


def test_cut_recovers_every_island(tmp_path):
    sheet = Image.new("RGB", (700, 400), (255, 255, 255))
    boxes = [(30, 30, 120, 200), (200, 40, 80, 150), (330, 35, 60, 60), (450, 30, 140, 180)]
    for i, (x, y, w, h) in enumerate(boxes):
        tile = Image.new("RGBA", (w, h), (250, 250, 248, 255) if i % 2 else (150, 100, 62, 255))
        Image.Image.paste(sheet, tile, (x, y), tile)
    # 흰 조각도 윤곽선이 있으면 배경과 구분된다
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for x, y, w, h in boxes:
        d.rectangle([x, y, x + w - 1, y + h - 1], outline=(60, 48, 42), width=3)
    f = tmp_path / "sheet.png"
    sheet.save(f)
    meta = cut(f, tmp_path / "pieces", min_area=400)
    assert len(meta["pieces"]) == len(boxes)


def test_white_garment_survives_background_removal(tmp_path):
    """후드가 흰색이라도 배경으로 지워지면 안 된다 (테두리와 이어진 흰색만 배경)."""
    sheet = Image.new("RGB", (300, 300), (255, 255, 255))
    from PIL import ImageDraw
    ImageDraw.Draw(sheet).ellipse([80, 80, 220, 220], fill=(255, 255, 255), outline=(60, 48, 42), width=4)
    im = to_alpha(sheet)
    a = np.array(im)[..., 3]
    assert a[150, 150] == 255          # 옷 안쪽 흰색은 남는다
    assert a[5, 5] == 0                # 바깥 배경은 지워진다


def test_anchor_is_measured_on_opaque_pixels():
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (20, 40, 60, 90))
    ax, ay = opaque_anchor(img, (0.5, 0.0))
    assert (ax, ay) == pytest.approx((40.0, 40.0))
