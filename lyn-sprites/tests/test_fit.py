"""비율 정렬 — AI 가 어떤 크기로 그려 오든 리그에 맞아야 한다."""
import numpy as np
import pytest
from PIL import Image

from rigkit.fit import consistency, fit_parts, fit_scale, measured_length
from rigkit.placeholder import build as build_placeholder
from rigkit.rig import target_length
from rigkit.split import split_limb


@pytest.mark.parametrize("factor", [0.31, 0.62, 1.0, 1.7, 3.4])
@pytest.mark.parametrize("slot", ["thigh_r", "torso", "head", "foot_r"])
def test_any_source_size_lands_on_the_rig(tmp_path, factor, slot):
    """같은 파츠를 3배 크게/3배 작게 받아도 배율을 걸면 뼈 길이에 정확히 맞는다."""
    build_placeholder(tmp_path, "east")
    img = Image.open(tmp_path / f"{slot}.png").convert("RGBA")
    import json
    anchor = tuple(json.loads((tmp_path / "parts.json").read_text())[slot]["anchor"])

    big = img.resize((max(1, int(img.width * factor)), max(1, int(img.height * factor))),
                     Image.LANCZOS)
    a2 = (anchor[0] * factor, anchor[1] * factor)
    s = fit_scale(big, a2, slot)
    effective = measured_length(big, a2, slot) * s
    assert effective == pytest.approx(target_length(slot), rel=0.02)


def test_fit_writes_scale_into_parts_json(tmp_path):
    build_placeholder(tmp_path, "east")
    import json
    r = fit_parts(tmp_path, write=True)
    meta = json.loads((tmp_path / "parts.json").read_text())
    for slot, v in r.items():
        assert meta[slot]["scale"] == pytest.approx(v["scale"], abs=1e-3)


def test_check_flags_a_single_wrong_sized_part(tmp_path):
    build_placeholder(tmp_path, "east")
    # 한 파츠만 60% 크기로 다시 그려 온 상황
    img = Image.open(tmp_path / "shin_r.png")
    img.resize((int(img.width * 0.6), int(img.height * 0.6))).save(tmp_path / "shin_r.png")
    c = consistency(tmp_path)
    assert abs(c["deviation_pct"]["shin_r"]) > 30
    assert abs(c["deviation_pct"]["thigh_r"]) < 10      # 나머지는 멀쩡하다


def test_split_keeps_every_pixel_and_overlaps(tmp_path):
    build_placeholder(tmp_path, "east")
    img = Image.open(tmp_path / "thigh_r.png").convert("RGBA")
    top, bottom, cut_y = split_limb(img, at=0.5, overlap=12)
    src = (np.array(img)[..., 3] > 128).sum()
    got = (np.array(top)[..., 3] > 128).sum() + (np.array(bottom)[..., 3] > 128).sum()
    assert got >= src                         # 겹침만큼 더 많고, 잃은 화소는 없다
    assert top.height + bottom.height == img.height + 2 * 12
