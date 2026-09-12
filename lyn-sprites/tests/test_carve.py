"""서 있는 전신 → 관절 높이 자르기."""
import numpy as np
import pytest
from PIL import Image

from rigkit.carve import carve, normalize
from rigkit.placeholder import build as build_placeholder
from rigkit.render import draw_pose, load_parts
from rigkit.rig import build_skeleton
from rigkit.spec import BODY_H, CANVAS, CENTER_X, SOLE_Y
from rigkit.walk import idle_pose


@pytest.fixture(scope="module")
def standing(tmp_path_factory):
    """차렷 자세 전신 한 장을 만들어 둔다 (진짜 원화 대역)."""
    d = tmp_path_factory.mktemp("ph")
    build_placeholder(d, "east")
    img = draw_pose(load_parts(d), build_skeleton("east"), idle_pose("east"))
    f = d / "body.png"
    img.save(f)
    return f


def test_normalize_puts_body_on_the_convention(standing):
    # 일부러 작게·치우치게 만든 뒤 정규화하면 규약 위치로 돌아와야 한다
    src = Image.open(standing).convert("RGBA")
    small = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    small.alpha_composite(src.resize((src.width // 3, src.height // 3), Image.LANCZOS), (11, 7))
    out = normalize(small)
    a = np.array(out)[..., 3] > 32
    ys, xs = np.nonzero(a)
    assert ys.max() + 1 == pytest.approx(SOLE_Y, abs=3)
    assert (ys.max() + 1 - ys.min()) == pytest.approx(BODY_H, abs=3)
    assert (xs.min() + xs.max()) / 2 == pytest.approx(CENTER_X, abs=4)


def test_carve_makes_the_expected_slots(tmp_path, standing):
    parts = carve(standing, tmp_path)
    assert set(parts) == {"head", "torso", "thigh_r", "shin_r", "foot_r"}
    for slot, m in parts.items():
        img = Image.open(tmp_path / m["file"]).convert("RGBA")
        assert (np.array(img)[..., 3] > 32).any(), f"{slot} 이 비었다"


def test_split_legs_gives_left_and_right(tmp_path, standing):
    parts = carve(standing, tmp_path, split_legs=True)
    for base in ("thigh", "shin", "foot"):
        assert f"{base}_l" in parts and f"{base}_r" in parts


def test_carve_loses_no_body_pixels(tmp_path, standing):
    """밴드로 나눈 조각들의 화소 합은 원본 이상이어야 한다 (겹침만큼 더 많다)."""
    parts = carve(standing, tmp_path, overlap=8)
    whole = (np.array(normalize(Image.open(standing).convert("RGBA")))[..., 3] > 32).sum()
    got = sum((np.array(Image.open(tmp_path / m["file"]).convert("RGBA"))[..., 3] > 32).sum()
              for m in parts.values())
    assert got >= whole * 0.98
