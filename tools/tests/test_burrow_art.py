"""Tests for the cutscene art QA tooling.

Run with: python3 -m unittest discover -s tools/tests -t tools
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from burrow_art import Spec, measure, normalize
from burrow_art.ccl import component_areas


def make_failing_frame(path: Path, width: int = 1374, height: int = 1145) -> None:
    """Write a frame with the failure profile the generator keeps producing."""
    yy, xx = np.mgrid[0:height, 0:width]
    r = np.sqrt(
        ((yy - height / 2) / (height / 2)) ** 2 + ((xx - width / 2) / (width / 2)) ** 2
    )
    alpha = np.clip((1.25 - r) * 300, 0, 255).astype(np.uint8)

    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[:, :, 0] = (120 + 60 * np.sin(xx / 40)).astype(np.uint8)
    rgb[:, :, 1] = (100 + 50 * np.cos(yy / 35)).astype(np.uint8)
    rgb[:, :, 2] = 90
    rgb[(alpha > 0) & (alpha < 90)] = (255, 0, 0)   # red fringe
    rgb[alpha == 0] = (7, 3, 11)                    # junk under transparency

    alpha[0, :] = alpha[-1, :] = 255                # touches the border
    alpha[:, 0] = alpha[:, -1] = 255
    for sy, sx in ((50, 60), (900, 1300), (300, 1200)):
        alpha[sy : sy + 3, sx : sx + 3] = 255
        rgb[sy : sy + 3, sx : sx + 3] = (200, 200, 200)

    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path)


class ConnectedComponentsTest(unittest.TestCase):
    def test_counts_diagonal_neighbours_as_one_component(self):
        mask = np.zeros((10, 10), bool)
        mask[1:4, 1:4] = True
        mask[6, 6] = True
        mask[7, 7] = True   # diagonal to (6, 6)
        mask[0, 9] = True
        _, areas = component_areas(mask)
        self.assertEqual(sorted(areas.tolist()), [1, 2, 9])

    def test_empty_mask_has_no_components(self):
        _, areas = component_areas(np.zeros((4, 4), bool))
        self.assertEqual(areas.size, 0)


class SpecTest(unittest.TestCase):
    def test_reads_verify_block_from_request(self):
        request = {
            "output": {"frame_width": 1800, "frame_height": 1500,
                       "filenames": ["M01_frame_01.png"]},
            "verify": {
                "each_frame_canvas": [1800, 1500],
                "alpha": {"opaque_must_be": 255, "partial_alpha_max_pct": 0.5,
                          "transparent_rgb_must_be_zero": True},
                "forbidden": {"pure_red_pixels": 0, "speck_components_max": 20},
                "scale": {"subject_height_px_min": 1400,
                          "subject_must_not_touch_canvas_border": True},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "REQUEST.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            spec = Spec.from_request(path)

        self.assertEqual(spec.canvas, (1800, 1500))
        self.assertEqual(spec.subject_height_px_min, 1400)
        self.assertEqual(spec.pure_red_max, 0)
        self.assertEqual(spec.filenames, ("M01_frame_01.png",))

    def test_falls_back_to_output_size_without_verify_canvas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "REQUEST.json"
            path.write_text(
                json.dumps({"output": {"frame_width": 900, "frame_height": 750}}),
                encoding="utf-8",
            )
            self.assertEqual(Spec.from_request(path).canvas, (900, 750))


class NormalizeTest(unittest.TestCase):
    def setUp(self):
        self.spec = Spec()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.source = self.tmp / "frame.png"
        make_failing_frame(self.source)

    def tearDown(self):
        self._tmp.cleanup()

    def test_source_fails_every_spec_rule_that_matters(self):
        report = measure(self.source, self.spec)
        failed = {c.name for c in report.checks if not c.ok}
        self.assertFalse(report.passed)
        self.assertIn("캔버스", failed)
        self.assertIn("부분 알파", failed)
        self.assertIn("순수 빨강 RGB(255,0,0)", failed)
        self.assertIn("피사체 높이", failed)

    def test_normalized_output_passes_every_check(self):
        out = self.tmp / "out" / "frame.png"
        normalize(self.source, out, self.spec)
        report = measure(out, self.spec)
        self.assertTrue(
            report.passed,
            "\n".join(f"{c.name}: {c.measured}" for c in report.checks if not c.ok),
        )

    def test_normalize_is_deterministic(self):
        first, second = self.tmp / "a.png", self.tmp / "b.png"
        normalize(self.source, first, self.spec)
        normalize(self.source, second, self.spec)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_portrait_source_also_normalizes(self):
        source = self.tmp / "portrait.png"
        make_failing_frame(source, width=1145, height=1374)
        out = self.tmp / "out" / "portrait.png"
        normalize(source, out, self.spec)
        self.assertTrue(measure(out, self.spec).passed)

    def test_margin_keeps_subject_off_the_border(self):
        out = self.tmp / "out" / "frame.png"
        normalize(self.source, out, self.spec, margin=40)
        with Image.open(out) as img:
            alpha = np.array(img.convert("RGBA"))[:, :, 3]
        self.assertEqual(alpha[:40, :].max(), 0)
        self.assertEqual(alpha[-40:, :].max(), 0)

    def test_rejects_a_source_too_flat_to_reach_the_minimum_height(self):
        source = self.tmp / "flat.png"
        make_failing_frame(source, width=1600, height=200)
        with self.assertRaises(ValueError):
            normalize(source, self.tmp / "out" / "flat.png", self.spec)

    def test_rejects_a_fully_transparent_source(self):
        source = self.tmp / "blank.png"
        Image.fromarray(np.zeros((100, 100, 4), np.uint8), "RGBA").save(source)
        with self.assertRaises(ValueError):
            normalize(source, self.tmp / "out" / "blank.png", self.spec)


if __name__ == "__main__":
    unittest.main()
