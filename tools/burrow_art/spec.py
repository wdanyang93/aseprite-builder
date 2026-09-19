"""Output spec for THE_BURROW cutscene art, read from a request's REQUEST.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Spec:
    """The `verify` / `output` contract a generated frame has to satisfy."""

    canvas: tuple[int, int] = (1800, 1500)
    opaque_alpha: int = 255
    partial_alpha_max_pct: float = 0.5
    transparent_rgb_must_be_zero: bool = True
    pure_red_max: int = 0
    speck_components_max: int = 20
    speck_max_area: int = 64
    subject_height_px_min: int = 1400
    subject_must_not_touch_border: bool = True
    filenames: tuple[str, ...] = field(default=())

    @property
    def width(self) -> int:
        return self.canvas[0]

    @property
    def height(self) -> int:
        return self.canvas[1]

    @classmethod
    def from_request(cls, request_path: str | Path) -> "Spec":
        data = json.loads(Path(request_path).read_text(encoding="utf-8"))
        verify = data.get("verify", {})
        output = data.get("output", {})

        alpha = verify.get("alpha", {})
        forbidden = verify.get("forbidden", {})
        scale = verify.get("scale", {})

        canvas = verify.get("each_frame_canvas")
        if not canvas:
            canvas = [output.get("frame_width", 1800), output.get("frame_height", 1500)]

        return cls(
            canvas=(int(canvas[0]), int(canvas[1])),
            opaque_alpha=int(alpha.get("opaque_must_be", 255)),
            partial_alpha_max_pct=float(alpha.get("partial_alpha_max_pct", 0.5)),
            transparent_rgb_must_be_zero=bool(
                alpha.get("transparent_rgb_must_be_zero", True)
            ),
            pure_red_max=int(forbidden.get("pure_red_pixels", 0)),
            speck_components_max=int(forbidden.get("speck_components_max", 20)),
            subject_height_px_min=int(scale.get("subject_height_px_min", 1400)),
            subject_must_not_touch_border=bool(
                scale.get("subject_must_not_touch_canvas_border", True)
            ),
            filenames=tuple(output.get("filenames", ())),
        )
