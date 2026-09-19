"""Deterministic QA and post-processing for THE_BURROW cutscene art.

The Slack pipeline (#굴-전체) generates cutscene frames into Drive and measures
them against the `verify` block of each request's REQUEST.json. This package is
the code side of that loop: it re-measures a frame independently and, when the
generator cannot hit the spec on its own, rewrites the frame to satisfy it.
"""

from .measure import Check, Report, measure
from .normalize import NormalizeResult, normalize
from .spec import Spec

__all__ = [
    "Check",
    "NormalizeResult",
    "Report",
    "Spec",
    "measure",
    "normalize",
]
