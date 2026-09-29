"""Scene profile: fit once from (neutral, reference), relight any frame."""
import json

from .align import Pairs
from .color import fit_color
from . import shading

VERSION = 2


def fit_scene(neutral, ref, view="walk34"):
    """neutral, ref: float RGBA of the same pose (neutral-lit drawing, hand-lit target).
    Returns (profile dict, pairs) - pairs is kept for reports."""
    pairs = Pairs(neutral, ref)
    prof = shading.fit_light(neutral, pairs, fit_color(pairs), view)
    prof["version"] = VERSION
    prof["meta"] = {"align_iou": round(pairs.iou, 4), "regions": len(pairs.regions), "fit_view": view}
    return prof, pairs


def relight(frame, profile, view=None):
    """Relight one float RGBA frame. `view`: front, front_left, front_right, left, right, back,
    back_left, back_right, walk34 (3/4 walking) or None. Alpha is never changed."""
    return shading.relight(frame, profile, view)


def save_profile(path, prof):
    with open(path, "w") as f:
        json.dump(prof, f, indent=1)


def load_profile(path):
    with open(path) as f:
        prof = json.load(f)
    if prof.get("version") != VERSION:
        raise ValueError(f"{path}: scene profile version {prof.get('version')} != {VERSION}; re-run fit")
    return prof
