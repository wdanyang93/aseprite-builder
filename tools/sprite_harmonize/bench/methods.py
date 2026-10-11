"""Method adapters: every method is  fit(pair) -> model  and  apply(model, frame) -> (RGBA, info).

pair = dict(neutral, ref (in the neutral's geometry), ref_raw (what the tool's own fit gets), neutral_path, drawing).
A model must be picklable (plain dicts / numpy arrays): prototype objects are stored as their attribute dicts.

  tool_profile  current tool, profile-only path: scenelight.fit_scene + scenelight.relight (wash + rim)
  tool_pack     current tool, reference-pack path: + transfer.build_pack / relight_transfer (gates, chest stage; a frame
                that fails the gates gets the wash+rim fallback, info['path'] says which)
  M4v3          R_methods V3(RatioTable(shrink=20))  multiplicative angle x depth x luminance table
  M2av3         R_methods V3(AngleTable(shrink=20))  colour grid + angle x depth x material residual table
  proto_ratio   R_audit proto_ratio.ratio_field      per-material low-pass light ratio field (same geometry only)
  model2        R_practice model2.fit2 / render2     multiply light + additive rim + hue-agnostic grade; azimuth from
                the pair's clamped-lobe estimate (R_practice estimate_direction), geometry first
Prototype code is loaded from the configured scratch folders (bench/protos.py), never copied into the repo.
A new method plugs in by subclassing Method (name, version, fit, apply, optionally applies) and adding it to METHODS.
"""
import os
import pickle
import time

import numpy as np

from . import config, protos


class Method:
    name = "base"
    version = "1"
    same_geometry_only = False

    def applies(self, case, frame, pair):
        """(True, '') or (False, reason)."""
        if self.same_geometry_only and not frame.same_geometry:
            return False, "needs the fit neutral's geometry"
        return True, ""

    def fit(self, pair):
        raise NotImplementedError

    def apply(self, model, F, frame=None):
        raise NotImplementedError


# ------------------------------------------------------------------ prototype object <-> plain state
def to_state(obj, mod):
    if isinstance(obj, (np.ndarray, np.generic, int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: to_state(v, mod) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_state(v, mod) for v in obj)
    if isinstance(obj, type):
        return {"__type__": obj.__module__ + "." + obj.__qualname__}
    cls = type(obj)
    assert getattr(mod, cls.__name__, None) is cls, f"{cls} is not from {mod.__name__}"
    return {"__cls__": cls.__name__, "state": {k: to_state(v, mod) for k, v in vars(obj).items()}}


def from_state(s, mod):
    if isinstance(s, dict) and "__cls__" in s:
        cls = getattr(mod, s["__cls__"])
        o = cls.__new__(cls)
        o.__dict__.update({k: from_state(v, mod) for k, v in s["state"].items()})
        return o
    if isinstance(s, dict) and "__type__" in s:
        return None
    if isinstance(s, dict):
        return {k: from_state(v, mod) for k, v in s.items()}
    if isinstance(s, (list, tuple)):
        return type(s)(from_state(v, mod) for v in s)
    return s


# ------------------------------------------------------------------ current tool
class ToolProfile(Method):
    name, version = "tool_profile", "1"

    def fit(self, pair):
        from scenelight import fit_scene
        prof, pairs = fit_scene(pair["neutral"], pair["ref_raw"])
        return {"prof": prof, "align_iou": pairs.iou}

    def apply(self, model, F, frame=None):
        from scenelight import relight
        return relight(F, model["prof"], frame.view if frame is not None else None), {"path": "wash+rim"}


class ToolPack(Method):
    """Pack path. Only where the frame can line up with the fit neutral (same drawing: itself, recolours, small and
    pseudo frames); other drawings fail the gates by design and would get exactly tool_profile."""
    name, version = "tool_pack", "1"

    def applies(self, case, frame, pair):
        if frame.drawing != pair.get("drawing"):
            return False, "other drawing: the pack gates fall back to wash+rim (= tool_profile)"
        return True, ""

    def fit(self, pair):
        import hashlib
        from scenelight import fit_scene, transfer
        prof, pairs = fit_scene(pair["neutral"], pair["ref_raw"])
        pk = transfer.build_pack(pair["neutral"], pair["ref_raw"], prof)
        h = hashlib.sha1(np.ascontiguousarray(pair["neutral"]).tobytes() + np.ascontiguousarray(pair["ref_raw"]).tobytes())
        d = config.cache_dir("packs", h.hexdigest()[:16])        # saved and re-loaded exactly like `fit --pack` + `apply`
        transfer.save_pack(d, pk)
        return {"prof": prof, "pack_dir": d, "align_iou": pairs.iou, "pack_align_iou": float(pk["align_iou"])}

    def apply(self, model, F, frame=None):
        from scenelight import transfer
        if "_pack" not in model:
            model["_pack"] = transfer.load_pack(model["pack_dir"])
        out, info = transfer.relight_transfer(F, model["prof"], model["_pack"], frame.view if frame is not None else None)
        info = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}
        info["path"] = "pack" if info.get("passed") else "wash+rim fallback"
        return out, info


# ------------------------------------------------------------------ R_methods prototypes
class _RM(Method):
    inner = None

    def _mod(self):
        return protos.load("R_methods", "methods")

    def fit(self, pair):
        RM = self._mod()
        m = RM.V3(getattr(RM, self.inner)(shrink=20.0)).fit(pair["neutral"], pair["ref"])
        return {"state": to_state(m, RM)}

    def apply(self, model, F, frame=None):
        RM = self._mod()
        m = from_state(model["state"], RM)
        return m.apply(F), {}


class M4v3(_RM):
    name, version, inner = "M4v3", "1", "RatioTable"


class M2av3(_RM):
    name, version, inner = "M2av3", "1", "AngleTable"


# ------------------------------------------------------------------ R_audit prototype
class ProtoRatio(Method):
    name, version = "proto_ratio", "1"
    same_geometry_only = True

    def applies(self, case, frame, pair):
        if not frame.same_geometry:
            return False, "light field lives in the fit neutral's pixels (same pose / geometry only)"
        return True, ""

    def fit(self, pair):
        PR = protos.load("R_audit", "proto_ratio")
        return {"G": PR.ratio_field(pair["neutral"], pair["ref"])}

    def apply(self, model, F, frame=None):
        PR = protos.load("R_audit", "proto_ratio")
        if model["G"].shape[:2] != F.shape[:2]:
            raise ValueError("frame geometry differs from the fit neutral")
        return PR.apply_ratio(F, model["G"]), {}


# ------------------------------------------------------------------ R_practice prototype
def practice_pair(neutral, ref, defringe_again=False):
    """scenemodel.Pair from arrays. The bench frames are already defringed, so the Pair's own defringe is skipped
    (defringe is not idempotent); defringe_again=True reproduces the research call on already-defringed arrays."""
    SM = protos.load("R_practice", "scenemodel")
    old_ld, old_df = SM.load_rgba, SM.defringe
    try:
        SM.load_rgba = lambda p: neutral.copy()
        if not defringe_again:
            SM.defringe = lambda rgb, a, radius=4: rgb
        return SM.Pair("<array>", ref.astype(np.float32), "bench")
    finally:
        SM.load_rgba, SM.defringe = old_ld, old_df


class Model2(Method):
    """Azimuth first, colour after (R_practice: geometry first - a joint least-squares fit is flat in the azimuth and
    drifts): phi = the pair's own clamped-lobe (Johnson & Farid) estimate, then fit2 at that phi."""
    name, version = "model2", "2"
    phi = None          # fixed azimuth (deg) instead of the estimate

    def fit(self, pair, phi=None, w=0.01):
        SM, M2 = protos.load("R_practice", "scenemodel"), protos.load("R_practice", "model2")
        P = practice_pair(pair["neutral"], pair["ref"])
        phi = self.phi if phi is None else phi
        est = None
        if phi is None:
            D = SM.estimate_direction(P, n_boot=20)
            est = {"phi": D["phi"], "ci": D["phi_ci"], "phi_linear": D["phi_linear"]}
            phi = float(D["phi"])
        m = M2.fit2(P, phi, w, iters=3)
        return {"model": m, "phi": phi, "estimate": est}

    def apply(self, model, F, frame=None):
        M2 = protos.load("R_practice", "model2")
        return M2.render2(model["model"], F), {"phi": round(float(model["phi"]), 1)}


METHODS = {m.name: m for m in (ToolProfile(), ToolPack(), M4v3(), M2av3(), ProtoRatio(), Model2())}
DEFAULT = ("tool_profile", "tool_pack", "M4v3", "M2av3", "proto_ratio", "model2")


# ------------------------------------------------------------------ fitted-model cache
def _safe(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def model_path(fit_id, method):
    m = METHODS[method]
    return os.path.join(config.cache_dir("fits"), f"{_safe(fit_id)}__{method}_v{m.version}.pkl")


def cached_error(fit_id, method):
    """Error string of a cached failed fit (None if the fit is cached fine or not cached)."""
    p = model_path(fit_id, method)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f).get("error")
    except Exception:
        return "unreadable cache"


def get_model(fit_id, method, pair_fn, refit=False):
    """Fitted model from the cache, or fit it now. Returns (model, fit_seconds, error or None)."""
    p = model_path(fit_id, method)
    if os.path.exists(p) and not refit:
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            return d["model"], d["t_fit"], d.get("error")
        except Exception:
            os.remove(p)
    t = time.time()
    err, model = None, None
    try:
        model = METHODS[method].fit(pair_fn())
    except Exception as e:  # recorded, never hidden: e.g. the tool's fit crash on 96 px frames
        import traceback
        err = f"{e.__class__.__name__}: {e} @ {traceback.extract_tb(e.__traceback__)[-1].filename.split('/')[-1]}:" \
              f"{traceback.extract_tb(e.__traceback__)[-1].lineno}"
    dt = time.time() - t
    tmp = p + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"model": model, "t_fit": dt, "error": err}, f, protocol=4)
    os.replace(tmp, p)
    return model, dt, err
