"""Load the research prototypes from the configured scratch folders WITHOUT copying them into the repo.

The prototype folders reuse module names (common.py, synth.py, methods.py ...). Each folder is imported with the colliding
names cleared from sys.modules, then every module that came from a scratch folder is moved to a private name
(_bench_proto_<group>_<name>) so the next folder can be imported cleanly. Functions keep their own module globals, so the
moved modules keep working.
"""
import importlib
import os
import sys

from . import config

_LOADED = {}


def load(group, mod):
    """Module `mod` of prototype folder `group` (keys of paths.local.json 'protos')."""
    key = (group, mod)
    if key in _LOADED:
        return _LOADED[key]
    d = config.proto_dir(group)
    root = os.path.abspath(config.paths()["scratch"])
    if not os.path.exists(os.path.join(d, mod + ".py")):
        raise FileNotFoundError(f"prototype {group}/{mod}.py not found under {d}")
    before = set(sys.modules)
    # names a scratch module could shadow: anything already imported from a scratch folder under a bare name
    shadow = {k: sys.modules.pop(k) for k in list(sys.modules)
              if _from(sys.modules.get(k), root) and not k.startswith("_bench_proto_")}
    old_path = list(sys.path)
    old_dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True        # never drop __pycache__ into the scratch folders from the bench
    sys.path.insert(0, d)
    try:
        m = importlib.import_module(mod)
    finally:
        for k in list(sys.modules):
            if k not in before and _from(sys.modules.get(k), root):
                v = sys.modules.pop(k)
                sys.modules[f"_bench_proto_{group}_{k}"] = v
                _LOADED.setdefault((group, k), v)
        sys.path[:] = old_path
        sys.modules.update(shadow)
        sys.dont_write_bytecode = old_dwb
    _LOADED[key] = m
    return m


def _from(m, root):
    f = getattr(m, "__file__", None)
    return bool(f) and os.path.abspath(f).startswith(root + os.sep)
