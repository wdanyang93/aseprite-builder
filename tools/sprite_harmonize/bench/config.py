"""Bench configuration: where the private data lives and where outputs go.

Paths come from bench/paths.local.json (gitignored; copy paths.example.json). Relative data paths are resolved against
'scratch'; '<tool>' expands to the tool folder. Nothing in the repo names a private path, and nothing is written into the repo.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
if TOOL not in sys.path:
    sys.path.insert(0, TOOL)

_P = None


def paths():
    """The parsed paths.local.json (env BENCH_PATHS overrides the location)."""
    global _P
    if _P is None:
        p = os.environ.get("BENCH_PATHS", os.path.join(HERE, "paths.local.json"))
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing: copy bench/paths.example.json to bench/paths.local.json and fill in the paths")
        with open(p) as f:
            _P = json.load(f)
    return _P


def _resolve(v):
    v = v.replace("<tool>", TOOL)
    return v if os.path.isabs(v) else os.path.join(paths()["scratch"], v)


def data(key):
    """Absolute path of a configured data item."""
    return _resolve(paths()["data"][key])


def has(key):
    try:
        return os.path.exists(data(key))
    except (KeyError, FileNotFoundError):
        return False


def proto_dir(group):
    return os.path.abspath(_resolve(paths()["protos"][group]))


def source_dir(group):
    return os.path.abspath(_resolve(paths()["sources"][group]))


def out_dir(*sub):
    d = os.path.join(paths()["out"], *sub)
    os.makedirs(d, exist_ok=True)
    return d


def cache_dir(*sub):
    return out_dir("cache", *sub)


def key_hash(*parts):
    """Short stable hash of the parts (for cache file names)."""
    h = hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
    return h[:12]


def cached_array(name, fn, sub="arrays"):
    """Load cache/<sub>/<name>.npz or compute fn() (float RGBA / arrays) and store it (float16 for images)."""
    p = os.path.join(cache_dir(sub), name + ".npz")
    if os.path.exists(p):
        try:
            with np.load(p) as z:
                return z["x"].astype(np.float32)
        except Exception:
            os.remove(p)
    x = np.asarray(fn(), np.float32)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, x=x.astype(np.float16) if x.ndim == 3 else x)
    os.replace(tmp, p)
    return x


def cached_json(name, fn, sub="json"):
    p = os.path.join(cache_dir(sub), name + ".json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            os.remove(p)
    v = fn()
    with open(p + ".tmp", "w") as f:
        json.dump(v, f, default=_jdefault)
    os.replace(p + ".tmp", p)
    return v


def _jdefault(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def jdump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_jdefault)
