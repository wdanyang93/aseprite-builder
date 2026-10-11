"""Run methods on cases: fit phase (unique fit pair x method, cached), apply phase (case x frame: every applicable
method, outputs + metrics cached), one printed table, results files and sheets under <out>/.

Caches (delete a folder to recompute): <out>/cache/fits (models), <out>/cache/out (outputs), <out>/results (metrics).
"""
import json
import os
import time
import traceback

import numpy as np

from . import config, metrics as MX
from . import cases as CS
from . import methods as MT

def _metrics_version():
    """Results are recomputed whenever the metric code changes (hash of metrics.py + parts.py)."""
    import hashlib
    h = hashlib.sha1()
    for f in ("metrics.py", "parts.py"):
        with open(os.path.join(os.path.dirname(__file__), f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:10]


METRICS_VERSION = _metrics_version()
COLS = ("dEa", "dEst", "rL", "sun", "shd", "con", "con_ref", "det", "lnC", "hp")


def _safe(s):
    return MT._safe(s)


def out_path(fit_id, frame, method):
    return os.path.join(config.cache_dir("out", _safe(fit_id)), f"{_safe(frame)}__{method}_v{MT.METHODS[method].version}.npz")


def res_path(case, frame, method):
    return os.path.join(config.out_dir("results", case), f"{_safe(frame)}__{method}.json")


# ------------------------------------------------------------------ phases
def plan(case_names, method_names):
    fits, units = [], []
    for cn in case_names:
        c = CS.get(cn)
        ms = [m for m in method_names if c.methods is None or m in c.methods]
        for m in ms:
            if (c.fit_id, m) not in fits:
                fits.append((c.fit_id, m))
        for i in range(len(c.frames)):
            units.append((cn, i, tuple(ms)))
    return fits, units


def do_fit(job):
    fit_id, m, refit = job
    try:
        model, dt, err = MT.get_model(fit_id, m, lambda: CS.fit_pair(fit_id), refit)
        return fit_id, m, dt, err
    except Exception as e:      # e.g. missing data
        return fit_id, m, 0.0, f"{e.__class__.__name__}: {e}"


_MODELS = {}


def _model(fit_id, m):
    k = (fit_id, m)
    if k not in _MODELS:
        if len(_MODELS) > 12:
            _MODELS.clear()
        _MODELS[k] = MT.get_model(fit_id, m, lambda: CS.fit_pair(fit_id))
    return _MODELS[k]


def do_unit(job):
    """One frame of one case: every method -> output (cached) -> metrics (cached). Returns result rows."""
    cn, i, ms, fresh = job
    c = CS.get(cn)
    fr = c.frames[i]
    rows = []
    try:
        F = fr.img()
        truth = fr.truth() if fr.has_truth else None
        labels = fr.labels()
        masks = fr.masks()
    except Exception as e:
        return [{"case": cn, "frame": fr.name, "method": m, "status": f"ERROR loading: {e.__class__.__name__}: {e}"} for m in ms]
    regions = None
    pair_info = None
    for m in ms:
        row = {"case": cn, "group": c.group, "frame": fr.name, "method": m, "fit_id": c.fit_id}
        rp = res_path(cn, fr.name, m)
        op = out_path(c.fit_id, fr.name, m)
        if not fresh and os.path.exists(rp) and os.path.exists(op):
            try:
                with open(rp) as f:
                    old = json.load(f)
                if old.get("metrics_version") == METRICS_VERSION and old.get("method_version") == MT.METHODS[m].version:
                    rows.append(old)
                    continue
            except Exception:
                pass
        try:
            if pair_info is None:
                pair_info = {"drawing": CS.fit_pair(c.fit_id)["drawing"]}
            ok, why = MT.METHODS[m].applies(c, fr, pair_info)
            if not ok:
                row.update(status="n/a", note=why)
                rows.append(row)
                continue
            model, t_fit, ferr = _model(c.fit_id, m)
            row["t_fit"] = round(t_fit, 1)
            if ferr:
                row.update(status="FIT ERROR", note=ferr)
                _save(rp, row)
                rows.append(row)
                continue
            info = {}
            if os.path.exists(op) and not fresh:
                with np.load(op) as z:
                    out = z["x"].astype(np.float32)
                    info = json.loads(str(z["info"]))
                    t_apply, alpha_same, nan_out = float(z["t"]), bool(z["alpha_same"]), bool(z["nan_out"])
            else:
                t = time.time()
                out, info = MT.METHODS[m].apply(model, F, fr)
                t_apply = time.time() - t
                out = np.asarray(out, np.float32)
                # bitwise alpha / NaN checks on the method's own float32 output (the cache keeps a float16 copy)
                alpha_same = bool(out.shape == F.shape and np.array_equal(out[..., 3], F[..., 3]))
                nan_out = bool(not np.isfinite(out).all())
                tmp = op + f".{os.getpid()}.tmp.npz"
                np.savez_compressed(tmp, x=out.astype(np.float16), info=json.dumps(info, default=str), t=t_apply,
                                    alpha_same=alpha_same, nan_out=nan_out)
                os.replace(tmp, op)
            row["t_apply"] = round(t_apply, 2)
            row["info"] = info
            if regions is None and truth is not None:
                from scenelight.color import to_lab
                rel = (F[..., 3] > 0.9) & (truth[..., 3] > 0.9)
                regions = MX._regions(to_lab(np.clip(F[..., :3], 0, 1)), rel)
            sc = MX.score(out, truth, F, fr.light_deg, labels, regions=regions)
            sc["alpha_same"] = alpha_same
            if nan_out and "nan_out" not in sc["nan_keys"]:
                sc["nan_out"] = True
                sc["nan_keys"].append("nan_out")
            row.update(sc)
            if masks is not None and truth is not None:
                row["region_dE"] = MX.region_dE(out, truth, {k: v for k, v in masks.items()
                                                             if k in ("wcloth", "dcloth", "skin", "fur", "line", "all")})
            row["status"] = "ok"
        except Exception as e:
            row.update(status="ERROR", note=f"{e.__class__.__name__}: {e}", tb=traceback.format_exc().splitlines()[-3:])
        row["metrics_version"] = METRICS_VERSION
        row["method_version"] = MT.METHODS[m].version
        _save(rp, row)
        rows.append(row)
    return rows


def _save(p, row):
    tmp = p + f".{os.getpid()}.tmp"
    config.jdump(tmp, row)
    os.replace(tmp, p)


def _pool(n):
    import multiprocessing as mp
    return mp.get_context("fork").Pool(n, initializer=_init_worker)


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def run(case_names, method_names, jobs=2, fresh=False, refit=False, log=print):
    fits, units = plan(case_names, method_names)
    t0 = time.time()
    # failed fits are retried every run (a real failure fails again fast; a transient one is not kept)
    todo = [(f, m, True) for f, m in fits if refit or not os.path.exists(MT.model_path(f, m)) or MT.cached_error(f, m)]
    log(f"[bench] {len(case_names)} cases, {len(fits)} fits ({len(todo)} to compute), {len(units)} frame units, jobs={jobs}")
    fit_log = {}
    if todo:
        if jobs > 1:
            with _pool(jobs) as p:
                for fid, m, dt, err in p.imap_unordered(do_fit, todo):
                    fit_log[(fid, m)] = (dt, err)
                    log(f"  fit {fid:<34} {m:<13} {dt:6.1f}s {'ERROR ' + err if err else ''}")
        else:
            for j in todo:
                fid, m, dt, err = do_fit(j)
                log(f"  fit {fid:<34} {m:<13} {dt:6.1f}s {'ERROR ' + err if err else ''}")
    rows = []
    jobs_u = [(cn, i, ms, fresh) for cn, i, ms in units]
    done = 0
    if jobs > 1:
        with _pool(jobs) as p:
            for r in p.imap_unordered(do_unit, jobs_u):
                rows += r
                done += 1
                if done % 10 == 0:
                    log(f"  {done}/{len(units)} frame units ({time.time() - t0:.0f}s)")
    else:
        for j in jobs_u:
            rows += do_unit(j)
    log(f"[bench] done in {time.time() - t0:.0f}s")
    return rows


# ------------------------------------------------------------------ aggregation and the table
def aggregate(rows, case_order, method_order):
    agg = []
    for cn in case_order:
        for m in method_order:
            rs = [r for r in rows if r["case"] == cn and r["method"] == m]
            if not rs:
                continue
            ok = [r for r in rs if r.get("status") == "ok"]
            a = {"case": cn, "method": m, "group": CS.get(cn).group, "n": len(ok), "n_frames": len(rs)}
            for k in COLS + ("mL", "mL_ref"):
                v = [r[k] for r in ok if k in r and r[k] is not None]
                a[k] = float(np.mean(v)) if v and all(np.isfinite(v)) else (float("nan") if v else None)
                if v and not all(np.isfinite(v)):
                    a.setdefault("nan_cols", []).append(k)
            a["alpha_same"] = all(r.get("alpha_same", False) for r in ok) if ok else None
            a["lit_lines"] = int(sum(r.get("lit_lines", 0) for r in ok))
            a["n_line"] = int(sum(r.get("n_line", 0) for r in ok))
            a["nan"] = sorted({k for r in ok for k in r.get("nan_keys", [])})
            a["t_fit"] = max([r.get("t_fit", 0) or 0 for r in rs])
            a["t_apply"] = float(np.mean([r.get("t_apply", 0) for r in ok])) if ok else None
            st = sorted({r.get("status", "?") for r in rs if r.get("status") != "ok"})
            paths = sorted({r.get("info", {}).get("path") for r in ok if r.get("info", {}).get("path")})
            a["note"] = "; ".join(st + paths + sorted({r.get("note", "")[:60] for r in rs if r.get("note")}))
            agg.append(a)
    return agg


def fmt_table(agg):
    hdr = ("case", "method", "n", "dEa", "dEst", "rL", "sun", "shd", "con", "(ref)", "det", "lnC", "hp", "a=", "lit",
           "NaN", "fit s", "app s", "note")
    w = (22, 12, 5, 6, 6, 5, 6, 6, 6, 6, 5, 5, 5, 3, 6, 4, 6, 6, 0)
    f = lambda v, d=2: "-" if v is None else ("nan!" if isinstance(v, float) and not np.isfinite(v) else f"{v:.{d}f}")
    lines = [" ".join(h.ljust(x) if x else h for h, x in zip(hdr, w))]
    lines.append("-" * 150)
    last = None
    for a in agg:
        if last and a["group"] != last:
            lines.append("")
        last = a["group"]
        cells = (a["case"][:22], a["method"], f"{a['n']}/{a['n_frames']}", f(a["dEa"]), f(a["dEst"]), f(a["rL"]),
                 f(a["sun"]), f(a["shd"]), f(a["con"], 1), f(a["con_ref"], 1), f(a["det"]), f(a["lnC"]), f(a["hp"]),
                 "-" if a["alpha_same"] is None else ("y" if a["alpha_same"] else "NO"),
                 str(a["lit_lines"]), str(len(a["nan"])) if a["nan"] else "0",
                 f(a["t_fit"], 0), f(a["t_apply"], 1), a["note"][:70])
        lines.append(" ".join(c.ljust(x) if x else c for c, x in zip(cells, w)))
    return "\n".join(lines)


def write_results(rows, agg, path_base):
    config.jdump(path_base + ".json", {"rows": rows, "agg": agg})
    keys = ("case", "group", "frame", "method", "status") + COLS + ("mL", "mL_ref", "alpha_same", "lit_lines", "n_line",
                                                                   "edge_mode", "t_fit", "t_apply", "note")
    with open(path_base + ".tsv", "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(k) is None else (f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k)))
                              for k in keys) + "\n")
