"""Screen-space light azimuth estimators compared on synthetic ground truth (design risk: ~20 vs ~50 deg disagreement).

  lin_methods   R_methods common.estimate_light_dir: least squares of the edge brightening (ref - neutral L*, minus each
                material's deep median) on the outward contour normal (depth <= 6 px)  -> atan2 of the coefficients
  lin_practice  R_practice fit_lobe's unclamped Lambert initialisation: log-ratio Y = a + b cos(t) + c sin(t), depth 2-6 px
  jf            R_practice estimate_direction: clamped lobe A + B max(0, cos(t - phi)) on the log-ratio (Johnson & Farid),
                robust soft-L1, block bootstrap CI
  grid          R_practice parametric model (fit_model: wash + sun rim + shadow rim + grade, least squares) on a 15 deg
                azimuth grid, w = 1.2 % H, p = 1; azimuth with the lowest self dE
All on the fit pair (neutral + synthetic truth), the truth misaligned by 1.5 px and blurred 0.7 px like R_practice synth.
Angles: deg, 0 = screen right, 90 = up. Error = smallest angular difference to the scene's true key azimuth.
"""
import os

import numpy as np

from . import config, protos, data, metrics as MX
from . import cases as CS
from . import synth_practice as SP
from .methods import practice_pair

DRAWS = ("lyn_R", "lyn_L", "P01")
EST = ("lin_methods", "lin_practice", "jf", "grid")


def estimate(neutral, ref, grid_step=15, n_boot=60):
    RMc = protos.load("R_methods", "common")
    SM = protos.load("R_practice", "scenemodel")
    out = {}
    out["lin_methods"] = float(RMc.estimate_light_dir(neutral, ref)[0])
    P = practice_pair(neutral, ref)
    D = SM.estimate_direction(P, n_boot=n_boot)
    out["jf"], out["jf_ci"], out["lin_practice"] = float(D["phi"]), [float(v) for v in D["phi_ci"]], float(D["phi_linear"])
    best = None
    curve = []
    for phi in range(0, 360, grid_step):
        m = SM.fit_model(P, phi, 0.012, 1.0)
        e = SM.score(SM.render(m, P.n, P.geo), P)["dE_mean"]
        curve.append((phi, round(float(e), 3)))
        if best is None or e < best[0]:
            best = (e, phi)
    out["grid"] = float(best[1])
    es = np.array([c[1] for c in curve])
    out["grid_flat"] = float(np.sort(es)[1] - es.min()) if len(es) > 1 else np.nan     # gap to the runner-up
    out["grid_curve"] = curve
    return out


def items():
    """(label, scene key, renderer, drawing, true azimuth, second azimuth or None)."""
    out = []
    for (scene, rend), (rkey, az, note) in CS.SYNTH.items():
        drs = ("lyn_R", "lyn_L") if rkey in CS.GRADED else DRAWS
        truth_az = None if rkey in CS.GRADED else az
        for dr in drs:
            out.append((f"syn_{scene}_{rend}", rkey, rend, dr, truth_az, CS.SECOND_KEY.get((scene, rend))))
    return out


def _one(it):
    label, rkey, rend, dr, az, az2 = it
    def make():
        N = data.drawing(dr)
        G = SP.misalign(CS.synth_truth(rend, rkey, dr))
        return estimate(N, G)
    try:
        r = config.cached_json(f"az_{label}_{dr}", make, sub="azimuth")
    except Exception as e:
        r = {"error": f"{e.__class__.__name__}: {e}"}
    return it, r


def real_items():
    return [("real", "lyn_R", lambda: (data.lyn("R")["neutral"], data.lyn("R")["ref"])),
            ("real", "lyn_L", lambda: (data.lyn("L")["neutral"], data.lyn("L")["ref"]))] + \
           [("real", f"P01_{v}", (lambda v=v: (data.p01(v)["neutral"], data.p01(v)["ref"]))) for v in data.P01_REFS]


def main(jobs=2, log=print):
    its = items()
    res = []
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as p:
            res = list(p.imap(_one, its))
    else:
        res = [_one(i) for i in its]
    real = []
    for _, name, fn in real_items():
        try:
            r = config.cached_json(f"az_real_{name}", lambda fn=fn: estimate(*fn()), sub="azimuth")
        except Exception as e:
            r = {"error": f"{e.__class__.__name__}: {e}"}
        real.append((name, r))
    # table
    lines = ["# Azimuth estimators on synthetic ground truth (B0)", "",
             "Angles in degrees, 0 = screen right, 90 = up. Fit pair = drawing + synthetic truth misaligned 1.5 px + 0.7 px blur.",
             "err = |estimate - true key azimuth| (smallest angle). 'flat' = grid dE gap between the best and the runner-up azimuth.",
             "", "| scene | drawing | true | lin_methods (err) | lin_practice (err) | jf [CI] (err) | grid (err) | grid flat |",
             "|---|---|---|---|---|---|---|---|"]
    errs = {e: [] for e in EST}
    by_scene = {}
    for (label, rkey, rend, dr, az, az2), r in res:
        if "error" in r:
            lines.append(f"| {label} | {dr} | {az} | ERROR {r['error'][:60]} | | | | |")
            continue
        cell = []
        for e in EST:
            v = r[e]
            if az is not None:
                er = float(MX.ang_diff(v, az))
                errs[e].append(er)
                by_scene.setdefault(label, {}).setdefault(e, []).append(er)
                s = f"{v:.1f} ({er:.0f})"
                if e == "jf":
                    s = f"{v:.1f} [{r['jf_ci'][0]:.0f},{r['jf_ci'][1]:.0f}] ({er:.0f})"
            else:
                s = f"{v:.1f}"
            cell.append(s)
        tr = "real ref" if az is None else (f"{az:.0f}" + (f" (+{az2:.0f})" if az2 else ""))
        lines.append(f"| {label} | {dr} | {tr} | " + " | ".join(cell) + f" | {r['grid_flat']:.2f} |")
    lines += ["", "| estimator | mean err | median err | max err | n | share < 15 deg |", "|---|---|---|---|---|---|"]
    summ = {}
    for e in EST:
        v = np.array(errs[e])
        if len(v):
            summ[e] = dict(mean=float(v.mean()), median=float(np.median(v)), max=float(v.max()), n=len(v),
                           ok=float((v < 15).mean()))
            lines.append(f"| {e} | {v.mean():.1f} | {np.median(v):.1f} | {v.max():.1f} | {len(v)} | {(v < 15).mean():.0%} |")
    lines += ["", "Per scene mean error:", "", "| scene | " + " | ".join(EST) + " |", "|---|" + "---|" * len(EST)]
    for label, d in by_scene.items():
        lines.append(f"| {label} | " + " | ".join(f"{np.mean(d[e]):.1f}" for e in EST) + " |")
    lines += ["", "Real pairs (no truth; the background sun of the sunset scene points to 22-29 deg, the sunrim metric uses 40):", "",
              "| pair | lin_methods | lin_practice | jf [CI] | grid |", "|---|---|---|---|---|"]
    for name, r in real:
        if "error" in r:
            lines.append(f"| {name} | ERROR {r['error'][:80]} | | | |")
            continue
        lines.append(f"| {name} | {r['lin_methods']:.1f} | {r['lin_practice']:.1f} | {r['jf']:.1f} [{r['jf_ci'][0]:.0f},"
                     f"{r['jf_ci'][1]:.0f}] | {r['grid']:.0f} |")
    txt = "\n".join(lines) + "\n"
    p = os.path.join(config.out_dir(), "AZIMUTH.md")
    with open(p, "w") as f:
        f.write(txt)
    config.jdump(os.path.join(config.out_dir(), "azimuth.json"), {"synthetic": [(list(i), r) for i, r in res],
                                                                   "real": real, "summary": summ})
    log(txt)
    log(f"azimuth table -> {p}")
    return summ
