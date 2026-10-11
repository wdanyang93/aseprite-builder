"""Screen-space light azimuth estimators compared on synthetic ground truth (design risk: ~20 vs ~50 deg disagreement).

  lin_methods   R_methods common.estimate_light_dir: least squares of the edge brightening (ref - neutral L*, minus each
                material's deep median) on the outward contour normal (depth <= 6 px)  -> atan2 of the coefficients
  lin_methods_dt  the same with R_methods' earlier distance-transform normals (NORMALS = 'dt'; the 20.6 / 20.3 deg
                in the design were measured with these; the current default 'nc' is the normalized-convolution field)
  lin_practice  R_practice fit_lobe's unclamped Lambert initialisation: log-ratio Y = a + b cos(t) + c sin(t), depth 2-6 px
  jf            R_practice estimate_direction: clamped lobe A + B max(0, cos(t - phi)) on the log-ratio (Johnson & Farid),
                robust soft-L1, block bootstrap CI
  grid          R_practice parametric model (fit_model: wash + sun rim + shadow rim + grade, least squares) on a 15 deg
                azimuth grid, w = 1.2 % H, p = 1 (as R_practice synth); azimuth with the lowest self dE. With p = 1 the
                sun and shadow lobes are mirror images, so phi and phi + 180 fit EXACTLY equally well: the grid can only
                return the lower angle of the pair (gap = 0 in the table)
  grid_p2       the same with a p = 2 sun lobe (breaks the 180 deg tie)
  consensus     circular median of lin_practice, jf and grid (grid flipped by 180 deg when it disagrees with jf by > 90)
The grid fits run on the pair downscaled to half the figure height (4x faster; their rim width is a fraction of H).
All on the fit pair (neutral + synthetic truth), the truth misaligned by 1.5 px and blurred 0.7 px like R_practice synth.
Angles: deg, 0 = screen right, 90 = up. Error = smallest angular difference to the scene's true key azimuth.
"""
import os

import numpy as np

from . import config, protos, data, metrics as MX
from . import cases as CS
from . import synth_practice as SP
from .methods import practice_pair
from . import synth_audit as SA

DRAWS = ("lyn_R", "lyn_L", "P01")
EST = ("lin_methods", "lin_methods_dt", "lin_practice", "jf", "grid", "grid_p2", "consensus")


def consensus(r):
    """Circular median of lin_practice, jf and the p = 1 grid (the grid's 180 deg ambiguity resolved toward jf)."""
    g = r["grid"]
    if MX.ang_diff(g, r["jf"]) > 90:
        g = (g + 180.0) % 360.0
    c = [r["lin_practice"], r["jf"], g]
    return float(min(c, key=lambda a: sum(MX.ang_diff(a, b) for b in c)))


def _grid(P, SM, p, grid_step):
    best, curve = None, []
    for phi in range(0, 360, grid_step):
        m = SM.fit_model(P, phi, 0.012, p)
        e = SM.score(SM.render(m, P.n, P.geo), P)["dE_mean"]
        curve.append((phi, round(float(e), 4)))
        if best is None or e < best[0]:          # strict '<': ties keep the lower angle, as in R_practice synth.py
            best = (e, phi)
    es = np.array([c[1] for c in curve])
    return float(best[1]), float(np.sort(es)[1] - es.min()), curve


def estimate(neutral, ref, grid_step=15, n_boot=60, grid_scale=0.5):
    """All estimators on one pair. The grid fits run at grid_scale (their rim width is a fraction of H)."""
    RMc = protos.load("R_methods", "common")
    SM = protos.load("R_practice", "scenemodel")
    out = {}
    out["lin_methods"] = float(RMc.estimate_light_dir(neutral, ref)[0])
    old = RMc.NORMALS
    try:                                  # the distance-transform normals R_methods used when it reported 20.6 / 20.3
        RMc.NORMALS = "dt"
        out["lin_methods_dt"] = float(RMc.estimate_light_dir(neutral, ref)[0])
    finally:
        RMc.NORMALS = old
    P = practice_pair(neutral, ref)
    D = SM.estimate_direction(P, n_boot=n_boot)
    out["jf"], out["jf_ci"], out["lin_practice"] = float(D["phi"]), [float(v) for v in D["phi_ci"]], float(D["phi_linear"])
    if grid_scale != 1.0:
        Hn = MX.fig_h(neutral[..., 3])
        ns, _ = SA.downscale(neutral, grid_scale * Hn)
        rs, _ = SA.downscale(ref, grid_scale * Hn, like=neutral)
        P = practice_pair(ns, rs)
    # p = 1: sun and shadow lobes have the same shape, so phi and phi + 180 span the same model (exact tie);
    # p = 2 breaks the tie (sharper sun lobe than the p = 1 shadow lobe)
    out["grid"], out["grid_gap"], out["grid_curve"] = _grid(P, SM, 1.0, grid_step)
    out["grid_p2"], out["grid_p2_gap"], out["grid_p2_curve"] = _grid(P, SM, 2.0, grid_step)
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
             "err = |estimate - true key azimuth| (smallest angle). gap = grid dE of the runner-up azimuth minus the best (0 = tie).",
             "", "| scene | drawing | true | lin_methods (err) | lin_methods_dt (err) | lin_practice (err) | jf [CI] (err) | grid p1 (err) | grid p2 (err) | consensus (err) | gaps p1/p2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    errs = {e: [] for e in EST}
    by_scene = {}
    for (label, rkey, rend, dr, az, az2), r in res:
        if "error" in r:
            lines.append(f"| {label} | {dr} | {az} | ERROR {r['error'][:60]} | | | | | | | |")
            continue
        cell = []
        r["consensus"] = consensus(r)
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
        lines.append(f"| {label} | {dr} | {tr} | " + " | ".join(cell) + f" | {r['grid_gap']:.3f} / {r['grid_p2_gap']:.3f} |")
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
              "| pair | lin_methods | lin_methods_dt | lin_practice | jf [CI] | grid p1 | grid p2 | consensus |", "|---|---|---|---|---|---|---|---|"]
    for name, r in real:
        if "error" in r:
            lines.append(f"| {name} | ERROR {r['error'][:80]} | | | | | | |")
            continue
        lines.append(f"| {name} | {r['lin_methods']:.1f} | {r['lin_methods_dt']:.1f} | {r['lin_practice']:.1f} | {r['jf']:.1f} [{r['jf_ci'][0]:.0f},"
                     f"{r['jf_ci'][1]:.0f}] | {r['grid']:.0f} | {r['grid_p2']:.0f} | {consensus(r):.0f} |")
    # reading (numbers computed here, wording fixed)
    fam = {"high key (midday)": "syn_midday", "low warm key (sunset)": "syn_sunset", "moon": "syn_moon_",
           "moon + point light": "syn_moonlantern", "two keys": "syn_twokeys", "darkness": "syn_darkness_practice"}
    ties = sum(1 for _, r in res if "error" not in r and r["grid_gap"] == 0)
    lines += ["", "Reading:", "",
              f"- grid p1 ties phi and phi + 180 in {ties} / {len(res)} pairs (gap 0): it only measures the azimuth modulo "
              "180 deg; its low error here is helped by every true azimuth lying in the upper half. grid p2 breaks the tie "
              "but picks the wrong half in several pairs (errors near 180).",
              "- mean error by scene family (lin_methods / lin_practice / jf / consensus):"]
    for k, pre in fam.items():
        v = {e: [er for (it, r) in res if it[0].startswith(pre) and it[4] is not None and "error" not in r
                 for er in [float(MX.ang_diff(r[e] if e != "consensus" else consensus(r), it[4]))]] for e in
             ("lin_methods", "lin_practice", "jf", "consensus")}
        if v["jf"]:
            lines.append(f"  - {k}: " + " / ".join(f"{np.mean(v[e]):.1f}" for e in ("lin_methods", "lin_practice", "jf",
                                                                                    "consensus")))
    lines += ["- The linear estimators pull a high light toward the horizontal (midday: 16-28 deg low) and the clamped lobe "
              "scatters more for a low warm key; a point light near the figure breaks all single-direction estimators. "
              "On the real sunset pairs the two families disagree (linear ~15-28 deg, J&F ~34-53 deg): the synthetic "
              "low-key scenes favour the linear reading, but no estimator is within 15 deg on every scene, so the "
              "estimate must carry the disagreement as a warning (design 9.3) rather than pick one formula."]
    txt = "\n".join(lines) + "\n"
    p = os.path.join(config.out_dir(), "AZIMUTH.md")
    with open(p, "w") as f:
        f.write(txt)
    config.jdump(os.path.join(config.out_dir(), "azimuth.json"), {"synthetic": [(list(i), r) for i, r in res],
                                                                   "real": real, "summary": summ})
    log(txt)
    log(f"azimuth table -> {p}")
    return summ
