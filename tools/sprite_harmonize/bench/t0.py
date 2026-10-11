"""T0: reproduce the numbers the research reports (R_methods, R_audit, R_practice, s598 BASELINE) with the bench's
adapters, renderers and the unified metrics module (compat replicas of each source's own metric). One table -> <out>/T0.md.

Source numbers are read from the research JSON files at run time (cited by file and key); numbers that only exist in the
design document / BASELINE.md are quoted with that citation. Pass = reproduced within +-5 % of the source value
(angles: within 5 % or 2 deg).
"""
import json
import os
import time

import numpy as np

from scenelight.color import figure_height

from . import config, data, protos, metrics as MX
from . import cases as CS
from . import methods as MT
from . import run as RUN
from . import synth_practice as SP
from . import synth_methods as SMe
from .methods import practice_pair

ROWS = []


class _F:      # minimal frame for adapter apply (view hint only)
    def __init__(self, view=None):
        self.view = view


def row(src, setting, method, qty, src_v, rep_v, note="", angle=False):
    if src_v is None or rep_v is None or not np.isfinite(src_v) or not np.isfinite(rep_v):
        d, ok = float("nan"), False
    else:
        d = (rep_v - src_v) / max(abs(src_v), 1e-9) * 100
        ok = abs(d) <= 5.0 or (angle and MX.ang_diff(rep_v, src_v) <= 2.0) or abs(rep_v - src_v) < 0.005
    ROWS.append(dict(src=src, setting=setting, method=method, qty=qty, src_v=src_v, rep_v=rep_v, d=d, ok=ok, note=note))


def _src(group, *p):
    return os.path.join(config.source_dir(group), *p)


def bench_output(fit_id, frame_name, method, F, view=None):
    """The bench adapter's output (same cache as `run`)."""
    p = RUN.out_path(fit_id, frame_name, method)
    if os.path.exists(p):
        with np.load(p) as z:
            return z["x"].astype(np.float32)
    model, _, err = MT.get_model(fit_id, method, lambda: CS.fit_pair(fit_id))
    if err:
        raise RuntimeError(err)
    t = time.time()
    out, info = MT.METHODS[method].apply(model, F, _F(view))
    out = np.asarray(out, np.float32)
    tmp = p + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, x=out.astype(np.float16), info=json.dumps(info, default=str), t=time.time() - t,
                        alpha_same=bool(np.array_equal(out[..., 3], F[..., 3])), nan_out=bool(not np.isfinite(out).all()))
    os.replace(tmp, p)
    return out


# ------------------------------------------------------------------ A-C: R_methods
RM_KEYS = (("dE_all", "dEa"), ("dE_struct", "dEst"), ("corrL_local", "rL"), ("rim_sun_gap", "sun"), ("rim_shadow_gap", "shd"),
           ("rim_contrast", "con"), ("detail_kept", "det"), ("line_contrast", "lnC"))


def r_methods():
    run3 = {r["label"].strip(): r for r in json.load(open(_src("R_methods", "run3.json")))}
    lab = {("R", "L", "M2av3"): "R->L v3 M2a angle table s20", ("R", "L", "M4v3"): "R->L v3 M4 ratio s20",
           ("L", "R", "M2av3"): "L->R v3 M2a angle table s20", ("L", "R", "M4v3"): "L->R v3 M4 ratio s20",
           ("R", "R", "M2av3"): "train R->R M2a", ("R", "R", "M4v3"): "train R->R M4",
           ("L", "L", "M2av3"): "train L->L M2a", ("L", "L", "M4v3"): "train L->L M4"}
    for (f, t, m), label in lab.items():
        nt, rt = data.lyn(t)["neutral"], data.lyn(t)["ref"]
        out = bench_output(f"lyn:{f}", f"lyn_{t}", m, nt)
        r = MX.score_rmethods(out, rt, nt, data.LYN[t][2])
        keys = RM_KEYS if f != t else RM_KEYS[:2] + RM_KEYS[3:5]
        for k, short in keys:
            row("R_methods run3.json", f"{label}", m, short, run3[label][k], r[k])
    # B: tool wash+rim with the stored profiles (as baselines.py) and with the bench's fresh fit
    base = {r["label"]: r for r in json.load(open(_src("R_methods", "baselines.json")))}
    from scenelight import scene
    prof = {"R": scene.load_profile(config.data("profile_R")), "L": scene.load_profile(config.data("profile_L"))}
    for f, t in (("R", "L"), ("L", "R")):
        nt, rt = data.lyn(t)["neutral"], data.lyn(t)["ref"]
        label = f"{f}->{t} tool wash+rim (other fit)"
        r = MX.score_rmethods(scene.relight(nt, prof[f], None), rt, nt, data.LYN[t][2])
        rb = MX.score_rmethods(bench_output(f"lyn:{f}", f"lyn_{t}", "tool_profile", nt), rt, nt, data.LYN[t][2])
        for k, short in (("dE_all", "dEa"), ("corrL_local", "rL"), ("rim_sun_gap", "sun"), ("rim_shadow_gap", "shd"),
                         ("rim_contrast", "con")):
            row("R_methods baselines.json", label + " [stored profile]", "tool_profile", short, base[label][k], r[k])
            row("R_methods baselines.json", label + " [bench fresh fit]", "tool_profile", short, base[label][k], rb[k],
                "fresh fit_scene on the pair vs the stored scenes/ profile")
    # C: synthetic night (R_methods renderer) R -> L and the 8 turnaround views
    night = {r["label"]: r for r in json.load(open(_src("R_methods", "run_night.json")))}
    fid = "syn:methods/moon_lantern/lyn_R"
    for m, lname, tname in (("M4v3", "night R->L M4 ratio", "night turn mean M4 ratio"),
                            ("M2av3", "night R->L M2a angle table", "night turn mean M2a angle table")):
        NL = data.drawing("lyn_L")
        r = MX.score_rmethods(bench_output(fid, "lyn_L", m, NL), CS.synth_truth("methods", "moon_lantern", "lyn_L"), NL, None)
        for k, short in RM_KEYS[:3] + RM_KEYS[6:7]:
            row("R_methods run_night.json", lname, m, short, night[lname][k], r[k])
        acc = []
        for v in data.TURN_VIEWS:
            T = data.drawing(f"turn_{v}")
            rr = MX.score_rmethods(bench_output(fid, f"turn_{v}", m, T, data.TURN_HINT[v]),
                                   CS.synth_truth("methods", "moon_lantern", f"turn_{v}"), T, None)
            acc.append((rr["dE_all"], rr["dE_struct"]))
        acc = np.array(acc)
        row("R_methods run_night.json", tname, m, "dEa", night[tname]["dE_all"], float(acc[:, 0].mean()))
        row("R_methods run_night.json", tname, m, "dEst", night[tname]["dE_struct"], float(acc[:, 1].mean()))


def r_methods_lomo():
    """DESIGN 2.5 / T6 (R_methods run_lomo_night.py, printed only): hide one material when fitting on the synthetic night,
    test it on the other direction. M4 skin 6.01 -> 5.83, M2a 6.30 -> 7.92."""
    RM = protos.load("R_methods", "methods")
    from scenelight.color import materials
    NR, NL = data.drawing("lyn_R"), data.drawing("lyn_L")
    TR, TL = CS.synth_truth("methods", "moon_lantern", "lyn_R"), CS.synth_truth("methods", "moon_lantern", "lyn_L")
    mats = materials(NR)
    def make():
        out = {}
        try:
            for hide in ("none", "skin"):
                RM.EXCL = None if hide == "none" else (mats[hide] > 0.3)
                for name, mk in (("M2a", lambda: RM.V3(RM.AngleTable(shrink=20.0))), ("M4", lambda: RM.V3(RM.RatioTable(shrink=20.0)))):
                    o = mk().fit(NR, TR).apply(NL)
                    out[f"{name}_{hide}"] = MX.score_rmethods(o, TL, NL, None)["dE_skin"]
        finally:
            RM.EXCL = None
        return out
    r = config.cached_json("t0_lomo_night", make, sub="t0")
    for key, src_v in (("M4_none", 6.01), ("M4_skin", 5.83), ("M2a_none", 6.30), ("M2a_skin", 7.92)):
        m, hide = key.split("_")
        row("DESIGN_GENERAL 2.5 (R_methods run_lomo_night.py)", f"synthetic night R->L, skin dE, hidden in fit: {hide}",
            m + "v3", "dE skin", src_v, r[key])


# ------------------------------------------------------------------ D-E: R_audit
def r_audit():
    pr = json.load(open(_src("R_audit", "proto_ratio.json")))
    M = data.audit_masks()
    N = data.lyn("R")["neutral"]
    Hf = figure_height(N[..., 3])
    tests = [("sunset: neutral", "lyn:R", "lyn_R", N, data.lyn("R")["ref"]),
             ("sunset: outfit midblue (existing scene)", "lyn:R", "outfit_midblue") + CS.recolour_frame("outfit_midblue"),
             ("sunset: outfit navy (existing scene)", "lyn:R", "outfit_navy") + CS.recolour_frame("outfit_navy"),
             ("sunset: dark skin (existing scene)", "lyn:R", "darkskin") + CS.recolour_frame("darkskin"),
             ("night_ref: neutral", "syn:audit/night_ref/lyn_R", "lyn_R", N, CS.synth_truth("audit", "night_ref", "lyn_R")),
             ("darkness_ref: neutral", "syn:audit/darkness_ref/lyn_R", "lyn_R", N, CS.synth_truth("audit", "darkness_ref", "lyn_R")),
             ("night_lantern: neutral", "syn:audit/night_lantern/lyn_R", "lyn_R", N, CS.synth_truth("audit", "night_lantern", "lyn_R")),
             ("midday: neutral", "syn:audit/midday/lyn_R", "lyn_R", N, CS.synth_truth("audit", "midday", "lyn_R"))]
    for name, fid, fname, F, GT in tests:
        for m, key in (("proto_ratio", "ratio"), ("tool_pack", "tool_pack")):
            ev = MX.eval_audit(bench_output(fid, fname, m, F), GT, M, Hf)
            for k in ("wcloth", "skin", "all"):
                row("R_audit proto_ratio.json", name, m, f"lp_dE {k}", pr[name][key][k]["lp_dE"], ev[k]["lp_dE"])
            row("R_audit proto_ratio.json", name, m, "hp all", pr[name][key]["all"]["hp_ratio"], ev["all"]["hp_ratio"])
    # the darkness case once more on R_audit's own float16 copy of the aligned reference (input-precision check)
    z = np.load(_src("R_audit", "cache", "lyn_right.npz"))
    from . import synth_audit as SAU
    G16 = SAU.darkness_from_ref(z["R0"].astype(np.float32))
    def make16():
        from scenelight import fit_scene, transfer
        prof, _ = fit_scene(N, G16)
        d = config.cache_dir("t0_pack_darkness_f16")
        transfer.save_pack(d, transfer.build_pack(N, G16, prof))
        out, _ = transfer.relight_transfer(N, prof, transfer.load_pack(d), None, None, True)
        ev = MX.eval_audit(out, G16, M, Hf)
        return {k: ev[k]["lp_dE"] for k in ("wcloth", "skin", "all")}
    r16 = config.cached_json("t0_darkness_f16", make16, sub="t0")
    for k in ("wcloth", "skin", "all"):
        row("R_audit proto_ratio.json", "darkness_ref: neutral [R_audit's float16 reference copy]", "tool_pack",
            f"lp_dE {k}", pr["darkness_ref: neutral"]["tool_pack"][k]["lp_dE"], r16[k],
            "same input bits as R_audit: exact")
    # out/<case>/log.json: the current tool on the real pair, the left frame, small frames, graded-reference alignment
    lg = lambda c: json.load(open(_src("R_audit", "out", c, "log.json")))
    L = lg("base_real")
    ev = MX.eval_audit(bench_output("lyn:R", "lyn_R", "tool_profile", N, None), data.lyn("R")["ref"], M, Hf)
    for k in ("wcloth", "skin", "all"):
        row("R_audit out/base_real/log.json", "neutral, wash+rim", "tool_profile", f"lp_dE {k}",
            L["frames"]["neutral"]["eval"]["wash"][k]["lp_dE"], ev[k]["lp_dE"])
    from scenelight import relight
    prof = MT.get_model("lyn:R", "tool_profile", lambda: CS.fit_pair("lyn:R"))[0]["prof"]
    NL, RL = data.lyn("L")["neutral"], data.lyn("L")["ref"]
    ev = MX.eval_audit(relight(NL, prof, "left"), RL, data.class_masks(NL), figure_height(NL[..., 3]))
    for k in ("skin", "all"):
        row("R_audit out/base_real/log.json", "left frame, wash+rim, view=left", "tool_profile", f"lp_dE {k}",
            L["frames"]["left"]["eval"]["wash"][k]["lp_dE"], ev[k]["lp_dE"])
    row("R_audit out/base_real/log.json", "fit alignment", "tool_profile", "align IoU", L["profile"]["align_iou"],
        MT.get_model("lyn:R", "tool_profile", lambda: CS.fit_pair("lyn:R"))[0]["align_iou"])
    Ls = lg("base_small")
    for h in (192, 96):
        Fs, Gs = CS.small_frame(h)
        ev = MX.eval_audit(bench_output("lyn:R", f"small_{h}", "tool_pack", Fs), Gs, {"all": Fs[..., 3] > 0.5},
                           figure_height(Fs[..., 3]))
        row("R_audit out/base_small/log.json", f"small_{h} (fit full size)", "tool_pack", "lp_dE all",
            Ls["frames"][f"small_{h}"]["eval"]["pack"]["all"]["lp_dE"], ev["all"]["lp_dE"])
    for c, fid in (("night_ref", "syn:audit/night_ref/lyn_R"), ("darkness_ref", "syn:audit/darkness_ref/lyn_R"),
                   ("midday", "syn:audit/midday/lyn_R"), ("night_lantern", "syn:audit/night_lantern/lyn_R")):
        Lc = lg(c)
        model = MT.get_model(fid, "tool_profile", lambda fid=fid: CS.fit_pair(fid))[0]
        row(f"R_audit out/{c}/log.json", f"{c}: fit alignment (dark refs break the gray-image flow)", "tool_profile",
            "align IoU", Lc["profile"]["align_iou"], model["align_iou"] if model else float("nan"))
    _, _, err = MT.get_model("small:96", "tool_profile", lambda: CS.fit_pair("small:96"))
    src_err = lg("small_fit_96")["fit"]
    ROWS.append(dict(src="R_audit out/small_fit_96/log.json", setting="fit at figure height 96 px", method="tool_profile",
                     qty="status", src_v=src_err[:60], rep_v=str(err)[:60], d=float("nan"),
                     ok=bool(err) and "concatenate" in str(err), note="same crash (shading.py:322)"))


# ------------------------------------------------------------------ F-H: R_practice
def _practice_synth(name):
    """R_practice synth.py main loop for one scene (direction, interior, grid fit, cross direction)."""
    SM = protos.load("R_practice", "scenemodel")
    def make():
        sc = SP.SCENES[name]
        nR, nL = SM.load_rgba(config.data("lyn_R_neutral")), SM.load_rgba(config.data("lyn_L_neutral"))
        refR, refL = SP.light_scene(nR, sc), SP.light_scene(nL, sc)
        PR = SM.Pair(config.data("lyn_R_neutral"), SP.misalign(refR), name + "_R")
        PL = SM.Pair(config.data("lyn_L_neutral"), refL, name + "_L")
        D = SM.estimate_direction(PR, n_boot=60)
        I = SM.estimate_interior(PR, D)
        best = None
        for phi in range(0, 360, 15):
            for wf in (0.005, 0.012, 0.025):
                m = SM.fit_model(PR, phi, wf, 1.0)
                e = MX.score_practice(SM.render(m, PR.n, PR.geo), PR.r, PR.valid, PR.geo.H)["dE_mean"]
                if best is None or e < best[0]:
                    best = (e, phi, wf, m)
        e_self, phi_fit, wf, model = best
        cross = MX.score_practice(SM.render(model, PL.n, PL.geo), PL.r, PL.valid, PL.geo.H)
        ident = MX.score_practice(PL.n, PL.r, PL.valid, PL.geo.H)
        return dict(phi=D["phi"], phi_linear=D["phi_linear"], grid=phi_fit, front=I["front_index"], cross=cross["dE_mean"],
                    ident=ident["dE_mean"], self=float(e_self))
    return config.cached_json(f"t0_practice_synth_{name}", make, sub="t0")


def _practice_swap(name, phi):
    """R_practice model2.py main (other character + other direction: recoloured left neutral) for one scene."""
    SM, M2 = protos.load("R_practice", "scenemodel"), protos.load("R_practice", "model2")
    def make():
        sc = SP.SCENES[name]
        nR, nL = SM.load_rgba(config.data("lyn_R_neutral")), SM.load_rgba(config.data("lyn_L_neutral"))
        nLs, sw = SP.recolour_swap(nL)
        PR = SM.Pair(config.data("lyn_R_neutral"), SP.light_scene(nR, sc), name)
        PL = SM.Pair(config.data("lyn_L_neutral"), SP.light_scene(nLs, sc), name + "_swapL")
        m = M2.fit2(PR, phi, 0.012)
        out = M2.render2(m, nLs, PL.geo)
        s_sw = MX.score_practice(out, PL.r, PL.valid & sw, PL.geo.H)
        s_all = MX.score_practice(out, PL.r, PL.valid, PL.geo.H)
        PL0 = SM.Pair(config.data("lyn_L_neutral"), SP.light_scene(nL, sc), name + "_L")
        s0 = MX.score_practice(M2.render2(m, nL, PL0.geo), PL0.r, PL0.valid, PL0.geo.H)
        return dict(recoloured=s_sw["dE_mean"], all=s_all["dE_mean"], same_char=s0["dE_mean"])
    return config.cached_json(f"t0_practice_swap_{name}", make, sub="t0")


def _practice_real():
    """R_practice est_pairs.py: direction on the real pairs (aligned reference passed through Pair, as there)."""
    SM = protos.load("R_practice", "scenemodel")
    def make():
        out = {}
        for k, side in (("right", "R"), ("left", "L")):
            P = SM.Pair(config.data(data.LYN[side][0]), data.lyn(side)["ref"], k)
            D = SM.estimate_direction(P, n_boot=200)
            out[k] = dict(phi=D["phi"], phi_linear=D["phi_linear"], ci=D["phi_ci"])
            # R_practice xfer.py grid (dE over phi at that pair's best w, p from models.json)
            mj = json.load(open(_src("R_practice", "models.json")))[k]["full"]
            best = None
            for phi in range(-10, 101, 10):
                m = SM.fit_model(P, phi, mj["w"], mj["p"])
                e = MX.score_practice(SM.render(m, P.n, P.geo), P.r, P.valid, P.geo.H)["dE_mean"]
                if best is None or e < best[0]:
                    best = (e, phi)
            out[k]["grid"] = best[1]
        return out
    return config.cached_json("t0_practice_real", make, sub="t0")


def r_practice(jobs=2):
    rep = json.load(open(_src("R_practice", "synth_report.json")))
    names = ("midday", "night_moon", "night_torch")
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(jobs, 3)) as p:
            res = dict(zip(names, p.map(_practice_synth, names)))
    else:
        res = {n: _practice_synth(n) for n in names}
    for n in names:
        r, s = res[n], rep[n]
        row("R_practice synth_report.json", f"{n}: azimuth, clamped lobe (J&F)", "R_practice", "deg", s["dir"]["phi"], r["phi"], angle=True)
        row("R_practice synth_report.json", f"{n}: azimuth, linear init", "R_practice", "deg", s["dir"]["phi_linear"], r["phi_linear"], angle=True)
        row("R_practice synth_report.json", f"{n}: azimuth, grid fit", "R_practice", "deg", s["phi_fit"], r["grid"], angle=True)
        row("R_practice synth_report.json", f"{n}: front index", "R_practice", "front", s["interior"]["front_index"], r["front"])
        row("R_practice synth_report.json", f"{n}: identity dE (left truth)", "R_practice", "dE", s["ident"]["dE_mean"], r["ident"])
        row("R_practice synth_report.json", f"{n}: fit R -> apply L dE", "R_practice", "dE", s["cross"]["dE_mean"], r["cross"])
    est = json.load(open(_src("R_practice", "est_pairs.json")))
    rr = _practice_real()
    models = json.load(open(_src("R_practice", "models.json")))
    for k in ("right", "left"):
        row("R_practice est_pairs.json", f"real {k}: azimuth, clamped lobe (J&F)", "R_practice", "deg", est[k]["dir"]["phi"], rr[k]["phi"], angle=True)
        row("R_practice est_pairs.json", f"real {k}: azimuth, linear init", "R_practice", "deg", est[k]["dir"]["phi_linear"], rr[k]["phi_linear"], angle=True)
        row("R_practice models.json", f"real {k}: azimuth, grid fit (best w, p)", "R_practice", "deg", models[k]["full"]["phi"], rr[k]["grid"], angle=True)
    # model2 swap (design 1.1: tone + one saturation + tint grade: 7.3 midday / 2.4 moon / 5.5 lantern night)
    for n, phi, src_v in (("midday", 90, 7.3), ("night_moon", 135, 2.4), ("night_torch", 150, 5.5)):
        r = _practice_swap(n, phi)
        row("DESIGN_GENERAL 1.1 (R_practice model2.py)", f"{n}: other character + direction, recoloured parts", "model2",
            "dE", src_v, r["recoloured"], f"all {r['all']:.2f}, same character {r['same_char']:.2f}")


# ------------------------------------------------------------------ I: s598 BASELINE (current tool on P01)
def s598():
    from scenelight.align import Pairs
    from scenelight import relight, transfer, report
    src = {"night_rgba": (0.788, 5.84, 0.936, 1.65), "night_key": (0.783, 6.07, 0.925, 1.65), "lantern_key": (0.801, 12.99, 0.955, 2.75)}
    for v, (sh, ce, shp, cep) in src.items():
        def make(v=v):
            P = data.p01(v)
            pm = MT.get_model(f"p01:{v}", "tool_pack", lambda: CS.fit_pair(f"p01:{v}"))[0]
            pairs = Pairs(P["neutral"], P["ref_raw"])
            m1 = report.metrics(relight(P["neutral"], pm["prof"], "walk34"), pairs)
            tr, _ = transfer.relight_transfer(P["neutral"], pm["prof"], transfer.load_pack(pm["pack_dir"]), "walk34")
            m2 = report.metrics(tr, pairs)
            return {"prof": m1, "pack": m2}
        r = config.cached_json(f"t0_s598_{v}", make, sub="t0")
        cite = "s598/BASELINE.md"
        row(cite, f"P01 {v}: profile only", "tool_profile", "shape", sh, r["prof"]["shape_agreement"])
        row(cite, f"P01 {v}: profile only", "tool_profile", "colour error", ce, r["prof"]["color_error_lowfreq"])
        row(cite, f"P01 {v}: reference pack", "tool_pack", "shape", shp, r["pack"]["shape_agreement"])
        row(cite, f"P01 {v}: reference pack", "tool_pack", "colour error", cep, r["pack"]["color_error_lowfreq"])


def design_azimuth():
    """DESIGN 1.2: R_methods linear estimator 20.6 / 20.3 deg on the real pairs (read from the azimuth comparison)."""
    from . import estimators
    for name, side, src_v in (("lyn_R", "R", 20.6), ("lyn_L", "L", 20.3)):
        r = config.cached_json(f"az_real_{name}", lambda side=side: estimators.estimate(data.lyn(side)["neutral"],
                                                                                        data.lyn(side)["ref"]), sub="azimuth")
        row("DESIGN_GENERAL 1.2 (R_methods estimate_light_dir)", f"real {name}: azimuth, linear (dt normals)", "R_methods",
            "deg", src_v, r["lin_methods_dt"], angle=True,
            note=f"current default normals ('nc') give {r['lin_methods']:.1f}")


NOTES = [
    "R_methods numbers are reproduced on the same aligned references (the bench's alignment is bit-identical to the "
    "R_methods / R_audit / R_practice caches) with metrics.score_rmethods (sigma 0.6 % H, sunrim parts, sun at 40 deg).",
    "R_methods' linear azimuth 20.6 / 20.3 deg (design 1.2) was measured with the distance-transform normals "
    "(common.NORMALS = 'dt'); the current default 'nc' (normalized convolution) gives 22.0 / 27.7 deg.",
    "R_practice real-pair azimuths: the right pair's linear init differs by -0.9 deg (21.5 vs 22.3) and J&F by -0.1 deg; "
    "the left pair is exact. Inputs are bit-identical aligned references; the source JSON predates the last edits of the "
    "research folder, so the residual is not traced further (within tolerance).",
    "The model2 rows quote the design (rounded to 0.1); the bench reruns R_practice model2.py's swap test "
    "(fit right pair, render the recoloured left neutral, score the recoloured parts).",
    "Design 2.2 quotes post-flow IoU 0.862 (night) / 0.633 (darkness); those numbers are not in any R_audit file. "
    "The R_audit logs hold the fit alignment IoU 0.936 / 0.775 (reproduced above, rows 'fit alignment').",
    "The tool's fit crash at a 96 px figure height (shading.py:322) is reproduced (status row).",
    "Synthetic truths are cached as float32: with a float16 cache the current tool's midday pack moved from 3.30 to 3.12 "
    "(skin), the tool's colour fit reacts to 1e-4 input changes.",
]

# explanations of reproduction mismatches (filled in after investigating each; keyed by a substring of the setting)
EXPLAIN = {
    "darkness_ref: neutral": "the current tool's pack on a near-black reference reacts to ~1e-4 input differences: the bench "
                             "builds the graded reference from its float32 aligned reference, R_audit from a float16 copy "
                             "(max difference 2.4e-4). On R_audit's own float16 copy the bench reproduces 0.91 / 0.32 / 0.36 "
                             "exactly (rows marked [R_audit's float16 reference copy]); absolute gaps are 0.02-0.08 dE.",
}


def write(path):
    n_ok = sum(r["ok"] for r in ROWS)
    lines = ["# T0 - baselines reproduced by the bench", "",
             f"**{n_ok} / {len(ROWS)}** source numbers reproduced within +-5 % (angles: or within 2 deg). Source = the research "
             "file (or document) the number was reported in, read at run time; bench = the bench's method adapters + copied "
             "renderers + the compat replica of that source's own metric in bench/metrics.py. Canonical bench numbers "
             "(results_all.txt) use the unified definitions (e.g. dEa at sigma 1 % H) and are not comparable 1:1 with these.", "",
             "| # | source | setting | method | quantity | source | bench | delta % | ok | note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: v if isinstance(v, str) else ("nan" if v is None or not np.isfinite(v) else
                                                (f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"))
    for i, r in enumerate(ROWS, 1):
        d = "" if not np.isfinite(r["d"]) else f"{r['d']:+.1f}"
        lines.append(f"| {i} | {r['src']} | {r['setting']} | {r['method']} | {r['qty']} | {f(r['src_v'])} | {f(r['rep_v'])} | "
                     f"{d} | {'yes' if r['ok'] else 'NO'} | {r['note']} |")
    lines += ["", "## Notes", ""] + [f"- {t}" for t in NOTES]
    bad = [(i, r) for i, r in enumerate(ROWS, 1) if not r["ok"]]
    if bad:
        lines += ["", "## Mismatches", ""]
        for i, r in bad:
            why = next((t for k, t in EXPLAIN.items() if k in r["setting"] or k == r["src"]), "not explained yet")
            lines.append(f"- row {i} ({r['src']}, {r['setting']}, {r['qty']}): {why}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return n_ok


def main(jobs=2, log=print):
    ROWS.clear()
    t = time.time()
    for part in (r_methods, r_methods_lomo, r_audit, lambda: r_practice(jobs), s598, design_azimuth):
        try:
            part()
        except Exception as e:
            import traceback
            ROWS.append(dict(src=getattr(part, "__name__", "part"), setting="ERROR", method="", qty="",
                             src_v=float("nan"), rep_v=float("nan"), d=float("nan"), ok=False,
                             note=f"{e.__class__.__name__}: {e} " + " / ".join(traceback.format_exc().splitlines()[-3:])))
    p = os.path.join(config.out_dir(), "T0.md")
    n_ok = write(p)
    config.jdump(os.path.join(config.out_dir(), "t0_rows.json"), ROWS)
    log(open(p).read())
    log(f"T0: {n_ok}/{len(ROWS)} within +-5 %  ({time.time() - t:.0f}s) -> {p}")
    return ROWS
