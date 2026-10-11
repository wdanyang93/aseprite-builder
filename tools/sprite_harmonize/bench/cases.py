"""Case registry. A case = one FIT pair (neutral + reference in the neutral's geometry) + test FRAMES with ground truth in
the frame's own geometry (or no truth: eye / truth-free checks only).

Groups
  real      lyn_R, lyn_L (self), cross_RL / cross_LR (fit one direction, truth = the other direction's reference),
            P01 night / lantern with both mattes (_rgba, _key), turn_sunset (real sunset on the turnaround, no truth)
  synth     syn_<scene>_<renderer>: a known scene rendered on EVERY drawing by one of the three research renderers
            (practice / methods / audit); fit on Lyn right, test on Lyn L, P01, P02, P03 and the 8 turnaround views
  recolour  rec_*: R_audit von Kries recolours (outfit midblue / navy / black / grey / steel / red, dark skin, pink hair,
            brown lines) of the real right pair -> consistent ground truth; the existing (unrecoloured) scene is applied;
            rec_midday_navy / rec_lantern_black: dark outfit fitted under an analytic light (exact truth)
  small     small_<h>: right pair downscaled to figure height h (fit at full size); small_fit_<h>: fit at that size
  frames    frames_lab2 (f1-f4), frames_s1 (g1-g5): validation pseudo frames; frames_heldout_R / _L: NEW test frames
            (bench/frames.py HELDOUT seeds), truth = the reference carried by the ground-truth map

Light azimuth (deg, 0 = screen right, 90 = up) per frame drives the sun / shadow edge metric: the real sunset uses 40
(sunrim convention), synthetic scenes their true key azimuth, P01 a nominal side (night 0 = moon rim strongest on the right,
lantern 180 = lantern at screen-left), from the s598 / peek measurements.
"""
import numpy as np

from . import config, data, frames as FR
from . import synth_practice as SP, synth_methods as SMe, synth_audit as SA
from . import parts as PARTS


class Frame:
    def __init__(self, name, img, truth=None, labels=None, light_deg=None, view=None, same_geometry=False, masks=None,
                 drawing=None, note=""):
        self.name, self._img, self._truth, self._labels, self._masks = name, img, truth, labels, masks
        self.light_deg, self.view, self.same_geometry, self.drawing, self.note = light_deg, view, same_geometry, drawing, note

    @staticmethod
    def _get(x):
        return x() if callable(x) else x

    def img(self):
        return self._get(self._img)

    def truth(self):
        return self._get(self._truth)

    def labels(self):
        return self._get(self._labels)

    def masks(self):
        return self._get(self._masks)

    @property
    def has_truth(self):
        return self._truth is not None


class Case:
    def __init__(self, name, group, fit_id, frames, desc="", methods=None):
        self.name, self.group, self.fit_id, self.frames, self.desc = name, group, fit_id, frames, desc
        self.methods = methods          # None = every method that applies


# ------------------------------------------------------------------ synthetic scenes
# (scene, renderer) -> renderer key, true key azimuth, note
SYNTH = {
    ("midday", "practice"): ("midday", 95.0, "R_practice midday: sun az 95 el 55, sky ambient"),
    ("midday", "audit"): ("midday", 75.0, "R_audit midday: sun az 75 el 35, L* contrast curve 1.25"),
    ("sunset", "practice"): ("sunset", 40.0, "bench sunset in the R_practice renderer: warm key az 40 el 5"),
    ("sunset", "audit"): ("sunset_like", 10.0, "R_audit sunset_like: warm key az 10 el 10"),
    ("moon", "practice"): ("night_moon", 140.0, "R_practice night_moon: moon az 140 el -25 (behind), Purkinje"),
    ("moon", "methods"): ("moon", 135.0, "R_methods night without the lantern: moon az 135 el 30"),
    ("moon", "audit"): ("night_moon", 140.0, "R_audit night_lantern without the lantern: moon az 140 el 15"),
    ("darkness", "practice"): ("darkness", 120.0, "bench darkness in the R_practice renderer: L* 2..30"),
    ("darkness", "audit"): ("darkness_ref", 40.0, "R_audit darkness_from_ref: the real sunset reference re-graded"),
    ("nightref", "audit"): ("night_ref", 40.0, "R_audit night_from_ref: the real sunset reference re-graded"),
    ("moonlantern", "practice"): ("night_torch", 140.0, "R_practice night_torch: moon az 140 + torch right-low"),
    ("moonlantern", "methods"): ("moon_lantern", 135.0, "R_methods night: moon az 135 + lantern lower right"),
    ("moonlantern", "audit"): ("night_lantern", 140.0, "R_audit night_lantern: moon az 140 + lantern right at hip"),
    ("twokeys", "practice"): ("two_keys", 30.0, "bench two directional lights: warm az 30 + cool az 160"),
    ("twokeys", "audit"): ("two_keys", 25.0, "bench two directional lights: warm az 25 + cool az 165"),
}
SECOND_KEY = {("twokeys", "practice"): 160.0, ("twokeys", "audit"): 165.0}
DRAWINGS = ["lyn_R", "lyn_L", "P01", "P02", "P03"] + [f"turn_{v}" for v in data.TURN_VIEWS]
GRADED = ("darkness_ref", "night_ref")          # need a real reference: only the Lyn drawings


def render(renderer, scene, N, drawing=None):
    if renderer == "practice":
        return SP.light_scene(N, SP.SCENES[scene])
    if renderer == "methods":
        return SMe.night(N, lantern=None) if scene == "moon" else SMe.night(N)
    if renderer == "audit":
        if scene in GRADED:
            side = {"lyn_R": "R", "lyn_L": "L"}[drawing]
            R = data.lyn(side)["ref"]
            return SA.night_from_ref(R) if scene == "night_ref" else SA.darkness_from_ref(R)
        return SA.lit(N, scene)
    raise KeyError(renderer)


def synth_truth(renderer, scene, drawing):
    """Ground truth of a synthetic scene on a drawing (cached)."""
    # float32: the current tool's fit reacts to 1e-4 input differences (a float16 truth moved a T0 number by 5 %)
    return config.cached_array(f"{renderer}_{scene}_{drawing}", lambda: render(renderer, scene, data.drawing(drawing), drawing),
                               sub="synth32", f32=True)


# ------------------------------------------------------------------ fit pairs
def fit_pair(fit_id):
    """dict(neutral, ref (neutral geometry), ref_raw (what the tool's own fit gets), neutral_path, drawing, view)."""
    def make():
        kind, _, rest = fit_id.partition(":")
        if kind == "lyn":
            L = data.lyn(rest)
            return {"neutral": L["neutral"], "ref": L["ref"], "ref_raw": L["ref_raw"], "neutral_path": L["neutral_path"],
                    "drawing": f"lyn_{rest}", "iou": L["iou"]}
        if kind == "p01":
            P = data.p01(rest)
            return {"neutral": P["neutral"], "ref": P["ref"], "ref_raw": P["ref_raw"], "neutral_path": P["neutral_path"],
                    "drawing": "P01", "iou": P["iou"]}
        if kind == "syn":
            renderer, scene, dr = rest.split("/")
            N = data.drawing(dr)
            G = synth_truth(renderer, scene, dr)
            return {"neutral": N, "ref": G, "ref_raw": G, "neutral_path": data.drawing_path(dr), "drawing": dr}
        if kind == "recsyn":
            scene, outfit = rest.split("/")
            N, _ = recolour_frame(outfit, "R")
            G = config.cached_array(f"recsyn_{scene}_{outfit}", lambda: SA.lit(N, scene), sub="synth32", f32=True)
            return {"neutral": N, "ref": G, "ref_raw": G, "neutral_path": None, "drawing": "lyn_R"}
        if kind == "small":
            h = int(rest)
            Fs, Gs = small_frame(h)
            return {"neutral": Fs, "ref": Gs, "ref_raw": Gs, "neutral_path": None, "drawing": "lyn_R"}
        raise KeyError(fit_id)
    return data.memo(("fit", fit_id), make)


# ------------------------------------------------------------------ recolours (R_audit cases + bench additions)
NAVY, MIDBLUE, BLACK = (0.04, 0.07, 0.30), (0.10, 0.25, 0.75), (0.03, 0.03, 0.035)
DARKSKIN = (0.35, 0.22, 0.15)
RECOLOURS = {
    # name: (mask key, k, second (mask, k) or None)
    "outfit_midblue": ("wcloth", MIDBLUE, None),
    "outfit_navy": ("wcloth", NAVY, None),
    "outfit_black": ("wcloth", BLACK, ("dcloth", (0.35, 0.35, 0.4))),
    "darkskin": ("skin_all", DARKSKIN, None),
    "outfit_grey": ("wcloth", (0.32, 0.32, 0.33), None),
    "outfit_steel": ("wcloth", (0.28, 0.33, 0.42), ("dcloth", (0.8, 0.9, 1.15))),
    "outfit_red": ("wcloth", (0.80, 0.07, 0.05), None),
    "hair_pink": ("fur", (1.0, 0.45, 0.62), None),
    "lines_brown": ("line", (5.0, 3.2, 2.2), None),
}


def recolour_frame(name, which="R"):
    """(recoloured neutral, recoloured aligned reference) of the right pair: a consistent pair (von Kries commutes
    with any per-channel light)."""
    def make():
        L = data.lyn("R")
        M = data.audit_masks()
        mk, k, second = RECOLOURS[name]
        N, R = SA.recolour(L["neutral"], M[mk], k), SA.recolour(L["ref"], M[mk], k)
        if second:
            N, R = SA.recolour(N, M[second[0]], second[1]), SA.recolour(R, M[second[0]], second[1])
        return N, R
    return data.memo(("recolour", name), make)


def small_frame(h):
    def make():
        L = data.lyn("R")
        Fs, k = SA.downscale(L["neutral"], h)
        Gs, _ = SA.downscale(L["ref"], h, like=L["neutral"])
        return Fs, Gs
    return data.memo(("small", h), make)


def small_labels(h):
    L = data.lyn("R")
    Fs, _ = small_frame(h)
    return _resize_labels(L["labels"], Fs.shape[:2])


def _resize_labels(lab, shape):
    import cv2
    return cv2.resize(lab, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------------ the registry
def _lyn_frame(side, light=40.0):
    L = lambda: data.lyn(side)
    return Frame(f"lyn_{side}", lambda: L()["neutral"], lambda: L()["ref"], lambda: L()["labels"], light, None,
                 True, None, f"lyn_{side}")


def build():
    C = {}

    def add(c):
        C[c.name] = c

    # (a) real pairs
    add(Case("lyn_R", "real", "lyn:R", [_lyn_frame("R")], "Lyn sunset right: fit on the pair, test on itself"))
    add(Case("lyn_L", "real", "lyn:L", [_lyn_frame("L")], "Lyn sunset left: fit on the pair, test on itself"))
    fr = _lyn_frame("L"); fr.same_geometry = False
    add(Case("cross_RL", "real", "lyn:R", [fr], "fit right pair, apply to the left neutral, truth = left reference"))
    fr = _lyn_frame("R"); fr.same_geometry = False
    add(Case("cross_LR", "real", "lyn:L", [fr], "fit left pair, apply to the right neutral, truth = right reference"))
    for v in data.P01_REFS:
        light = 0.0 if v.startswith("night") else 180.0
        fl = [Frame("P01", lambda v=v: data.p01(v)["neutral"], lambda v=v: data.p01(v)["ref"], None, light, None, True,
                    None, "P01")]
        if v.endswith("rgba"):
            for o in ("P02", "P03"):
                fl.append(Frame(o, lambda o=o: data.drawing(o), None, None, light, "back", False, None, o, "no truth"))
        add(Case(f"P01_{v}", "real", f"p01:{v}", fl, f"P01 clothed Lyn, {v.replace('_', ' ')} reference (s598 matte)"))
    add(Case("turn_sunset", "real", "lyn:R",
             [Frame(f"turn_{t}", lambda t=t: data.drawing(f"turn_{t}"), None, None, 40.0, data.TURN_HINT[t], False, None,
                    f"turn_{t}", "no truth") for t in data.TURN_VIEWS],
             "real sunset (right pair) on the other-outfit turnaround views: eye check, no truth"))

    # (b) synthetic scenes: fit on lyn_R, test on every drawing
    for (scene, rend), (rkey, az, note) in SYNTH.items():
        drs = ["lyn_R", "lyn_L"] if rkey in GRADED else DRAWINGS
        fl = []
        for dr in drs:
            fl.append(Frame(dr, lambda dr=dr: data.drawing(dr), lambda dr=dr, r=rend, s=rkey: synth_truth(r, s, dr),
                            lambda dr=dr: data.drawing_labels(dr, data.drawing(dr).shape[:2]), az,
                            data.TURN_HINT.get(dr[5:]) if dr.startswith("turn_") else None, dr == "lyn_R", None, dr))
        add(Case(f"syn_{scene}_{rend}", "synth", f"syn:{rend}/{rkey}/lyn_R", fl, note))

    # (c) recolours: the existing scene (right pair) on recoloured frames
    for name in RECOLOURS:
        fl = [Frame(name, lambda n=name: recolour_frame(n)[0], lambda n=name: recolour_frame(n)[1],
                    lambda: data.lyn("R")["labels"], 40.0, None, True, data.audit_masks, "lyn_R")]
        add(Case(f"rec_{name}", "recolour", "lyn:R", fl, f"right sunset scene applied to the {name.replace('_', ' ')} recolour"))
    for scene, outfit in (("midday", "outfit_navy"), ("night_lantern", "outfit_black")):
        fid = f"recsyn:{scene}/{outfit}"
        fl = [Frame(outfit, lambda o=outfit: recolour_frame(o)[0],
                    lambda s=scene, o=outfit: fit_pair(f"recsyn:{s}/{o}")["ref"], lambda: data.lyn("R")["labels"],
                    SA.SCENES[scene]["key_dir"][0], None, True, data.audit_masks, "lyn_R")]
        add(Case(f"rec_{scene.split('_')[-1]}_{outfit.split('_')[-1]}", "recolour", fid, fl,
                 f"{outfit} fitted and tested under the analytic {scene} light (R_audit)"))

    # (d) small frames
    for h in (384, 192, 128, 96, 48):
        fl = [Frame(f"small_{h}", lambda h=h: small_frame(h)[0], lambda h=h: small_frame(h)[1], lambda h=h: small_labels(h),
                    40.0, None, False, None, "lyn_R")]
        add(Case(f"small_{h}", "small", "lyn:R", fl, f"right pair fitted at full size, applied at figure height {h} px"))
    for h in (192, 128, 96):
        fl = [Frame(f"small_{h}", lambda h=h: small_frame(h)[0], lambda h=h: small_frame(h)[1], lambda h=h: small_labels(h),
                    40.0, None, True, None, "lyn_R")]
        add(Case(f"small_fit_{h}", "small", f"small:{h}", fl, f"fit and apply at figure height {h} px"))

    # (e) pseudo frames with ground-truth maps
    def pseudo(name):
        side = FR.side_of(name)

        def truth():
            F, mp = FR.get(name)
            return PARTS.carry(data.lyn(side)["ref"], mp)

        def labels():
            F, mp = FR.get(name)
            return PARTS.carry(data.lyn(side)["labels"], mp, nearest=True)
        return Frame(name, lambda: FR.get(name)[0], truth, labels, 40.0, None, False, None, f"lyn_{side}")
    add(Case("frames_lab2", "frames", "lyn:R", [pseudo(n) for n in FR.LAB2], "right pseudo frames f1-f4 (validation)"))
    add(Case("frames_s1", "frames", "lyn:L", [pseudo(n) for n in FR.S1], "left pseudo frames g1-g5 (validation)"))
    add(Case("frames_heldout_R", "frames", "lyn:R", [pseudo(n) for n in FR.HELDOUT if n.startswith("hR")],
             "NEW held-out right frames (seeds 7101-7303)"))
    add(Case("frames_heldout_L", "frames", "lyn:L", [pseudo(n) for n in FR.HELDOUT if n.startswith("hL")],
             "NEW held-out left frames (seeds 8101-8303)"))
    return C


CASES = None


def get(name=None):
    global CASES
    if CASES is None:
        CASES = build()
    return CASES if name is None else CASES[name]


def select(spec):
    """'all', a case name, a group name, or a comma list of those."""
    C = get()
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "all":
            out += list(C)
        elif tok in C:
            out.append(tok)
        else:
            m = [k for k, c in C.items() if c.group == tok or k.startswith(tok)]
            if not m:
                raise KeyError(f"unknown case / group '{tok}' (try: python -m bench list)")
            out += m
    seen = set()
    return [k for k in out if not (k in seen or seen.add(k))]
