"""Pseudo animation frames with ground-truth maps (frame px -> neutral px).

  lab2 f1-f4      right neutral, from the research frames folder (used to tune the pack: validation only)
  S1 g1-g5        left neutral, dir2 S1_pack frames (validation)
  held-out h*     NEW test frames generated here, never used for tuning anything: same warp model as the research
                  generators (RBF handle warp + similarity, analytic on a padded grid) with seeded jitter of every moved
                  handle (+-12 px) and of the RBF width (130-190 px). Seeds are listed in HELDOUT and never reused.
"""
import os

import numpy as np
import cv2

from scenelight.color import save_rgba

from . import config, data

PAD = 300
# handles on each neutral (x, y); FIXED anchor head / chest / hip / torso sides (research generators)
RIG = {
    "R": {"fixed": [(615, 225), (630, 450), (630, 630), (560, 520), (700, 520)],
          "J": {"fhand": (360, 780), "felbow": (405, 585), "bhand": (870, 765), "belbow": (787, 585), "bknee": (525, 975),
                "bfoot": (465, 1200), "fknee": (660, 1020), "ffoot": (750, 1410), "tail": (90, 960), "head": (615, 225)}},
    "L": {"fixed": [(400, 200), (420, 430), (440, 650), (320, 560), (540, 560)],
          "J": {"fhand": (160, 785), "felbow": (240, 610), "bhand": (712, 795), "belbow": (640, 600), "fknee": (372, 1080),
                "ffoot": (210, 1470), "bknee": (535, 1045), "bfoot": (640, 1320), "tail": (900, 900), "head": (400, 200)}},
}
# name: (side, moves, seed, scale, rot, shift) - all different from f1-f4 / g1-g5 / S3 t1-t5 (seeds 101-505)
HELDOUT = {
    "hR1_wave": ("R", {"fhand": (-50, -40), "felbow": (-20, -15), "bhand": (40, 30), "tail": (30, 35)}, 7101, 0.95, 1.5, (25, 15)),
    "hR2_crouch": ("R", {"ffoot": (40, -45), "fknee": (55, -30), "bfoot": (-35, -30), "bknee": (-45, -20), "head": (-10, 12)},
                   7202, 0.88, -2.0, (-20, 50)),
    "hR3_small": ("R", {"fhand": (30, 20), "bhand": (-30, 15), "tail": (-25, -30)}, 7303, 0.45, 5.0, (40, -30)),
    "hL1_wave": ("L", {"bhand": (50, -45), "belbow": (20, -20), "fhand": (-35, 20), "tail": (-30, 30)}, 8101, 0.93, -1.5, (-25, 20)),
    "hL2_crouch": ("L", {"ffoot": (-40, -40), "fknee": (-50, -25), "bfoot": (40, -35), "bknee": (45, -15), "head": (12, 10)},
                   8202, 0.9, 2.5, (30, 45)),
    "hL3_small": ("L", {"fhand": (-25, 25), "bhand": (25, 20), "tail": (30, -25)}, 8303, 0.5, -5.0, (-35, -25)),
}
LAB2 = ("f1_swing", "f2_small_rot", "f3_stride", "f4_lean")
S1 = ("g1_swing", "g2_small", "g3_stride", "g4_lean", "g5_shift")


def rbf_map(src_pts, dst_pts, shape, sigma):
    h, w = shape
    yy, xx = np.mgrid[-PAD:h + PAD, -PAD:w + PAD].astype(np.float32)
    src, dst = np.array(src_pts, np.float32), np.array(dst_pts, np.float32)
    d = dst[:, None, :] - dst[None, :, :]
    K = np.exp(-(d ** 2).sum(-1) / (2 * sigma ** 2)) + 1e-6 * np.eye(len(dst))
    coef = np.linalg.solve(K, src - dst)
    mx, my = xx.copy(), yy.copy()
    for (qx, qy), (cx_, cy_) in zip(dst, coef):
        k = np.exp(-((xx - qx) ** 2 + (yy - qy) ** 2) / (2 * sigma ** 2))
        mx += k * cx_
        my += k * cy_
    return mx, my


def make_frame(N, rig, moves, seed, scale=1.0, rot=0.0, shift=(0, 0)):
    """Warp neutral N: returns (frame RGBA, map HxWx2 frame px -> neutral px)."""
    H, W = N.shape[:2]
    rng = np.random.default_rng(seed)
    moves = {k: (v[0] + float(rng.uniform(-12, 12)), v[1] + float(rng.uniform(-12, 12))) for k, v in moves.items()}
    sigma = float(rng.uniform(130, 190))
    J = rig["J"]
    fixed = [p for p in rig["fixed"] if not ("head" in moves and p == J["head"])]
    src = fixed + [J[k] for k in moves]
    dst = fixed + [(J[k][0] + moves[k][0], J[k][1] + moves[k][1]) for k in moves]
    mx, my = rbf_map(src, dst, (H, W), sigma)
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2 + shift[0], H / 2 + shift[1]
    ux, uy = (xx - cx) / scale, (yy - cy) / scale
    px, py = (c * ux + s * uy + W / 2).astype(np.float32), (-s * ux + c * uy + H / 2).astype(np.float32)
    fx = cv2.remap(mx, px + PAD, py + PAD, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fy = cv2.remap(my, px + PAD, py + PAD, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    F = cv2.remap(N, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(F, 0, 1).astype(np.float32), np.dstack([fx, fy]).astype(np.float32)


def heldout(name):
    """(frame, map) of a held-out frame, generated once into the cache (frame PNG for viewing + npz)."""
    def make():
        d = config.cache_dir("frames_heldout")
        p = os.path.join(d, name + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            return z["F"].astype(np.float32), z["map"]
        side, moves, seed, scale, rot, shift = HELDOUT[name]
        N = data.lyn(side)["neutral"]
        F, mp = make_frame(N, RIG[side], moves, seed, scale, rot, shift)
        tmp = p + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, F=F, map=mp)
        os.replace(tmp, p)
        save_rgba(os.path.join(d, name + ".png"), F)
        return F, mp
    return data.memo(("heldout", name), make)


def research(name):
    """(frame, map) of an existing research pseudo frame (lab2 f*, S1 g*)."""
    def make():
        folder = config.data("lab2_frames") if name in LAB2 else config.data("s1_frames")
        F = data.ld(os.path.join(folder, name + ".png"))
        return F, np.load(os.path.join(folder, name + "_map.npy"))
    return data.memo(("research", name), make)


def side_of(name):
    if name in LAB2:
        return "R"
    if name in S1:
        return "L"
    return HELDOUT[name][0]


def get(name):
    return heldout(name) if name in HELDOUT else research(name)
