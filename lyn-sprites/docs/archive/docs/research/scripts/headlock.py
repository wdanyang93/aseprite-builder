"""머리를 한 장으로 고정한다 — 정체성은 얼굴에 있다.

왜
    AI 가 뽑은 36장은 몸은 거의 같은데 **얼굴이 미세하게 흔들린다.**
    사람이 가장 먼저 알아보는 것이 얼굴이라 이 흔들림은 눈에 띈다.
    옷을 아무리 잘 옮겨도 얼굴이 프레임마다 다르면 같은 인물로 안 보인다.

실측 — 고정은 거의 공짜다
    누드 36장에서 머리의 자리는 이미 매우 안정적이다.

        머리 중심 x − 시상축    -0.39 ± 0.49 px
        목 y                   145.6 ± 3.10 px
        머리 화소 수            14,532 ± 72 (0.5%)

    즉 **붙일 자리는 이미 정해져 있다.** 얼굴 내용만 한 장으로 통일하면 된다.
    (얼굴 내용의 흔들림은 평균 얼굴 대비 9.4/255 = 3.7%, 옷 입은 시트에서는 18.9)

기준점
    시상면 x (몸통 띠를 접어 대칭이 최대인 x) 와 목 y (t = 0.237).
    머리는 거의 원형이라 주축 각도가 의미 없다 (sd 86.8°). 회전은 쓰지 않는다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/claude/handoff/code")
sys.path.insert(0, "/home/claude/handoff/work")

import gpiece

NECK = 0.237


def head_mask(mask, t, tail=None, neck=NECK):
    h = mask & (t < neck)
    if tail is not None:
        h = h & ~tail
    return h


def head_anchor(mask, t, tail=None, neck=NECK):
    """(시상면 x, 목 y). 붙일 자리."""
    body = mask & ~tail if tail is not None else mask
    sx = gpiece.sagittal_x(body, t)
    ys, xs = np.nonzero(body)
    sel = np.abs(t[ys, xs] - neck) < 0.008
    ny = float(ys[sel].mean()) if sel.sum() > 5 else float(np.percentile(ys, 20))
    return sx, ny


class Head:
    """정본 머리 하나."""

    def __init__(self, rgba, mask, t, tail=None, neck=NECK):
        self.rgba = rgba
        self.mask = head_mask(mask, t, tail, neck)
        self.anchor = head_anchor(mask, t, tail, neck)

    def apply(self, dst_rgba, tgt_mask, tgt_t, tgt_tail=None, neck=NECK, feather=1):
        """대상의 머리를 지우고 정본 머리를 그 자리에 붙인다."""
        import cv2
        out = dst_rgba.copy()
        old = head_mask(tgt_mask, tgt_t, tgt_tail, neck)
        if old.any():
            # 목 경계는 남겨 이음매가 생기지 않게 한다
            cut = old & (tgt_t < neck - 0.004)
            out[cut, 3] = 0
        ax, ay = head_anchor(tgt_mask, tgt_t, tgt_tail, neck)
        dx, dy = ax - self.anchor[0], ay - self.anchor[1]
        ys, xs = np.nonzero(self.mask)
        Y = np.round(ys + dy).astype(int); X = np.round(xs + dx).astype(int)
        ok = (Y >= 0) & (Y < out.shape[0]) & (X >= 0) & (X < out.shape[1])
        out[Y[ok], X[ok], :3] = self.rgba[ys[ok], xs[ok], :3]
        out[Y[ok], X[ok], 3] = self.rgba[ys[ok], xs[ok], 3]
        m = np.zeros(out.shape[:2], bool); m[Y[ok], X[ok]] = True
        if feather:
            a = out[..., 3].astype(np.float32)
            k = np.ones((3, 3), np.uint8)
            ring = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~m
            out[..., 3] = np.where(ring & (a == 0), 0, out[..., 3])
        return out, m
