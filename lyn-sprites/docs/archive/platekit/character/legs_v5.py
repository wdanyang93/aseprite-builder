"""다리 판정 v5 = 선(lineart) 칸 + 구조적 꼬리 배제 + 명암 라벨.

v4(3구간 교차 검증) 폐기 이유: 꼬리가 "밝은쪽" 그룹에 통째로 들어가 접합부 x중심을
꼬리가 지배했다. #8을 못 고치고 v1에서 맞던 f1·f3·f5를 뒤집었다. 구간을 늘리는 문제가
아니라 꼬리가 섞인 것이 근본 원인.

v5: §3-5 의 "작동함" 목록에 있는 선(L*<=150)으로 칸을 나눈다. 꼬리·근다리·원다리가
각각 하나의 칸이 되고, 칸은 쪼개지지 않으므로 "다리가 몸 중간에서 반대편으로 건너간다"
가 구조적으로 불가능해진다. 3구간 교정 패스 자체가 필요 없다.

꼬리 배제 기준: 다리는 바닥에 닿고 꼬리는 닿지 않는다 (y_max/h).
  실측 꼬리 0.752~0.768 / 다리 0.916~0.988. 임계 0.85 는 양쪽에서 0.08 이상 떨어져 있다.
"""
import sys; 
import numpy as np, cv2
from PIL import Image
from platekit.character.lineart import lineart_mask, interior_cells

FRAMES = 'work/v1_13/frames/f{}.png'
SAT, VAL, BLOB = 22, 230, 150      # 하의 키 (면적 문턱만 300 -> 150; f6 하의 226px)
FLOOR_Y   = 0.85                   # 이 아래까지 닿지 않으면 다리가 아니다
CELL_MIN  = 300                    # 이보다 작은 칸은 선 노이즈
FOOT_BAND = 0.75                   # 발 x 를 재는 구간 (다리 높이 비율)

def load(i):
    a = np.array(Image.open(FRAMES.format(i)).convert('RGBA'))
    return a, a[...,3] > 0

def hip_row(rgba, mask):
    h, _w = mask.shape
    hsv = cv2.cvtColor(rgba[...,:3], cv2.COLOR_RGB2HSV)
    cloth = mask & (hsv[...,1] < SAT) & (hsv[...,2] > VAL)
    n, _l, st, _c = cv2.connectedComponentsWithStats(cloth.astype(np.uint8), 8)
    cand = [(st[k, cv2.CC_STAT_AREA], int(st[k, cv2.CC_STAT_TOP]+st[k, cv2.CC_STAT_HEIGHT]))
            for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= BLOB
            and 0.45 <= (st[k, cv2.CC_STAT_TOP]+st[k, cv2.CC_STAT_HEIGHT])/h <= 0.66]
    return max(cand)[1] if cand else None

def solve(i):
    rgba, mask = load(i); h, w = mask.shape
    hip  = hip_row(rgba, mask)
    lab_l = cv2.cvtColor(rgba[...,:3], cv2.COLOR_RGB2LAB)[...,0].astype(float)
    lines = lineart_mask(rgba, mask)
    labels, n = interior_cells(mask, lines)
    legs = mask.copy(); legs[:hip] = False
    ybot = int(np.nonzero(mask)[0].max()); legH = ybot - hip + 1

    cells = []
    for c in range(1, n):
        sel = (labels == c) & legs
        px = int(sel.sum())
        if px < CELL_MIN: continue
        ys, xs = np.nonzero(sel)
        cells.append(dict(id=c, mask=sel, px=px, ymax=ys.max()/h, ymin=ys.min()/h,
                          xc=xs.mean()/w, L=float(np.median(lab_l[sel]))))
    tail = [c for c in cells if c['ymax'] < FLOOR_Y and c['px'] > 2000]
    legc = [c for c in cells if c not in tail]
    legc.sort(key=lambda c: -c['px'])
    # 발 구간 x (칸 중심이 아니라 발 위치로 앞뒤를 판정)
    fy = hip + int(round(FOOT_BAND*legH))
    for c in legc:
        f = c['mask'].copy(); f[:fy] = False
        c['foot_x'] = float(np.nonzero(f)[1].mean())/w if f.any() else None
    # 명암 두 그룹: 가장 큰 칸 2개를 씨앗으로 잡고 나머지 칸은 L* 가 가까운 쪽에 붙인다
    if len(legc) >= 2:
        a, b = legc[0], legc[1]
        bright, dark = (a, b) if a['L'] >= b['L'] else (b, a)
        gb = bright['mask'].copy(); gd = dark['mask'].copy()
        extra = []
        for c in legc[2:]:
            if abs(c['L']-bright['L']) <= abs(c['L']-dark['L']): gb |= c['mask']; extra.append((c['id'],'bright'))
            else: gd |= c['mask']; extra.append((c['id'],'dark'))
    else:
        gb = legc[0]['mask'] if legc else np.zeros_like(mask); gd = np.zeros_like(mask)
        bright, dark, extra = (legc[0] if legc else None), None, []
    tail_m = np.zeros_like(mask)
    for c in tail: tail_m |= c['mask']
    # 선 픽셀 최근접 배정 (칸 사이 경계선이 구멍으로 남지 않게)
    assigned = gb | gd | tail_m
    hole = legs & ~assigned
    if hole.any():
        _d, idx = cv2.distanceTransformWithLabels((~assigned).astype(np.uint8), cv2.DIST_L2, 3,
                                                  labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(assigned); code = np.zeros(idx.max()+1, np.uint8)
        src = np.where(gb[ys,xs], 1, np.where(gd[ys,xs], 2, 3))
        code[idx[ys,xs]] = src
        m = code[idx]
        gb |= hole & (m==1); gd |= hole & (m==2); tail_m |= hole & (m==3)
    def fx(m):
        f = m.copy(); f[:fy] = False
        return float(np.nonzero(f)[1].mean())/w if f.any() else None
    return dict(frame=i, rgba=rgba, mask=mask, legs=legs, hip=hip, hip_y=hip/h, ybot=ybot,
                legH=legH, h=h, w=w, tail=tail_m, bright=gb, dark=gd,
                bright_L=bright['L'] if bright else None, dark_L=dark['L'] if dark else None,
                bright_fx=fx(gb), dark_fx=fx(gd), cells=cells, tail_cells=len(tail),
                leg_cells=len(legc), extra=extra, foot_row=fy,
                covered=int((gb|gd|tail_m).sum()), total=int(legs.sum()))

if __name__ == '__main__':
    from platekit.character.gait import cycle_plan
    plan = cycle_plan()
    print("f  힙선 다리칸 꼬리칸  밝은칸L* 어두운칸L*  Δ   밝은발x 어두운발x  앞선쪽  | 규칙 near  일치?")
    for i in range(1,9):
        d = solve(i); p = plan[i-1]
        lead = "밝은쪽" if d['bright_fx'] > d['dark_fx'] else "어두운쪽"
        rule = "앞" if p['near_foot_x'] > 0 else ("뒤" if p['near_foot_x'] < 0 else "중간")
        if rule == "중간": agree = "—(passing)"
        else: agree = "일치" if ((lead=="밝은쪽") == (rule=="앞")) else "X 불일치"
        print(f"f{i} {d['hip']:4d}   {d['leg_cells']}    {d['tail_cells']}     "
              f"{d['bright_L']:5.1f}    {d['dark_L']:5.1f}  {d['bright_L']-d['dark_L']:5.1f}"
              f"   {d['bright_fx']:.3f}   {d['dark_fx']:.3f}   {lead:6s} | {p['near_foot_x']:+.3f} {rule:4s} {agree}")
        assert d['covered'] == d['total'], (i, d['covered'], d['total'])
    print("\n모든 프레임: 꼬리+근다리+원다리 = 다리영역 전체 (미배정 0px)")

# 8프레임 궤적 적합으로 확정된 라벨 교정 (전수 256가지 탐색)
#   뒤집기 없음  상관 +0.240
#   f5·f8 뒤집기 상관 +0.988   <- 채택
# f3·f7 은 passing 이라 부호가 무의미하다(뒤집어도 RMS 불변). 최소 교정은 {5, 8}.
LABEL_SWAP_FRAMES = frozenset({5, 8})
