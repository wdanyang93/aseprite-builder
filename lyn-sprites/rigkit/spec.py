"""좌표 규약 — 모든 모듈이 공유한다.

리깅은 '그림을 분석해 자세를 알아내는 것'이 아니라 '뼈대가 자세를 만들고 그림은 따라간다'.
그래서 규약은 딱 두 가지만 고정하면 된다: 캔버스 크기와 기준 포즈(A-pose)에서의 관절 위치.
"""
from __future__ import annotations

# 작업 캔버스 (파츠 시트·리그·미리보기가 모두 이 좌표계)
CANVAS = (512, 640)          # w, h
BODY_H = 560                 # 정수리→발바닥 (꼬리 제외)
CENTER_X = 256               # 몸 중심선
SOLE_Y = 620                 # 발바닥 y
TOP_Y = SOLE_Y - BODY_H      # 정수리 y = 60

# 몸축 t = (y - TOP_Y) / BODY_H,  0 = 정수리, 1 = 발바닥
def t_to_y(t: float) -> float:
    return TOP_Y + t * BODY_H

def y_to_t(y: float) -> float:
    return (y - TOP_Y) / BODY_H

# 기준 포즈(A-pose)에서 관절이 놓이는 t 값. 8등신이 아니라 이 캐릭터(약 6.5등신) 기준.
JOINT_T = {
    # 6.5등신 기준: 머리 높이 = H/6.5 ≈ 86px
    "head_top":  0.00,
    "head":      0.170,   # 목 = 머리 회전 중심 (정수리 t=0, 턱 t≈0.155)
    "chest":     0.20,
    "shoulder":  0.205,
    "elbow":     0.345,
    "wrist":     0.480,
    "hips":      0.505,
    "knee":      0.715,
    "ankle":     0.925,
    "sole":      1.00,
}

# 좌우 오프셋 (픽셀, 중심선 기준). 화면 오른쪽이 +x.
SHOULDER_DX = 62
ELBOW_DX = 78
WRIST_DX = 86
HIP_DX = 30
KNEE_DX = 34
ANKLE_DX = 36

# 한 보행 주기 = 두 걸음. 위상 phi in [0,1). 프레임 수.
WALK_FRAMES = 8
