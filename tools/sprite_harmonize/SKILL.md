---
name: scene-relight
description: 2D 게임 캐릭터 스프라이트/애니메이션 프레임을 배경 장면의 조명에 맞게 다시 칠한다 (색감 + 몸 형태를 따라 휘는 명암 + 역광). 장면마다 손으로 칠한 참고 그림 한 장에서 빛을 한 번 배우고(fit), 참고가 없는 모든 프레임·모든 방향에 적용(apply)하고, 배경 위 합성 미리보기(preview)를 만든다. "스프라이트가 배경과 안 어울린다", "린 빛", "장면 조명 맞추기", "relight sprites", "match sprite lighting to background" 같은 요청에 사용.
---

# scene-relight — 스프라이트를 장면 조명에 맞추기

중립 조명으로 그린 캐릭터 프레임을 장면(예: 숲 석양)의 빛으로 다시 칠한다.
하는 일: 재질별 색 맞춤(피부·흰 천/털/머리·어두운 천) → 몸 전체 윤곽 + 자동 검출한 가슴 모양으로 면 방향을 만들고
장면 빛(주광 + 역광)으로 명암 → 얇은 털 가장자리 역광 → (미리보기) 발밑 그림자·가장자리 빛 번짐.

## 준비 (한 번)
```bash
cd <이 스킬 폴더>
pip install -r requirements.txt        # numpy, opencv-python-headless, pillow, scipy
```

## 작업 흐름

### 1) 장면마다 한 번: 빛 배우기 (fit)
같은 포즈·같은 구도의 두 장이 필요하다.
- `neutral.png`: 지금 게임에 쓰는 중립 조명 프레임 (투명 배경 PNG/WebP)
- `reference.png`: 그 프레임을 그 장면 빛으로 손으로(또는 AI로) 칠한 그림 (투명 배경)
```bash
python relight.py fit neutral.png reference.png -o scenes/<장면이름>.json --sheet check_<장면이름>.png
```
- 1~2분 걸린다. 출력의 `alignment IoU` 가 0.85 미만이면 두 그림의 포즈/구도가 달라서 결과를 믿기 어렵다 → 사용자에게 알린다.
- `shape agreement`(명암 모양 일치도, 1이 완전 일치)가 neutral 값보다 올라가야 하고, `mean L*` 가 reference 와 ±2 안이어야 한다.
- `check_*.png` 를 사용자에게 보여 준다 (중립 | 새 식 | 참고, 윗몸 확대, 부위 안 명암 지도: 빨강 = 밝음, 파랑 = 어두움).

이미 있는 장면: `scenes/forest_sunset.json` (숲 석양, 린. 참고 = 사용자가 만든 따뜻한 참고 그림).

### 2) 모든 프레임에 적용 (apply)
```bash
python relight.py apply scenes/forest_sunset.json frames/*.png -o out/forest_sunset/ --view-from-name --sheet before_after.png
```
- 프레임당 1~3초. 알파(투명도)는 절대 바꾸지 않는다. 결과는 RGBA PNG.
- `--view-from-name`: 파일 이름에서 방향을 읽는다. `front`/`s`/`남`, `front_left`/`sw`/`남서`, `front_right`/`se`/`남동`,
  `left`/`w`/`서`, `right`/`e`/`동`, `back`/`n`/`북`, `back_left`/`nw`, `back_right`/`ne`. 못 읽으면 `--view` 값(기본: 정면·3/4 취급).
- 방향이 중요한 이유: 뒤(`back*`)는 가슴 모양을 넣지 않고, 옆(`left`/`right`)은 하나만 넣는다.
- 한 방향짜리 묶음이면 `--view back` 처럼 직접 준다.

### 3) 게임 화면 미리보기 (preview)
```bash
python relight.py preview scenes/forest_sunset.json frames/walk_03.png --bg background.png --place 640,602,240 -o shot.png
```
- `--place 발중심x,발y,스프라이트높이` (배경 픽셀 단위). 이미 relight 한 PNG 면 `--already-relit`.
- `--shadow`(발밑 그림자 0.55) `--wrap`(가장자리 빛 번짐 0.35) `--dim`(장면 안 밝기 0.95).
- 게임 엔진에서는 relight 된 프레임을 쓰고, 발밑 그림자·빛 번짐은 엔진 쪽에서 같은 값으로 넣는다.

### 4) 결과 보고
사용자에게는 말로만 "됐다"고 하지 말고 이미지(before_after, check sheet, preview)와 숫자를 보여 준다.

## 규칙 / 주의
- **최종 렌더된 프레임에 적용한다.** lyn-sprites 리그의 원화 파츠(`assets/parts/`)에는 적용하지 않는다 (그 저장소는 "색칠·재채색 금지" 규칙이 있다).
  리그가 굽은 프레임(스트립/GIF 원본 PNG)이나 AI가 그린 동작 프레임에 쓴다.
- 장면이 바뀌면 그 장면의 참고 그림 한 장으로 `fit` 을 새로 한다. 다른 장면 프로필을 돌려쓰지 않는다.
- 픽셀아트(도트)용이 아니다. 도트에 쓰면 중간색이 생긴다.
- 가슴 모양은 윗몸의 밝은 흰 천(윗옷) 덩어리로 찾는다. 흰 윗옷이 없으면 가슴 모양 없이 몸 윤곽만으로 명암을 넣는다 (깨지지는 않는다).
- 실패하거나 이상하면 숨기지 말고 이미지와 함께 사실대로 알린다.

## 테스트
```bash
python -m pytest -q tests      # 합성 인형으로 13개 불변식 (알파 보존, 결정성, 빛 방향 학습, 방향 힌트 ...)
```
