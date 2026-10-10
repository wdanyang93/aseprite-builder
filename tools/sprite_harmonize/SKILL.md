---
name: scene-relight
description: 2D 게임 캐릭터 스프라이트/애니메이션 프레임을 배경 장면의 조명에 맞게 다시 칠한다 (색감 + 몸 형태를 따라 휘는 명암 + 역광). 장면마다 손으로 칠한 참고 그림 한 장에서 빛을 한 번 배우고(fit), 참고가 없는 모든 프레임·모든 방향에 적용(apply)하고, 배경 위 합성 미리보기(preview)를 만든다. "스프라이트가 배경과 안 어울린다", "린 빛", "장면 조명 맞추기", "relight sprites", "match sprite lighting to background" 같은 요청에 사용.
---

# scene-relight — 스프라이트를 장면 조명에 맞추기

> 이 폴더를 처음 받은 AI 는 먼저 `HANDOFF.md` 를 읽는다 (지금 상태, 하지 말 것, 확인 방법).

중립 조명으로 그린 캐릭터 프레임을 장면(예: 숲 석양)의 빛으로 다시 칠한다.
하는 일: 재질별 색 맞춤(피부·흰 천/털/머리·어두운 천) → 장면 빛을 **더한다**: 그림 전체의 완만한 밝기 기울기(워시) +
실루엣 가장자리의 얇은 림(주광 쪽·역광 쪽). 몸 안쪽에는 어떤 모양도 새로 만들지 않고 원화의 명암·가슴 모양을 그대로 둔다
→ 얇은 털 가장자리 역광 → (미리보기) 발밑 그림자·가장자리 빛 번짐.

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
python relight.py fit neutral.png reference.png -o scenes/<장면이름>.json --pack scenes/<장면이름>.pack --sheet check_<장면이름>.png
```
- `--pack` = **참고 팩**: 참고 그림의 명암·색을 그대로 옮기기 위한 데이터(중립 그림 + 명암 차이 지도). 사용자가 원하는 "참고처럼"은 이것으로 한다.
  팩에는 사용자의 그림이 들어 있다 → 공개 저장소에 올리지 않는다 (`.gitignore` 처리됨).
- 1~2분 걸린다. 출력의 `alignment IoU` 가 0.85 미만이면 두 그림의 포즈/구도가 달라서 결과를 믿기 어렵다 → 사용자에게 알린다.
- `shape agreement`(명암 모양 일치도, 1이 완전 일치)가 neutral 값보다 올라가야 하고, `mean L*` 가 reference 와 ±2 안이어야 한다.
- `drawn form kept`(원화 모양 보존)는 0.95 이상이어야 한다. 낮으면 그림의 가슴 모양이 달라 보인다 → 사용자에게 알리고 시트를 보여 준다.
- `check_*.png` 를 사용자에게 보여 준다 (중립 | 새 식 | 참고, 윗몸 확대, 부위 안 명암 지도: 빨강 = 밝음, 파랑 = 어두움).
  `--pack` 이면 `check_*_pack.png` 도 나온다. 팩 결과: 린 숲 석양 `with pack: shape agreement 0.900`,
  `soft transfer (body): drawn form kept 0.953`, `chest stage: on`.
  `soft transfer (body)` 값이 0.95 미만이면 경고가 나온다 → 사용자에게 알린다. 가슴 단계 줄의 값(약 0.90)은 가슴 안쪽이 일부러
  참고의 빛을 따르기 때문에 낮은 것이다(사용자 요구).

이미 있는 장면: `scenes/forest_sunset.json` (숲 석양, 린. 참고 = 사용자가 만든 따뜻한 참고 그림).
참고 팩 `scenes/forest_sunset.pack/` 은 사용자에게 따로 받은 zip 에만 들어 있다 (공개 저장소에는 없다).

### 2) 모든 프레임에 적용 (apply)
```bash
python relight.py apply scenes/forest_sunset.json frames/*.png -o out/forest_sunset/ --pack scenes/forest_sunset.pack --view-from-name --sheet before_after.png
```
- **팩이 있으면 항상 `--pack` 을 쓴다.** 먼저 모든 프레임을 중립 그림과 맞춰 보고(관문: 접힘 ≤ 0.055, 구조 일치 ≥ 0.68,
  흐름 전 실루엣 겹침 ≥ 0.71), 방향 묶음마다 80% 이상 통과하면 통과한 프레임에 참고 빛을, 아니면 그 묶음 전체를 워시+림으로 한다.
  관문을 못 넘은 프레임은 절대 참고 빛을 받지 않는다. 출력에 `view=..: 92% of 12 frames pass ...` 와 프레임마다
  `reference 95% (fold .., match .., iou ..)  chest light: on` 또는 `wash+rim fallback (...)` 이 찍힌다 → 사용자에게 알린다. 프레임마다 따로 정하려면 `--per-frame`.
- **가슴 단계** (팩 모드 기본): 참고 빛을 받는 프레임은 가슴 구역에서 참고의 빛을 가는 크기로 더 받는다. 하이라이트·그림자가 그려진 선
  (골 선, 컵 윤곽)에서 끊기고, 컵 무늬 대비는 40%로 낮춰 형태 빛이 읽힌다. 선은 밝아지지 않고 구역 밖은 그대로다.
  컵을 못 찾으면 `chest light: skipped (이유)` → 그 프레임은 가슴 단계 없이 저장된다. 끄려면 `--no-chest`.
  결과를 보고할 때 가슴 확대(전 | 후 | 참고)를 꼭 보여 준다 — 사용자가 가장 신경 쓰는 곳이다.
- 방향마다 참고 그림이 하나씩 있으면 방향마다 팩을 만들어 쓴다 (예: `forest_sunset_se.pack`). 참고가 없는 방향은 워시+림이다.
- 팩 없이: 프레임당 1~3초. 팩 사용: 프레임당 약 5초. 프레임 하나가 실패해도 나머지는 저장된다. 알파(투명도)는 절대 바꾸지 않는다.
- 그림 높이 약 200 px 미만의 아주 작은 스프라이트, 허리에서 잘린 윗몸 그림은 맞추기가 안 되어 워시+림이 된다 (안전한 쪽).
- `--view-from-name`: 파일 이름에서 방향을 읽는다. `front`/`s`/`남`, `front_left`/`sw`/`남서`, `front_right`/`se`/`남동`,
  `left`/`w`/`서`, `right`/`e`/`동`, `back`/`n`/`북`, `back_left`/`nw`, `back_right`/`ne`. 못 읽으면 `--view` 값(기본: 정면·3/4 취급).
- 한 방향짜리 묶음이면 `--view back` 처럼 직접 준다.

### 3) 게임 화면 미리보기 (preview)
```bash
python relight.py preview scenes/forest_sunset.json frames/walk_03.png --bg background.png --place 640,602,240 -o shot.png
```
- `--place 발중심x,발y,스프라이트높이` (배경 픽셀 단위). 이미 relight 한 PNG 면 `--already-relit`. 팩이 있으면 `--pack` 도 준다.
- `--shadow`(발밑 그림자 0.55) `--wrap`(가장자리 빛 번짐 0.35) `--dim`(장면 안 밝기 0.95).
- 게임 엔진에서는 relight 된 프레임을 쓰고, 발밑 그림자·빛 번짐은 엔진 쪽에서 같은 값으로 넣는다.

### 4) 결과 보고
사용자에게는 말로만 "됐다"고 하지 말고 이미지(before_after, check sheet, preview)와 숫자를 보여 준다.

## 규칙 / 주의
- **최종 렌더된 프레임에 적용한다.** lyn-sprites 리그의 원화 파츠(`assets/parts/`)에는 적용하지 않는다 (그 저장소는 "색칠·재채색 금지" 규칙이 있다).
  리그가 굽은 프레임(스트립/GIF 원본 PNG)이나 AI가 그린 동작 프레임에 쓴다.
- 장면이 바뀌면 그 장면의 참고 그림 한 장으로 `fit` 을 새로 한다. 다른 장면 프로필을 돌려쓰지 않는다.
- 픽셀아트(도트)용이 아니다. 도트에 쓰면 중간색이 생긴다.
- 원화의 모양(특히 가슴 가운데)을 바꾸면 안 된다는 것이 사용자 요구다. 가슴 타원체(`geo.bh > 0`)와 옛 빛 모델(`cfg.model = "normals"`)은
  모양을 바꿔 보이게 해서 꺼 두었다 (RESEARCH.md 1·9번). 켜지 않는다.
- 가슴의 입체감은 참고 그림의 실제 빛을 선 안쪽에서만 옮겨서 낸다(가슴 단계, RESEARCH.md 12번). 빛 모양을 지어내는 방법
  (부풀림 조명, 둥근 빛 덩어리)은 검토에서 탈락했다.
- Adobe 로 같은 "영역 선택 + 밝기 보정"을 할 수 있지만 Adobe 가 가장자리 알파를 바꾼다 → 결과의 알파는 원본으로 되돌린다 (HANDOFF.md).
- 실패하거나 이상하면 숨기지 말고 이미지와 함께 사실대로 알린다.

## 테스트
```bash
python -m pytest -q tests      # 합성 인형으로 22개 불변식 (알파 보존, 결정성, 빛 방향, 원화 모양 보존, 몸 안쪽 무모양, 참고 팩: 참고에 가까워짐·다른 그림은 완전 대체·위치 이동 따라감, 가슴 단계: 참고에 가까워짐·선은 밝아지지 않음·구역 밖 그대로·컵 없으면 그대로 ...)
```
