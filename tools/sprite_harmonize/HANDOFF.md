# 인수인계 — scene-relight (다른 Claude 에게 이 파일부터 읽게 한다)

## 한 줄 요약
중립 조명으로 그린 캐릭터 프레임을 장면 빛(지금은 숲 석양)에 맞게 다시 칠하는 도구.
장면마다 참고 그림 한 장으로 빛을 한 번 배우고(`fit`), 모든 프레임에 적용한다(`apply`).

## 지금 상태 (최종 커밋: wash + rim)
- 빛 = **워시**(그림 전체의 완만한 밝기 기울기) + **림**(실루엣 가장자리의 얇은 빛, 목·겨드랑이 같은 오목한 곳은 약하게)
  + 털 가장자리 주황 역광. 색은 재질별(피부 / 흰 천·털·머리 / 어두운 천)로 맞춘다.
- **몸 안쪽에는 어떤 모양도 새로 만들지 않는다.** 원화의 명암(가슴·골·배)은 그대로 둔다.
- 숫자 (린, 참고 대비): 원화 모양 보존 0.995, 몸 안쪽 비평면 빛 0.04 L*, 참고 일치 0.757 (중립 0.599), 평균 L* 66.0 = 참고.

## 왜 이렇게 됐나 (사용자 지적 → 고침)
1. 가슴 타원체를 얹었더니 **가운데 가슴 모양이 바뀌어 보였다** → 타원체 끔 (`geo.bh = 0`).
2. 맞춤이 원화 피부 명암을 −0.33배 **지웠다** → 원화 명암 계수 ≥ 0 (더하기만).
3. 실루엣 부풀림 위 주광이 목→오른쪽 가슴 **대각선 빛 경계**를 만들어 **윗가슴 가운데가 불룩해 보였다**
   → 빛 모델을 워시 + 얇은 림으로 교체 (`cfg.model = "wash"`).
자세한 수치와 다른 실패들: `RESEARCH.md`.

## 절대 하지 말 것
- 가슴 타원체(`geo.bh > 0`)나 옛 빛 모델(`cfg.model = "normals"`)을 다시 켜기.
- 원화 명암을 빼는 방향의 계수 허용.
- 참고와 더 닮게 하려고 몸 안쪽 명암을 다시 조각하기 (두 번 해서 두 번 다 모양이 바뀌었다).
- lyn-sprites 원화 파츠(`assets/parts/`)에 적용하기 — 렌더된 최종 프레임에만 쓴다.

## 확인 방법
```bash
pip install -r requirements.txt
python -m pytest -q tests          # 15개 통과해야 함 (원화 모양 보존, 몸 안쪽 무모양 테스트 포함)
python relight.py apply scenes/forest_sunset.json 프레임/*.png -o 결과/ --view-from-name --sheet 전후.png
```
`fit`/`check` 출력의 `drawn form kept` 가 0.95 미만이면 모양이 바뀐 것 → 사용자에게 이미지와 함께 알린다.

## 파일
`SKILL.md` 사용법 · `README.md` 방법과 표 · `RESEARCH.md` 시도와 실패 · `relight.py` 명령 ·
`scenelight/` 코드 (`shading.py` 빛 모델, `color.py` 색, `align.py` 정렬, `report.py` 지표) · `scenes/forest_sunset.json` 장면 프로필 · `tests/`
원본 저장소: `wdanyang93/aseprite-builder`, 브랜치 `claude/upbeat-goldberg-s58fdw`, 폴더 `tools/sprite_harmonize/`.
