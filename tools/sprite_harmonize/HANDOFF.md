# 인수인계 — scene-relight (다른 Claude 에게 이 파일부터 읽게 한다)

## 한 줄 요약
2D 캐릭터 프레임을 장면 빛에 맞추는 도구. 한 번 맞추고(작은 JSON) 같은 장면·같은 빛의 모든 프레임에 적용한다.
연구는 끝났다. 쓰는 법은 `SKILL.md` (짧은 실행 순서), 숫자는 `README.md`. 안 되는 경우는 연구하지 않고 그림을 주문한다
(`SKILL.md` 의 비용 규칙).

## 지금 상태 (2026-10-12)
세 방법, 명령은 `relight.py` 하나:

| 방법 | 명령 | 코드 | 입력 |
|---|---|---|---|
| A. 참고 팩 (+ 워시·림 대체) | `fit` / `apply` (`--pack`) | `transfer.py`, `chest.py`, `shading.py`, `align.py` | 중립 + 같은 포즈의 장면 빛 참고 |
| B. 장면 블렌드 | `blend-fit` / `blend` | `blend.py` (267줄) | 프레임 하나 + 화가의 장면 그림 + 두 그림의 Adobe 마스크 |
| C. 참고 룩 | `look-fit` / `look` | `look.py` (약 290줄) | 원본 + 빛 받은 참고 + 원본의 Adobe 마스크 |

- **부위·선은 Adobe 결과만** (`parts.py`): `image_select_subject` bodyParts / `image_select_by_prompt` 마스크와 `image_vectorize`
  SVG 의 가는 어두운 경로. 그림 `X.png` 의 결과는 `masks/X/<부위>.png` + `vector.svg`; 이름과 호출은 `SKILL.md` 의 표
  (hair, skin, hands, shoes, legs, tail, C 의 옷, 장면의 subject). `<부위>_<무엇>.png` 은 그 부위에 합쳐진다 (옷 프롬프트 여러 개).
  명령이 크기를 맞추고, 겹친 곳은 `parts.NAMES` 순서로 먼저 온 부위에 준다 (shoes 가 legs 보다 먼저). 선화가 없으면 B·C 는 그
  프레임을 거부한다.
- **Adobe 올리기** (2026-10-12 에 직접 확인): `asset_initialize_file_upload` → transfer href 에 `curl -L -X PUT` → `asset_finalize_file_upload`
  가 `presignedAssetUrl` 을 준다. bodyParts 를 여러 개 한 번에 주면 한 마스크로 온다 (RGBA, 흰 RGB = 선택; 로더가 읽는다).
  같은 그림에 같은 호출은 같은 마스크를 돌려준다.
- **B (블렌드)**: 부위마다 L* 분위 톤 곡선 + a*/b* 이동 (게임 크기에서 잰다, 바깥 2 px 는 따로) → 머리의 작은 검은 얼룩 정리 →
  premultiplied 축소 → 부위마다 장면의 외곽선 색으로 ~1.3 px 바깥선 (더 어두운 화소는 그대로). 그림자 없음.
  명령 쪽: 두 그림에 다 있는 부위만 쓰고, 나머지 인물은 `rest` 부위 (옷 마스크가 필요 없다; JSON 의 `rest_of`). 장면의 인물 =
  subject ∪ 모든 부위 (화가 그림에서 Adobe subject 가 꼬리의 97 % 를 놓친 적이 있다).
- **C (룩)**: 실루엣 깊이 띠 × 빛 쪽 방향 칸마다 선형 RGB 로그 비 표 + 깊이별 톤 기울기, 한 쌍에서 ridge 최소제곱으로 맞춤.
  각 프레임 자신의 실루엣 기하로 적용 → 발이 바뀌면 빛 쪽 다리가 빛을 받는다. 흰색으로 안 날아가는 soft knee.
  명령 쪽: 참고가 원본의 절반도 안 겹치면 `CHECK FAILED` 로 룩을 쓰지 않는다. `rest` 는 C 에 넣지 않았다 (린 전달이 나빠졌다).
- **A (팩)**: 그대로 (같은 방향 프레임만, 관문 통과 못하면 워시·림). 린 숲 석양 `scenes/forest_sunset.json`, 팩은 공개 저장소 밖.
- 테스트 89개, 합성 그림만: `python -m pytest -q`.
- `bench/` 는 평가 전용이다 (데이터 경로는 gitignored `paths.local.json`). `bench/metrics.thin_lines` 는 평가 지표용 선 검출로
  파이프라인에는 쓰지 않는다. `scenelight/geom.py` 는 지금 `tests/test_geom.py` 만 쓴다 (지워도 된다: 테스트 85개가 된다).

## 해 봤다가 버린 것 — 다시 하지 않는다
- **화소별 ML 모델** (v7 GBM, 첫 단계의 94특징 회귀): 자기 맞춤은 좋았지만(v7 self dE 4.6) 다른 프레임에서 흰 화소 4천여 개가 새로 생겼고
  교차검증이 무너졌다(회귀 0.833 → 0.76). 무겁고 고치기 어렵다. C 의 작은 표 모델로 바꿨다.
- **우리 쪽 선·영역·캐릭터 검출기** (`lines.py`, `regions.py`, `character.py`, 1006줄 삭제): 사용자가 거부했다. Adobe 결과만 쓴다.
- **그림자 합성** (바닥·발밑·접촉 그림자, v10 의 보드 그림자): 연구 범위에서 뺐다. 결과 프레임에 그림자를 넣지 않는다.
  (`preview` 의 발밑 그림자는 화면 확인용일 뿐이다.)
- **화가 맞춤형 긴 체인** (v9/v10, 사례별 단계 약 1600줄: 머리 방향×깊이 보정, 꼬리 방사 프로필, 소매 패치, 눈 톤 ...):
  B 의 네 단계로 줄였다. 부위 분위는 거의 같다; 바깥 띠는 손으로 묶은 마스크로는 v10 만큼 화가에 가깝고, SKILL 마스크로는
  더 어둡다 (`README.md` 표).
- **오려 붙이기 도구**: 만들지 않는다. 특별한 경우는 작가/AI 그림을 주문해서 Adobe `image_remove_background` 로 오려 쓴다.
- 그 전 단계에서 버린 것 (가슴 타원체, 실루엣 Lambert 주광, 원화 명암 빼기, 한 가지 주황 곱 ...): `RESEARCH.md`.

## 남은 차이 (사실대로) — 숫자는 SKILL 의 Adobe 호출 그대로 받은 마스크로 잰 것 (`README.md`)
- B: 부위 L*·a*·b* 는 화가와 1.6 이내지만, 머리·피부·옷의 바깥 띠가 화가보다 어둡다 (L* 35.7 / 29.5 / 27.7 vs 50.3 / 53.2 / 42.8).
  연구 때 손으로 묶은 마스크로는 0.6 이내, 바깥 띠 49.0 / 47.7 / 32.6 였다. `rest` 곡선 하나가 어두운 조끼와 밝은 소매를 같이
  다룬다. 피부(얼굴·손목)는 가늘어 이웃 곡선과 섞인다 (명령의 표에서 참고용 줄). 걷기 프레임: 이름 붙은 부위 42 % / 37 %.
- C: 옷 프롬프트까지 받아야 L1 이 검사를 넘는다 (자기 dE 7.81, 검사선 8; bodyParts 만이면 8.54). 이전 연구판(v6 / v7)보다 뒤지는
  곳이 있다 — 자기 dE (L1 7.81 vs 6.49 / 4.56, L2 5.98 vs 5.29 / 3.55, L3 5.01 vs 4.11 / 3.03), 전달 띠 L2 walk b 7 vs 14 / 15,
  L1 walk b 5 vs 9 / 4, P02 21 vs 28 / 28. 빛 쪽 다리 차이가 작다 (walk a +1.5 vs 먼 다리 +0.4). 가는 다리 가운데 희미한 줄
  (2배 확대), 달빛에서 올리브 천이 청록, 밝은 재질은 참고만큼 밝아지지 않는다. 대신 새 흰 화소 0, 시트 크기에서 이음매 없음,
  밝아진 선화 화소 0.
- Adobe 는 그림마다 다르게 고른다: 린 P01 의 Upper Clothes / Coat 는 27 / 121 화소, 손·다리 묶음 호출은 한쪽만 돌려줬다,
  걷기 B 는 얼굴을 못 찾았다 (그 얼굴은 `rest` 곡선을 받아 어둡다). 이런 프레임은 비용 규칙으로 간다.
- 이 차이를 줄이려고 연구를 더 하지 않는다 → 비용 규칙.

## 절대 하지 말 것
- 알파 바꾸기, 선화 밝히기, 그림자 그려 넣기, 우리 쪽 부위·선 검출기 만들기.
- 빛 참고끼리 빼거나 합치기 (빛 종류마다 참고 하나, 룩 하나).
- 사용자 그림·마스크·팩·결과 그림이나 그 경로를 커밋하기 (공개 저장소; `.gitignore` 는 팩, `masks/`, `out/` 만 막는다).
- A 의 옛 금지 사항: 가슴 타원체(`geo.bh > 0`)·옛 빛 모델(`cfg.model = "normals"`) 켜기, 관문 값 근거 없이 바꾸기 (`RESEARCH.md`).
- lyn-sprites 원화 파츠(`assets/parts/`)에 적용하기 — 렌더된 최종 프레임에만 쓴다.

## 확인 방법
```bash
pip install -r requirements.txt
python -m pytest -q                         # 89 passed
python relight.py look-fit orig.png lamp.png -o look.json --sheet check.png    # masks/orig/ 에 Adobe 결과
```

## 파일
`SKILL.md` 실행 순서 · `README.md` 방법과 숫자 · `RESEARCH.md` 첫 단계(팩·워시·림)의 시도와 실패 · `relight.py` 명령 ·
`scenelight/` (`blend.py` B, `look.py` C, `parts.py` Adobe 로더, `transfer.py`·`chest.py`·`shading.py` A, `align*.py` 정렬,
`color.py` 색, `report.py` 지표·시트) · `bench/` 평가 · `tests/`.
원본 저장소: `wdanyang93/aseprite-builder`, 브랜치 `claude/upbeat-goldberg-s58fdw`, 폴더 `tools/sprite_harmonize/`.
