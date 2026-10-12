---
name: scene-relight
description: 2D 게임 캐릭터 스프라이트·애니메이션 프레임을 장면 빛에 맞춘다 (색·톤·외곽선, 알파와 선화는 그대로, 그림자 없음). 세 방법 — (A) 참고 팩: 중립 프레임 + 같은 포즈를 장면 빛으로 칠한 참고 → 같은 방향 프레임 (fit/apply), (B) 장면 블렌드: 화가가 장면 안에 그려 넣은 캐릭터 그림 한 장 → 게임 크기 프레임 (blend-fit/blend), (C) 참고 룩: 원본 + 램프·달 등 빛 받은 참고 한 쌍 → 다른 프레임·동작에 빛 방향대로 (look-fit/look). 부위 마스크와 선화는 Adobe 에서 받는다. "스프라이트가 배경과 안 어울린다", "장면 조명 맞추기", "램프 빛 받은 것처럼", "relight sprites", "match sprite lighting to the scene" 같은 요청에 사용.
---

# scene-relight — 스프라이트를 장면 빛에 맞추기

> 처음 받은 AI 는 `HANDOFF.md`(지금 상태, 버린 것)도 읽는다. 숫자의 근거: `README.md`.

## 하는 일
캐릭터 프레임의 색·명암·외곽선만 바꿔 장면에 맞춘다. 한 번 맞추고(fit → 작은 JSON) 같은 장면·같은 빛의 모든 프레임에
적용한다(프레임당 몇 초). 알파는 그대로, 선화는 절대 밝아지지 않고, 그림자나 새 모양은 그리지 않는다.

## 비용 규칙 — 먼저 읽는다
**fit 을 한 번 한 뒤 프레임이 아래 검사를 통과하지 못하거나, 경우가 특별한 처리를 요구하면 (다른 옷, 다른 방향, 잘린 그림,
안 맞는 마스크 ...) 거기서 멈춘다.** 매개변수를 만지거나 새 방법을 연구하지 않는다. 그 프레임의 빛 받은 그림 / 장면 그림을
작가나 AI 에게 주문하고 Adobe `image_remove_background` 로 오려서 쓴다 — 연구를 더 하는 것보다 싸다.
사용자에게는 무엇이 왜 안 됐는지 시트와 숫자로 알린다.

## 준비
```bash
cd <이 스킬 폴더> && pip install -r requirements.txt     # numpy, opencv, pillow, scipy, cairosvg (SVG 선화)
```

## Adobe 단계 (B·C 에 필요 — 명령을 돌리기 전에 Claude 가 직접 한다)
그림 `X.png` 의 Adobe 결과는 `masks/X/` 에 둔다: 부위 마스크 `<이름>.png` (흰색 = 선택) + 선화 `vector.svg`.
필요한 그림: B 는 fit 프레임·장면 그림·적용할 프레임마다, C 는 원본과 적용할 프레임마다 (참고 그림은 필요 없다).
1. `adobe_mandatory_init` (대화에서 한 번).
2. 올리기: `asset_initialize_file_upload` (path, file_size, media_type `image/png`) → 응답의 업로드 URL 로
   `curl -X PUT --data-binary @X.png "<url>"` → `asset_finalize_file_upload` (filename, transfer_document) → 받은 URL 이 imageURI.
3. 부위: `image_select_subject` 를 부위마다 한 번 (`options.bodyParts`), 결과 URL 을 `curl -o masks/X/<파일> "<url>"` 로 받는다.

   | bodyParts | 파일 | bodyParts | 파일 |
   |---|---|---|---|
   | `Hair` | `hair.png` | `Left Leg` / `Right Leg` | `leftleg.png` / `rightleg.png` |
   | `Face` | `face.png` | `Left Shoe` + `Right Shoe` (한 번에) | `shoes.png` |
   | `Neck` | `neck.png` | `Left Hand` / `Right Hand` | `lefthand.png` / `righthand.png` |
   | `Upper Clothes` | `upper.png` | `image_select_by_prompt` "the fluffy animal tail" | `tail.png` |
   | `Lower Clothes` | `lower.png` | bodyParts 없이 (B 의 장면 그림에 꼭) | `subject.png` |
   | `Coat` | `coat.png` | | |
4. 선화: `image_vectorize` → SVG 를 `vector.svg` 로 저장 (cairosvg 를 못 쓰면 그림과 같은 크기로 렌더한 `vector.png`).
- "bare skin" 같은 피부 프롬프트는 거절된다 → 피부는 Face / Neck / Hand / Leg 부위로 받는다.
- Adobe 의 Left/Right 는 해부학 기준이다 (캐릭터의 왼쪽 = 정면 그림에서 화면 오른쪽). 이름을 바꾸지 않는다.
- 겹치는 마스크: 다리 − 신발. 명령은 위 이름 순서로 먼저 온 부위에 겹친 곳을 준다 (shoes 가 leftleg 보다 먼저) → 손으로 빼지 않아도 된다.
- 마스크 크기가 그림과 달라도 명령이 맞춘다. 거의 빈 마스크(수십 화소)는 그냥 둔다 — 전체 곡선으로 간다.

## 세 방법 (가진 그림으로 고른다)
**A. 참고 팩** — 중립 프레임 + 같은 포즈를 장면 빛으로 칠한 참고, 같은 방향의 프레임들 (Adobe 불필요).
```bash
python relight.py fit neutral.png reference.png -o scenes/S.json --pack scenes/S.pack --sheet check.png
python relight.py apply scenes/S.json frames/*.png -o out/ --pack scenes/S.pack --view-from-name --sheet ba.png
```
**B. 장면 블렌드** — 화가가 장면 안에 그려 넣은 캐릭터 그림(게임 크기) + 그 캐릭터의 프레임들. 결과는 게임 높이 PNG.
```bash
python relight.py blend-fit frames/idle.png scene.png -o blend.json --sheet check.png   # idle = 화가가 그린 그 프레임
python relight.py blend blend.json frames/*.png -o out/ --sheet ba.png
```
**C. 참고 룩** — 원본 + 같은 포즈의 빛 받은 참고(투명 배경; 아니면 먼저 `image_remove_background`) → 다른 프레임·동작.
```bash
python relight.py look-fit orig.png lamp_ref.png -o look_lamp.json --sheet check.png
python relight.py look look_lamp.json frames/*.png -o out/ --sheet ba.png
```
- 빛 종류마다 참고 하나, 룩 하나 (램프만 / 달만 / 달 + 랜턴 = 룩 셋). 참고끼리 빼거나 합치거나 평균 내지 않는다.
- 마스크 폴더가 `masks/` 가 아니면 `--masks 폴더`. 마스크·선화가 없는 프레임은 `FAILED` 로 건너뛰고 나머지는 저장된다.

## 검사 — 시트를 열어 보고, 숫자와 함께 사용자에게 보여 준다
**A** fit: `alignment IoU` ≥ 0.85, `soft transfer (body): drawn form kept` ≥ 0.95, `mean L*` 가 참고 ±2.
apply: 방향 묶음마다 `pass the pack gates` ≥ 80 % 면 참고 빛. `wash+rim fallback` 프레임은 참고 빛을 못 받은 것이다.
시트: 중립 | 결과 | 참고 — 가슴·윤곽 모양이 원화 그대로여야 한다.

**B** blend-fit 표: 부위마다 `L* median`·`a*`·`b*` 가 장면과 3 이내 (`<- off by > 3` 없음; 린 화가 장면에서는 1 이내),
`alpha = plain downscale True`, `line px lighter 0`, `Adobe parts cover` ≥ 70 %.
`outer edge L*` 는 참고용 (린: 머리 50 / 장면 53, 피부 49 / 56, 부츠 17 / 14, 꼬리 30 / 32, 옷 33 / 43 — 옷은 원래 외곽선이
더 어둡고 밝히지 않는 규칙이라 남는 차이). 시트 (줄인 원본 | 블렌드 | 장면, 이어서 머리 확대): 색·톤이 장면 속 캐릭터 같고, 외곽선 두께·
어둡기가 장면 같고, 테두리 번짐과 머리의 검은 얼룩이 없어야 한다.
blend: 프레임마다 `Adobe parts cover` < 50 % 면 `<- under 50 %: check the masks` → 그 프레임 마스크를 다시 받는다.

**C** look-fit: `light from N deg` (0 = 화면 오른쪽, 90 = 위, 180 = 왼쪽) 가 참고에서 빛이 오는 쪽과 맞고,
`self colour error dE` ≤ 8 이면서 `unchanged` 의 절반 이하 (검증: 린 달 + 랜턴 7.7~7.9, 린 달빛 5.8, 브루나 램프 5.0;
그대로 두면 19.1 / 33.6 / 15.2),
`alpha identical True, line px lighter 0, new white px 0` (`CHECK FAILED` 없음).
시트 (원본 | 룩 | 맞춘 참고, 이어서 머리 확대): 빛 쪽이 참고처럼 밝고 따뜻하고, 줄무늬·이음매·하얗게 날아간 곳이 없어야 한다.
look: 프레임마다 같은 세 숫자. before|after 시트에서 빛 쪽 다리가 더 밝다 (발이 바뀌면 밝은 다리도 바뀐다).

알려진 한계 (통과로 친다): C — 가는 다리 가운데 희미한 줄(2배 확대에서만), 달빛에서 올리브 천이 청록, 밝은 재질은 참고만큼
밝아지지 않고 주황으로 남는다. B — 옷 곡선 하나가 어두운 조끼와 밝은 소매를 같이 다뤄 소매가 조금 회색이다.

## 절대 규칙
- 알파를 바꾸지 않는다 (B 는 원래 알파를 게임 높이로 줄인 것 그대로). Adobe 결과 그림의 알파를 쓰지 않는다.
- 선화는 절대 밝아지지 않는다. 선화(`vector.svg`) 없이 B·C 를 돌리지 않는다 (명령이 거부한다).
- 그림자를 만들지 않는다 (바닥·발밑 그림자 포함). 하이라이트나 모양을 새로 그려 넣지 않는다.
- 부위와 선은 Adobe 결과만 쓴다. 우리 쪽 검출기를 만들지 않는다.
- 사용자 그림·마스크·팩·결과 PNG 와 그 경로를 절대 커밋하지 않는다 (공개 저장소).
- 렌더된 최종 프레임에만 쓴다 (lyn-sprites 원화 파츠 `assets/parts/` 에는 쓰지 않는다). 픽셀아트용이 아니다.

## 테스트
```bash
python -m pytest -q        # 합성 그림만, 86개
```
