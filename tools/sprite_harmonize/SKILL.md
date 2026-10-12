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
사용자에게는 무엇이 왜 안 됐는지 시트와 숫자로 알린다. (마스크를 한 번 다시 받는 것까지는 해도 된다.)

## 준비
```bash
cd <이 스킬 폴더> && pip install -r requirements.txt     # numpy, opencv, pillow, scipy, cairosvg (SVG 선화)
```

## Adobe 단계 (B·C 에 필요 — 명령을 돌리기 전에 Claude 가 직접 한다)
그림 `X.png` 의 결과는 `masks/X/` 에 둔다: 부위 마스크 `<이름>.png` (흰색 = 선택) + 선화 `vector.svg`.
받을 그림: B 는 fit 프레임·장면 그림·적용할 프레임마다, C 는 원본과 적용할 프레임마다. C 의 참고 그림은 마스크가 필요 없다
(투명 배경이 아니면 `image_remove_background` 만).
1. `adobe_mandatory_init` (대화에서 한 번).
2. 올리기: 투명 배경 프레임은 같은 캔버스의 회색(#808080) 배경에 얹은 PNG 로 올린다 (검증 때 그렇게 했다).
   `asset_initialize_file_upload` (path, file_size, media_type `image/png`) → 받은 transfer href 마다
   `curl -L -X PUT -H "Content-Type: image/png" --data-binary @X.png "<href>"` (기본 block_size 10 MB 까지 PUT 한 번; 더 크면
   block_size 를 올리거나 블록마다 PUT) → `asset_finalize_file_upload` (filename, 받은 transfer_document 그대로) →
   응답의 `presignedAssetUrl` 이 imageURI. https URL 이 없으면 `asset_get_presigned_urls` ({id: assetId, sourceType: "acp"}) 의 `previewUrl`.
3. 부위: 아래 호출을 **모든 그림에 똑같이** 하고, 결과 `outputUrl` 을 `curl -L -o masks/X/<파일> "<url>"` 로 받는다.

   | 호출 | 파일 |
   |---|---|
   | `image_select_subject` bodyParts `["Hair"]` | `hair.png` |
   | bodyParts `["Face", "Neck", "Left Arm", "Right Arm"]` (한 번에) | `skin.png` |
   | bodyParts `["Left Hand", "Right Hand"]` | `hands.png` |
   | bodyParts `["Left Shoe", "Right Shoe"]` | `shoes.png` |
   | bodyParts `["Left Leg", "Right Leg"]` | `legs.png` |
   | `image_select_by_prompt` prompt "the fluffy animal tail" | `tail.png` |
   | `image_select_subject` bodyParts 없이 (B 의 장면 그림) | `subject.png` |
   | C 만: bodyParts `["Upper Clothes"]` / `["Lower Clothes"]` / `["Coat"]` | `upper.png` / `lower.png` / `coat.png` |

   - B 는 옷을 받지 않는다: 이름 붙은 부위가 덮지 않은 나머지가 자동으로 `rest` 부위가 된다 (프레임과 장면 둘 다).
   - C 의 옷: 결과가 거의 비었거나 (인물의 1 % 미만) 인물 전체면 버리고, 보이는 옷마다 `image_select_by_prompt` 한 번
     ("the vest", "the shirt", "the sleeves", "the belt", "the skirt", "the gloves" ...) 해서 `<묶음>_<옷>.png` 로 저장한다:
     겉옷·조끼·벨트 → `coat_`, 셔츠·튜닉·소매 → `upper_`, 치마·반바지·바지 → `lower_`, 장갑 → `hands_`. 같은 묶음은 합쳐진다.
     (린 P01: Upper Clothes 27 화소, Coat 121 화소였다 → 옷 프롬프트 없이 C 는 검사를 못 넘었다.)
4. 선화: `image_vectorize` → SVG 를 `vector.svg` 로. 프레임과 원본만 (장면 그림·참고 그림은 필요 없다).
   cairosvg 를 못 쓰면 그림과 같은 크기로 렌더한 `vector.png`.
- "bare skin" 같은 피부 프롬프트는 거절된다 → 피부는 위 bodyParts 로. Adobe 의 Left/Right 는 해부학 기준이다. 묶은 호출이 한쪽만
  돌려주거나 (린: 손 하나, 다리 하나) 아무것도 못 찾으면 그대로 둔다 — 그 부위가 없는 것으로 친다.
- 겹친 곳은 `parts.py` 이름 순서로 먼저 온 부위가 가진다 (shoes 가 legs 보다, upper 가 coat 보다 먼저) → 손으로 빼지 않는다.
  마스크 크기가 그림과 달라도 명령이 맞춘다. 장면의 인물 = `subject.png` + 모든 부위 (subject 가 꼬리를 놓쳐도 된다).

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
**C. 참고 룩** — 원본 + 같은 포즈의 빛 받은 참고(투명 배경) → 다른 프레임·동작.
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

**B** blend-fit: 표에 `<- off by > 3` 이 없다 (인물의 5 % 이상인 부위만 검사; `small part`·`whole` 줄은 참고용),
`named Adobe parts cover` ≥ 50 % (`under 50 %` 없음), `alpha = plain downscale True`, `line px lighter 0`.
`outer edge L*` 는 참고용 — 옷 외곽선은 장면보다 어두울 수 있다 (선화를 밝히지 않는 규칙).
시트 (줄인 원본 | 블렌드 | 장면, 이어서 머리 확대): 색·톤이 장면 속 캐릭터 같고, 외곽선이 장면 같고, 테두리 번짐과 머리의 검은
얼룩이 없어야 한다. blend: `<- no mask for: X` 가 뜬 프레임은 X 가 그림에 보이면 그 마스크를 다시 받는다 (맨발이면 shoes 는 정상).

**C** look-fit: `light from N deg` (0 = 화면 오른쪽, 90 = 위, 180 = 왼쪽) 가 참고에서 빛이 오는 쪽과 맞고,
`self colour error dE` ≤ 8 이면서 `unchanged` 의 절반 이하 (위 마스크로: 린 달 + 랜턴 7.8, 린 달빛 6.0, 브루나 램프 5.0),
`alpha identical True, line px lighter 0, new white px 0` (`CHECK FAILED` 없음). 참고가 원본과 절반도 안 겹치면 명령이
`CHECK FAILED` 로 거부한다 (같은 포즈가 아니다). 시트 (원본 | 룩 | 맞춘 참고, 이어서 머리 확대): 빛 쪽이 참고처럼 밝고 따뜻하고,
줄무늬·이음매·하얗게 날아간 곳이 없어야 한다. look: 프레임마다 같은 세 숫자; before|after 시트에서 빛 쪽 다리가 더 밝다.

알려진 한계 (통과로 친다): C — 가는 다리 가운데 희미한 줄(2배 확대에서만), 달빛에서 올리브 천이 청록, 밝은 재질은 참고만큼
밝아지지 않고 주황으로 남는다. B — `rest` 곡선 하나가 어두운 조끼와 밝은 소매를 같이 다룬다; 외곽선이 장면보다 조금 어둡다.

## 절대 규칙
- 알파를 바꾸지 않는다 (B 는 원래 알파를 게임 높이로 줄인 것 그대로). Adobe 결과 그림의 알파를 쓰지 않는다.
- 선화는 절대 밝아지지 않는다. 선화(`vector.svg`) 없이 B·C 를 돌리지 않는다 (명령이 거부한다).
- 그림자를 만들지 않는다 (바닥·발밑 그림자 포함). 하이라이트나 모양을 새로 그려 넣지 않는다.
- 부위와 선은 Adobe 결과만 쓴다. 우리 쪽 검출기를 만들지 않는다.
- 사용자 그림·마스크·팩·결과 PNG 와 그 경로를 절대 커밋하지 않는다 (공개 저장소).
- 렌더된 최종 프레임에만 쓴다 (lyn-sprites 원화 파츠 `assets/parts/` 에는 쓰지 않는다). 픽셀아트용이 아니다.

## 테스트
```bash
python -m pytest -q        # 합성 그림만, 89개
```
