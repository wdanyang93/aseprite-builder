# 파츠 시트 규격서 — AI 에게 매번 같은 그림을 받아내는 방법

AI 가 매번 다른 비율·다른 걸음으로 그려서 통일감이 없던 것이 이 프로젝트의 원래 문제였다.
해법은 **AI 에게 애니메이션을 시키지 않는 것**이다. AI 에게는 "기준 포즈의 부품"만 받고,
움직임은 리그가 만든다. 그러면 AI 가 얼마나 변덕스럽든 상관없다.

## 1. AI 에게 요구할 것 (요구하지 말 것)

| 요구한다 | 요구하지 않는다 |
|---|---|
| 기준 포즈(정면 A-pose) 전신 1장 | 걷는 시트, 여러 프레임 |
| 같은 캐릭터의 4방향 턴어라운드 (정면/측면/후면/반측면) | "자연스러운 포즈" |
| 옷 파츠만 따로 늘어놓은 시트 (흰 배경) | 배경, 그림자, 이펙트 |
| 좌우 파츠를 따로 (장갑 L/R, 부츠 L/R, 레깅스 L/R) | 한 덩어리로 붙은 전신 옷 |

## 2. 파츠 시트 조건

- 배경은 **순백(#FFFFFF) 단색**, 그림자 없음. 파츠끼리 **겹치지 않게** 띄워 놓는다.
- 흰 옷(후드·셔츠)은 **반드시 윤곽선**이 있어야 한다. 윤곽선이 없으면 배경과 붙어 잘려나간다.
  (`cut` 은 '테두리와 이어진 흰색'만 배경으로 본다 — `tests/test_rig.py::test_white_garment_survives_background_removal`)
- 한 시트 안에서 **모든 파츠가 같은 축척**. 부츠만 크게 그리지 않는다.
- 해상도는 전신 기준 세로 1000px 이상. 캔버스 512×640 으로 줄여 쓴다.

## 3. 슬롯 목록 (`rigkit/rig.py` 의 SLOTS 와 1:1)

| 슬롯 | 파츠 | 앵커(관절점) |
|---|---|---|
| `head` | 머리+머리카락+귀 | 목 (아래쪽 78%) |
| `torso` | 상의(후드/셔츠)+가슴 벨트 | 가슴 (위쪽 6%) |
| `hips_wear` | 하의(반바지/스커트)+허리 벨트 | 골반 (위쪽 10%) |
| `upperarm_l/r` | 소매 윗부분 | 어깨 (윗변 중앙) |
| `forearm_l/r` | 소매 아랫부분/팔뚝 | 팔꿈치 (윗변 중앙) |
| `hand_l/r` | 장갑/손 | 손목 (윗변 중앙) |
| `thigh_l/r` | 레깅스 허벅지 | 골반 (윗변 중앙) |
| `shin_l/r` | 레깅스 종아리 | 무릎 (윗변 중앙) |
| `foot_l/r` | 부츠 | 발목 (위 12%, 가로 45%) |
| `tail1/2/3` | 꼬리 3단 | 각 마디 뿌리 |
| `backpack`, `bag` | 배낭, 가방 | 가슴 |

**팔다리는 관절에서 잘린 조각으로 받아야 한다.** 통짜 다리 1장을 받으면 무릎에서 접을 수 없다.
AI 가 통짜로 주면 `cut` 후 이미지 편집기에서 관절선으로 한 번 더 나눈다 (겹침 여유 10~15px 남길 것).

## 4. 프롬프트 틀

> character reference sheet, {캐릭터 설명}, A-pose standing straight, front view,
> plain pure white background, no shadow, full body, consistent proportions,
> clean line art
>
> (파츠 시트) equipment parts laid out separately on pure white background:
> hood cloak / shirt with belts / shorts / leggings left and right separately /
> gloves left and right / boots left and right, each part fully separated and not
> overlapping, same scale, clean outlines, no background

## 5. 넣는 순서

```bash
# 1) 시트를 assets/raw/ 에 넣는다 (Git LFS)
python -m rigkit cut assets/raw/outfit_front.png -o build/pieces_front
# 2) build/pieces_front/CONTACT.png 를 보고 names.json 을 적는다
#    {"01": {"slot": "torso"}, "04": {"slot": "thigh_l"}, ...}
python -m rigkit assign build/pieces_front names.json -o assets/parts/front
# 3) 확인
python -m rigkit preview assets/parts/front -o build/preview_front --side front
# 4) Godot
python -m rigkit godot assets/parts/front/parts.json -o godot/lyn_front.tscn --side front
```
