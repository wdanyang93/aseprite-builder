# CLAUDE.md — lyn-sprites (Lyn 2D 컷아웃 리깅)

이 저장소는 **원화를 파트로 자르고, 그 파트를 뼈대에 붙여 움직이게 한다.** 그게 전부다.
걸음은 뼈대가 만들고, 원화는 그 위에 얹힌다. 그래서 옷이 바뀌어도 걸음은 그대로다 = 통일감.

## 응답 규칙 (사용자 지시)
- 모든 작업 응답 첫 줄에 필요 노력을 적는다: `필요 노력: 낮음 | 중간 | 높음 | 엑스트라 | 최대`.
- "됐다"고 말로만 하지 말고 결과 이미지(스트립·GIF)와 수치를 보여준다.
- 실패한 시도는 지우지 말고 `docs/failures.md` 에 사실대로 적는다.

## 절대 규칙
1. **파츠는 통째로만 움직인다.** 회전·이동·(균일)확대만. 행 단위로 늘리거나 잘라 붙이지 않는다.
   (이전 연구에서 row-warp 는 화소 24% 중복·찢김을 만들었다 — `docs/archive/docs/research/failures.md`)
2. **색칠·재채색 금지.** 옷은 원화 화소를 그대로 쓴다.
3. **원화에서 자세를 추출하지 않는다.** 원화는 기준 포즈(A-pose) 파츠일 뿐이고, 자세는 `rigkit/walk.py` 가 만든다.
   "걷는 시트 36장을 분석해서 …" 로 시작하는 제안은 거절한다. 그게 이전 연구가 막힌 지점이다.
4. **파츠 이름을 자동으로 알아맞히지 않는다.** `cut` 이 번호를 붙이면 사람이 `names.json` 에 슬롯을 적는다.
   색·성분으로 파츠를 추론하려던 시도는 16번 실패했다.
5. **모르는 화소는 모른다고 남긴다.** 가려진 부분을 추정으로 채우지 않는다. 필요하면 그 각도의 파츠를 새로 요구한다.
6. **AI 가 그려 준 크기를 믿지 않는다.** 파츠 길이를 재서 리그 뼈 길이에 맞춘다(`fit`).
   "AI 에게 정확한 크기로 다시 그려 달라"는 해법은 쓰지 않는다 — 그게 비율이 무너지던 원인이다.
   AI 에게 요구할 것은 한 시트 안에서 파츠끼리 비율이 맞는 것뿐이고, 그건 `check` 로 확인한다.
7. **포즈 변형 시트를 요구하지 않는다.** 굽힌 팔·든 다리 같은 변형은 뼈대가 만든다.
   원화는 곧게 편 파츠 한 벌이면 된다.
8. **뷰 간 전이 금지.** 정면 파츠로 측면을 만들지 않는다. 각 뷰는 자기 파츠 세트를 가진다.
9. Godot 쪽 기존 동작(Background, Dig, Room, Light, Walk)은 바꾸지 않는다. 이 리그는 새로 추가되는 씬이다.

## 좌표 규약 (`rigkit/spec.py`)
- 캔버스 **512×640**, 몸 높이(정수리→발바닥) **H=560**, 중심선 x=256, 발바닥 y=620.
- 몸축 **t** = (y − 정수리)/H. 관절 위치는 전부 t 로 말한다 (`JOINT_T`: 어깨 .205, 팔꿈치 .345,
  손목 .48, 골반 .505, 무릎 .715, 발목 .925).
- 각도 **+ = 화면 시계방향**. `walk.py` 안에서는 '앞으로 = +' 로 쓰고 반환할 때 뒤집는다.
- 위상 **phi ∈ [0,1)** = 한 보행 주기(두 걸음). phi 0 = 오른다리 최대 앞 + 몸이 가장 낮음(착지).

## 파이프라인 (`rigkit/`, 실행 순서)
```
cut      파츠 시트 → 조각 PNG + CONTACT.png (번호)      python -m rigkit cut  SHEET -o build/pieces
split    통짜 팔다리를 관절에서 두 조각으로            python -m rigkit split LEG thigh_l shin_l -o assets/parts --at 0.5
assign   names.json 으로 조각 → 슬롯 + 앵커 계산         python -m rigkit assign build/pieces names.json -o assets/parts
fit      파츠 크기를 뼈 길이에 맞춘다 (scale 계산)       python -m rigkit fit   assets/parts
check    시트 안 파츠끼리 비율이 맞는지 본다             python -m rigkit check assets/parts
preview  파이썬으로 걷기 스트립/GIF 굽기 (검증용)        python -m rigkit preview assets/parts -o build/preview
godot    Godot 4 씬(.tscn) 내보내기                      python -m rigkit godot assets/parts/parts.json -o godot/lyn_rig.tscn
```
`rig.py` 뼈대·슬롯·z순서, `walk.py` 걷기 곡선, `render.py` FK+합성, `spec.py` 좌표 규약.
**하드코딩 경로를 새로 만들지 않는다.** 모든 입출력은 CLI 인자다.

## 개발 방식
- 테스트 먼저. `pytest -q` 가 CI. 불변식(파츠 화소 보존, 반주기 대칭, 무릎 역꺾임 금지)을 수치로 건다.
- 한 PR = 한 가지. 결과 스트립/GIF 를 PR 에 첨부한다.
- 큰 이진 파일은 Git LFS (`assets/raw/`, `assets/parts/`). `build/` 는 산출물이라 .gitignore.

## 현재 상태
- 뼈대 19개, 슬롯 20개, 걷기 8프레임, 비율 자동 정렬, 관절 분할, 파이썬 미리보기 + Godot .tscn. 테스트 37개 통과.
- **아직 실제 원화 파츠가 없다.** `build/placeholder_parts` 는 검증용 임시 도형이다.
  진짜 파츠가 들어오면 `assets/raw/` → `cut` → `assign` → 같은 명령으로 그대로 돌아간다.

## 남은 한계
- 측면 리그는 좌우 오프셋을 원근으로 줄인 근사다. 측면 전용 파츠(팔·다리 옆모습)가 있어야 제대로 된다.
- 망토·꼬리는 뼈 3단 근사다. 천 시뮬레이션이 아니다.
- 8방향 중 동/남만 규약이 정해져 있다. 북·대각은 그 뷰의 파츠 세트가 있어야 추가할 수 있다.
