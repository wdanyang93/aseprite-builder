# lyn-sprites — Claude Code 로 구현하는 순서

이 폴더는 연구 세션(Cowork)에서 검증된 스크립트·골든 파일·문서를 **저장소 씨앗**으로 묶은 것이다.
Claude Code 는 `CLAUDE.md` 를 매 세션 자동으로 읽으므로, 거기 적힌 절대 규칙·좌표 규약·실패 목록이 곧 "제대로 구현" 의 기준이 된다.

## 0. 저장소 만들기 (한 번)
```bash
# 이 폴더를 새 저장소 루트로
git init && git lfs install
git add -A && git commit -m "seed: walkbank pipeline, goldens, docs"
gh repo create lyn-sprites --private --source . --push     # 또는 GitHub 에서 만든 뒤 remote 추가
# 원화 시트 11장을 assets/raw/sheets/{set}.png 로 넣고 (MANIFEST.md 참고) LFS 로 커밋
git add assets/raw/sheets && git commit -m "assets: raw walk sheets (LFS)" && git push
```

## 1. Claude Code 세션 시작
```bash
cd lyn-sprites
claude                         # CLAUDE.md 자동 로드
/install-github-app            # (선택) PR 에서 @claude 로 리뷰·수정 요청 가능하게
```
첫 프롬프트 예:
> M1 을 계획 모드로 설계해줘. `platekit/walkbank/bank.py` 의 하드코딩 경로(UP, W)를 CLI 인자로 바꾸고,
> `python -m platekit.walkbank build --sheets assets/raw/sheets --out build/bank` 로 실행되게.
> 결과 `build/bank/labels.json` 이 `golden/labels.json` 과 같아야 하고(437장, 세트별 방향), 그걸 검증하는
> 테스트를 `tests/test_walkbank.py::test_bank_regression_against_golden` 에 활성화해줘. 동작은 바꾸지 마.

## 2. 마일스톤별 PR (한 PR = 한 마일스톤, 작게)
| M | 할 일 | 완료 기준 |
|---|---|---|
| M1 | bank CLI 화 + 테스트 | 골든 labels 일치, CI 녹색 |
| M2 | posebank/frontwalk 의 자세·위상을 `platekit/walkbank/pose_side.py`, `pose_front.py` 로 정리 | 골든 pose.json / front_pairs8.json 의 N/F·phi 일치 |
| M3 | pairs8 → `pairs.py`, 시트 출력 | 골든 pairs8.json 8쌍 일치 |
| M4 | assemble/frontwalk 조립 → `build/walk/{east,south}/{0..7}.png` + GIF | 스트립·GIF 를 PR 에 첨부 |
| M5 | Godot 4.3+ 내보내기: SpriteFrames `.tres` + 아틀라스, 기존 walk compiler(5방향→8방향) 연결 | Godot 프로젝트에서 재생 |
| M6 | 북쪽·대각 시트 추가 시 뷰별 자세 신호만 교체 | 같은 파이프라인으로 8방향 |

각 PR 에서 Claude Code 에게 요구할 것: (1) 계획 모드 → 승인 후 구현, (2) 테스트 먼저, (3) 결과 이미지 첨부,
(4) 실패하면 `docs/research/failures.md` 에 추가, (5) `golden/` 과 다르면 이유를 수치로.

## 3. 프롬프트 요령 (이 프로젝트에서 효과 있었던 것)
- "말로만 됐다고 하지 말고 8프레임 스트립을 보여줘" — 항상 이미지·수치로 검증.
- 큰 근거부터: 얼굴 방향 → 진행 방향 → 발. 발만 보고 판단하는 코드는 거부.
- 세트끼리는 **자세로** 맞춘다 (프레임 번호 비교 금지). 두 AI 세트는 걸음·비율이 다르다는 전제.
- 불가능한 것(팔 분리, 허리 이음 완전 제거)은 CLAUDE.md 에 있다 — 다시 시도하자고 하면 거절하고 대안을 말하게.

## 4. 폴더
```
platekit/walkbank/   최종 파이프라인 (bank, posebank, pairs8, assemble, frontwalk, pose, debg, garmentstudy.tail_mask)
platekit/character/  이전 단계 패키지 (zones2.head_side 등)
golden/              이 세션의 측정 결과 = 회귀 기준
assets/canon/        고정 상부(측면 cE2_27, 정면 cS1_34), 고정 꼬리(cS1_18), 정면 나체 캐논 f05
assets/raw/sheets/   원화 시트 (LFS, MANIFEST.md)
docs/research/       failures.md(실패 16건), walk-bank-and-assembly.md, 연구 스크립트
docs/handoff/        다른 AI 와의 인계서 + 상태 JSON
docs/results/        현재 8프레임 결과 스트립
tests/               합성 데이터 단위 테스트 + 골든 회귀
```
