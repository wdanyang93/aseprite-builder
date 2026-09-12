# lyn-sprites

Lyn 캐릭터의 **2D 컷아웃 리깅**. 원화를 파트로 자르고, 뼈대에 붙이고, 걷게 만든다.
옷 교체 = 같은 슬롯 이름의 파츠 묶음 교체. 걸음은 한 번만 만들면 모든 옷이 같은 걸음을 걷는다.

```bash
pip install -e ".[dev]"

python -m rigkit cut      assets/raw/outfit.png -o build/pieces      # 시트 → 조각 + CONTACT.png
python -m rigkit assign   build/pieces names.json -o assets/parts    # 조각 → 슬롯 + 앵커
python -m rigkit fit      assets/parts                               # 비율을 뼈 길이에 맞춘다
python -m rigkit check    assets/parts                               # 파츠끼리 비율이 맞는지 확인
python -m rigkit preview  assets/parts -o build/preview --side east  # 걷기 스트립/GIF
python -m rigkit godot    assets/parts/parts.json -o godot/lyn.tscn  # Godot 4 씬
```

실제 원화가 아직 없다면 임시 도형으로 리그만 먼저 확인할 수 있다:

```bash
python -m rigkit placeholder -o build/pp && python -m rigkit preview build/pp -o build/preview
```

- `rigkit/spec.py` 좌표 규약 · `rig.py` 뼈대 19개·슬롯 20개 · `walk.py` 걷기 곡선
- `rigkit/render.py` 파이썬 미리보기 · `godot.py` Skeleton2D+Bone2D+Sprite2D 씬 내보내기
- `rigkit/fit.py` **비율 자동 정렬** — AI 가 몇 배로 그려 오든 뼈 길이에 맞춘다
- `rigkit/split.py` 통짜 팔다리를 관절에서 분할 (겹침 여유 포함)
- `docs/spec/PARTS-SPEC.md` **AI 에게 파츠를 일관되게 받아내는 규격서**
- `docs/archive/` 이전 연구(걷기 시트 분석) 기록 — 왜 그만뒀는지는 `docs/failures.md` F-17
