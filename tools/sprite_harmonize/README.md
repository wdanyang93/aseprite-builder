# sprite_harmonize

배경과 따로 노는 2D 스프라이트를 배경의 조명에 맞게 후처리하는 Python 스크립트.

```
pip install -r requirements.txt
```

## 하는 일

| 단계 | 내용 |
|---|---|
| defringe | 반투명 가장자리의 흰 테두리(흰 배경에서 뽑은 잔재)를 안쪽 색으로 교체 |
| grade (auto) | 배경 상위 10% 밝은 영역에서 **키 라이트 색**을 추출해 스프라이트에 곱하고, 스프라이트 평균 밝기를 배경의 밝은 영역(상위 4%) 수준으로 맞춤 |
| grade (`--ref`) | 원하는 느낌으로 직접 만든 스프라이트가 있으면, 그 색 통계(LAB: 밝기 평균/분산 + 명부·중간·암부별 색조)를 학습해서 적용 |
| rim light | `--light-angle` 방향의 실루엣 가장자리에 키 라이트 색 역광(얇은 선 + 부드러운 번짐) |
| composite | `--place` 로 배경에 합성 미리보기: 축소, 발밑 접지 그림자, 라이트 랩(배경색이 가장자리에 번짐), 장면 내 밝기(`--dim`) |

모든 색 연산은 선형(linear) 공간에서 함.

## 사용 예

```bash
# 1) 배경만 보고 자동 보정 + 배경 위 합성 미리보기
python harmonize.py sprite.png --bg background.png -o sprite_lit.png \
    --place 640,602,240          # 발 중심 x, 발 y, 스프라이트 높이(px)

# 2) 직접 만든 "정답" 스프라이트에서 룩을 학습해 저장
python harmonize.py sprite.png --bg background.png --ref target_sprite.png \
    --save-look forest_sunset.json -o sprite_lit.png

# 3) 저장한 룩을 애니메이션 프레임 전체에 일괄 적용 (프레임 간 깜빡임 없음)
python harmonize.py frames/*.png --look forest_sunset.json -o graded_frames/
```

auto 모드도 `--save-look` 으로 저장해 두면, 첫 프레임에서 계산한 노출값이 고정돼 나머지 프레임에 똑같이 적용됨.

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--tint` | 0.7 | 키 라이트 색을 얼마나 입힐지 (0 = 원색 유지) |
| `--exposure` | 자동 | 밝기 배율(선형). 자동값이 너무 어둡거나 밝을 때 직접 지정 |
| `--haze` | 0.08 | 암부를 배경 그림자 색 쪽으로 들어올림 |
| `--light-angle` | 160 | 역광이 오는 방향 (0=오른쪽, 90=위, 180=왼쪽) |
| `--rim` / `--rim-width` | 1.2 / 높이÷60 | 림라이트 세기 / 두께(px) |
| `--shadow` | 0.55 | 접지 그림자 진하기 |
| `--wrap` | 0.35 | 라이트 랩 양 |
| `--dim` | 0.92 | 합성 시 장면 안에서의 밝기 |

## 참고

- 일러스트 스타일 스프라이트 기준으로 만듦. 도트(픽셀아트)에 쓰면 새 중간색이 생기므로, 결과를 Aseprite에서 팔레트로 다시 인덱싱하는 것을 권장.
- 합성(`--place`)은 확인용 미리보기. 게임에는 보정된 RGBA 스프라이트(`-o`)를 쓰고 그림자/라이트 랩은 엔진 쪽 셰이더로 처리하는 편이 좋음.
