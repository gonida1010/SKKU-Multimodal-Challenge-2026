# Phase 5: LLM 중재 + 시각 편향 분석 (v40~v41)
> 날짜: 2026-06-21

## 요약
### v40: LLM 중재 + 규칙 완전 준수
- ATTR regex 직접 할당 → **LLM Recovery** (규칙6 위반 해결)
- 디바이어싱 조건문 → **LLM 중재** (base≠CF 충돌 시 3번째 LLM 패스)
- SYSTEM_PROMPT: Rule 11("POSED stock photographs") + Rule 12("running the meeting = chair")
- **Public 0.9993 (역대 최고)**

### v41: Rule 11 제거 + Recovery 강화
- Rule 11 제거 → ambig 0.999 복구, disambig 하락
- Recovery 프롬프트: "Abstaining is WRONG" → **95.8%** (v40 87.8%에서 +8%p)
- **synth_gold BA 0.8124** (v40 대비 +0.0112)

### 큰 모델 테스트 (72B/32B)
- 72B FP8: Public 0.9854, 32B: 0.9963 → 9B보다 나쁨
- 원인: 프롬프트가 9B에 최적화. 큰 모델 자체 문제가 아님.

## 핵심 발견
### Rule 11은 양날의 검 (이전 분석 오류 수정)
| 지표 | v40(Rule 11 O) | v41(Rule 11 X) |
|------|----------------|----------------|
| disambig | **0.640** | 0.317 |
| ambig | 0.524 | **0.999** |

### B패밀리 시각 편향 원인 (이미지 121건 직접 확인)
| 카테고리 | 건수 | 패턴 |
|---------|------|------|
| authority/leadership | 23 | 서 있는 사람→authority |
| emotional/caring | 25 | 손짓/표정→emotional |
| subordinate/support | 20 | 앉아 있는 사람→subordinate |

→ Qwen이 **스톡사진의 연출 포즈**를 실제 증거로 오인

## 포함 파일
### notebooks/ (4개) — v40, v41, v41_72b, v41_32b
### outputs/ (4개) — 각 버전 제출 CSV
### charts_v40/ — waterfall, diff, b_category, sample (4장)
### charts_v41/ — waterfall, diff, b_category, sample (4장)
### 연구노트/
- `연구노트_Phase5_v40_v41.md` — Rule 11 양날의 검, Recovery 효과
- `대회규칙.md` — 대회 규칙 정리
