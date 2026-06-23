# Phase 2a: 프롬프트 공학 (v23~v25)
> 기간: 2026-06-12 ~ 06-13

## 요약
- v23: Dual-Route Recovery + PermSC(Permutation Self-Consistency) 본격 도입
  - 선택지 순서 3회 셔플 → 일치=확정, 불일치=LLM arbiter 종합 판단
- v24: G-Suite Prompt — 선택지 그룹 식별 정규식(`OPT_GRP`) 도입
- v25: Majority Tiebreak — 다수결 기반 최종 결정

## 핵심 발견
- PermSC가 순서 편향(position bias) 제거에 효과적
- G-Suite 정규식이 선택지의 인구통계 그룹을 자동 식별 → 반사실 디바이어싱의 기반
- v15 기준 6.07% 변경 (v25)

## 성능 추이
| 버전 | 핵심 기법 | 변경률 |
|------|----------|--------|
| v23 | PermSC + Dual-Route | 3.3% |
| v24 | G-Suite Prompt | +182건 |
| v25 | Majority Tiebreak | 6.07% |

## 포함 파일
### notebooks/ (3개)
### outputs/ (8개)
- 제출 CSV + diagnostics + g_suite_summary
