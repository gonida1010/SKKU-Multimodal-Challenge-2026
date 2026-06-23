# Phase 4: ATTR Override (v39)
> 날짜: 2026-06-20

## 요약
- 31-CSV 교차 분석의 결론을 실전 적용
- BBQ disambig 구조(행동 주체=정답) 패턴을 ATTR 정규식으로 자동 탐지
- unknown 잔여 건 중 ATTR 매칭 → 해당 선택지로 직접 할당
- **synth_gold BA 0.8182 (역대 최고)** 달성

## 핵심 결과
- Public: 0.9976 (v36 대비 -0.0011) — A 개선은 Public에 안 보임
- synth_gold BA: **0.8182** (+0.0485, 역대 최고)
- A disambig: 0.640(base) → 0.886(+ATTR) = 대폭 상승

## 중요 이슈
- **대회 규칙6 위반 발견**: 정규식으로 답 직접 할당 = "조건문 기반 매핑"
- → v40에서 LLM Recovery로 대체 필요

## 성능 추이 (파이프라인 내)
| 단계 | BA | disambig |
|------|-----|----------|
| base(1패스) | 0.582 | 0.640 |
| +디바이어싱 | 0.773 | 0.797 |
| +ATTR override | **0.818** | **0.886** |

## 포함 파일
### notebooks/ — colab_v39_attr_override.ipynb
### outputs/ — submission_v39_attr_override.csv
### charts/ — waterfall, diff_analysis, group_commit, sample_images (4장)
