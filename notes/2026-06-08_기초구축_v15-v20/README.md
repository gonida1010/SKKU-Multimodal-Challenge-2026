# Phase 1: 기초 구축 (v15~v20)
> 기간: 2026-06-08 ~ 06-10

## 요약
- v15: 768px 기본 베이스라인 확립
- v17: Evidence Gate 규칙 추가 (v15 대비 147건 변경)
- v18: Context-First + Thinking OFF → 안정적 파싱
- v19: Context Implication Gate 실험
- v20: Commit Recovery 도입 (1024px) + JPEG Q95
- Gemma4-12B 비교 실험 (Qwen이 우세 확인)

## 핵심 발견
- thinking 모드 비활성화(`enable_thinking: False`)가 출력 안정성 대폭 향상
- 768px이 속도/정확도 균형점 (1024px은 미미한 개선에 비해 속도 저하)
- Commit Recovery: unknown 답변에 재추론 → 정답 회수 가능성 확인

## 성능 추이
| 버전 | Public | 핵심 변경 |
|------|--------|----------|
| v15 | (베이스라인) | 768px 기본 |
| v17 | +evidence gate | 147건 변경 |
| v18 | +context-first | thinking OFF |
| v20 | +commit recovery | 1024px, Q95 |

## 포함 파일
### notebooks/ (6개)
- v15, v18, v18_gemma4, v19, v20, v20_1

### outputs/ (12개)
- 각 버전 제출 CSV + audit/diagnostics CSV
