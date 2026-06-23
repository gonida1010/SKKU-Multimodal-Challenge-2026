# Phase 2b: 규칙 체계화 + COREVQA (v26~v29)
> 기간: 2026-06-15 ~ 06-16

## 요약
- v26: COREVQA 연구 — 일반화 성능 프록시 벤치마크 설계
- v27: **Descriptor Grounding** + Rule 1~10 체계적 정립 (**Public 0.9983**)
  - 12가지 프롬프트 규칙 A/B 테스트로 최적 조합 도출
  - 소거법, 증거 기반 추론, 고정관념 금지 등 체계화
- v28: Count Probe — 이미지 내 인원수 기반 판단 실험
- v29/v29b: Precision Gate — 정밀 게이트 개선

## 핵심 발견
- **Rule 1~10 체계 확립**: 이후 모든 버전의 SYSTEM_PROMPT 기반
- v28 Count Probe: 이미지 인원수로 commit 뒤집기는 **틀림** (이미지는 decoupled 스톡사진)
- COREVQA: 복합 부정/연언문에서 False→True 편향 발견
- 이미지 해상도별 COREVQA 비교: 768px이 최적

## 성능 추이
| 버전 | Public | 핵심 |
|------|--------|------|
| v27 | **0.9983** | descriptor grounding + Rule 1~10 |
| v29 | — | precision gate 실험 |

## 포함 파일
### notebooks/ (6개)
### outputs/ (11개)
### charts/ — v26 연구 차트 + v28 리뷰 이미지
### corevqa/ — 해상도별(512/768/1024/1280) COREVQA 결과 + 오답 HTML
