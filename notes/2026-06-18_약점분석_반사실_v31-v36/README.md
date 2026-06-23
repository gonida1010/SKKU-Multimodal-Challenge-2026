# Phase 3~4: 약점 분석 + 반사실 디바이어싱 (v31~v36)
> 기간: 2026-06-18 ~ 06-19

## 요약
- **31-CSV 교차 분석**: 18개 버전 제출 CSV 전수 비교 → 2,052건(24%) 불일치 발견
- v31: Grounding OFF → **Public 0.9986** (grounding 제거가 오히려 개선)
- v32: 1-pass 베이스라인 ablation 측정
- v34~v35: **반사실(Counterfactual) 디바이어싱** 도입
  - A패밀리: 집단 라벨 교환 (African↔European)
  - B패밀리: 성별 라벨 교환 (man↔woman)
- v36: CF + Recovery 3패스 → **Public 0.9987** (시간 초과 문제)
- v38: A패밀리 선별 Recovery (70분 내 충족 시도)

## 핵심 발견
- **B패밀리 1,008건 = 최대 미개척 영역** (31-CSV 분석)
- **BBQ disambig 구조**: 행동 주체(target) = 정답 (99.7%)
- ATTR 패턴 정규식: `\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b`
- **반사실 불변성**: 라벨만 바꿨을 때 답이 바뀌면 = 편향에 의한 답
- A패밀리 10.6% flip = 잔존 집단편향(Hidden 리스크)

## 성능 추이
| 버전 | Public | synth_gold BA | 핵심 |
|------|--------|---------------|------|
| v31 | **0.9986** | 0.770 | grounding OFF |
| v35 | — | — | 반사실 디바이어싱 |
| v36 | **0.9987** | 0.770 | CF + recovery (90분) |

## 포함 파일
### notebooks/ (8개) — v31, v32, v34, v35, v36, v38, 회수.ipynb
### outputs/ (6개) — 제출 CSV + diagnostics
### 연구노트/ (4개)
- `MASTER_HANDOFF.md` — 전체 프로젝트 마스터 핸드오프
- `HANDOFF_v28_v30.md` — v28~v30 핸드오프
- `HANDOFF_v31_grounding_analysis.md` — grounding 분석
- `연구노트_Phase4_약점분석_v39.md` — 31-CSV 교차 분석 + ATTR 패턴
### analysis_wrong/ — 오답 이미지(A/B/C) + 오답 CSV
