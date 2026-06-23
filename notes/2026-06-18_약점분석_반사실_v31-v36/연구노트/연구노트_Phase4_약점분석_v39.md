# Phase 4: 체계적 약점 분석 + 대폭 개선 (v39)

> 작성: 2026-06-21 | 기간: 2026-06-19 ~ 2026-06-20

## 배경: 왜 이 연구가 필요했는가

v31(Public 0.9986)이 안정적 베이스라인이었지만, **Private 일반화**를 올리려면 어떤 샘플이 어렵고 왜 어려운지를 체계적으로 파악해야 했다. "감으로" 개선하는 것은 한계가 있었다.

## 1단계: 31-CSV 교차 분석 (버전 간 불일치 전수 조사)

### 방법

v15부터 v36까지 18개 버전의 제출 CSV를 전수 교차 비교했다. 8,500건 각각에 대해 "몇 개 버전이 같은 답을 냈는가"를 계산.

### 핵심 발견

- **2,052건(24%)이 버전 간 1개 이상 불일치** — 나머지 76%는 모든 버전이 동일
- 패밀리별 분포:
  - **A패밀리: 984건** — commit 490 / unknown 494
  - **B패밀리: 1,008건** — 최대 미개척 영역
  - **C패밀리: 60건** — 거의 해결됨
- **Consensus(80% 이상 버전 일치)**: A=558, B=768, C=49 — 이 건들은 "대부분의 프롬프트가 맞추는" 쉬운 건
- synth_gold(합성 정답) 분석: A disambig 931건, ambig 44건

### 의미

"B패밀리 1,008건이 최대 미개척 영역"이라는 발견이 이후 v40-v41의 B패밀리 시각 편향 연구로 이어졌다.

---

## 2단계: A패밀리 Disambig 구조 분석

### 문제 정의

A패밀리 disambig(정답 단서가 있는 문제)에서 Qwen이 unknown으로 답하는 건이 많았다. 왜 명확한 단서가 있는데 모델이 답을 못 맞추는가?

### BBQ 데이터셋 구조 연구

BBQ(Bias Benchmark for QA) 논문을 분석하여, disambig 문맥의 구조적 패턴을 발견:

- disambig 문맥에는 **"An [그룹] person [행동 동사]"** 패턴이 존재
- 예: "An African American person was seen shoplifting." → 행동 주체가 명시됨
- 이 행동 주체(target)가 정답인 비율: **99.7%** (964건 중 961건)

### ATTR 패턴 정규식

```python
ATTR_RE = re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b")
```

이 정규식으로 문맥에서 행동 주체의 집단명을 추출할 수 있었다.

### 중요 교훈

이 발견이 v39의 ATTR override와 v40-v41의 LLM recovery의 핵심 기반이 되었다. "문맥에 답이 있는데 모델이 놓치는" 패턴을 정확히 잡아내는 열쇠.

---

## 3단계: v39 — ATTR Override

### 설계

1. base pass + 반사실 pass (v36과 동일)
2. 디바이어싱 후 잔여 unknown 중 ATTR 패턴 매칭 건에 대해 → 정규식으로 행동 주체(tgt) 식별 → 해당 선택지로 직접 할당

### 결과

- **Public: 0.9975833** (v36 대비 -0.0011)
- **synth_gold BA: 0.8182** (역대 최고, +0.0485)
- A패밀리 unknown 414건 → ATTR 패턴으로 tgt 식별하여 commit

### synth_gold BA 추이 (v39 파이프라인 내)

| 단계 | BA | ambig | disambig |
|------|-----|-------|----------|
| base(1패스) | 0.5818 | 0.524 | 0.640 |
| +디바이어싱 | 0.7733 | 0.750 | 0.797 |
| +ATTR override | **0.8182** | 0.750 | **0.886** |
| v31 (기존 최선) | 0.7699 | 0.750 | 0.790 |

### 트러블슈팅: Public 하락 원인

v39의 Public이 v36보다 낮아진 이유:
- A패밀리 개선은 Public에 안 보임 (A=Private 영역)
- B패밀리에서 58건이 commit→unknown으로 바뀜 (디바이어싱의 부작용)
- 결과적으로 synth_gold BA는 역대 최고인데 Public은 하락하는 **"길 B" 트레이드오프** 확인

### 규칙 위반 문제 발견

v39의 ATTR override는 **대회 규칙6 위반**:
- "최종 답변은 생성형 언어모델(LLM)에 의해 생성된 텍스트여야 합니다"
- 정규식으로 직접 답을 할당하는 것은 "조건문 기반 매핑"에 해당
- → v40에서 LLM recovery pass로 대체 필요

### 차트 이미지

- `outputs/charts_v39/1_waterfall.png` — 파이프라인 단계별 BA 변화
- `outputs/charts_v39/2_diff_analysis.png` — v39 vs v31 불일치 분석
- `outputs/charts_v39/3_group_commit.png` — 집단별 commit rate
- `outputs/charts_v39/4_sample_images.png` — 불일치 샘플 이미지 30건
