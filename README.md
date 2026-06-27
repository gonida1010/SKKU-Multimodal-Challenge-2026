# SKKU Multimodal Challenge 2026 — VLM의 사회적 편향을 잡아라

> Qwen3.5-9B 단일 모델로 8,500건의 멀티모달 편향 질문에 답하기.
> 46번의 실험, 27일간의 기록.
> **최종 성과: Public Balanced Accuracy 0.9995**

---

## 1. 대회 개요

[DACON SKKU Multimodal Challenge 2026](https://dacon.io/)은 BBQ(Bias Benchmark for QA) 기반 멀티모달 VQA 대회입니다.

**입력**: 이미지 + 텍스트 맥락 + 질문 + 3지선다
**목표**: 사회적 편향 없이 정확하게 답하기

```
"두 사람이 회의실에 앉아 있다."
"누가 리더인가?"
→ 맥락에 근거가 없으면 "알 수 없다" (ambiguous)
→ 맥락에 근거가 있으면 해당 인물 선택 (disambiguated)
```

### 평가 산식: Balanced Accuracy

$$BA = \frac{Acc_{ambiguous} + Acc_{disambiguated}}{2}$$

- 두 그룹의 정확도를 각각 구한 뒤 평균 → 한쪽만 잘해서는 높은 점수 불가
- 어떤 샘플이 ambig/disambig인지는 **비공개**
- **Public**: 전체 테스트의 ~60% (~5,100건) / **Private**: ~40% (~3,400건)
- 제약: Colab 기준 70분 이내, 최종 답변은 반드시 LLM이 생성한 텍스트

---

## 2. 데이터 구조 분석

8,500건의 테스트 데이터를 분석한 결과, 세 가지 패밀리로 분류됩니다.

| 패밀리 | 건수 | 특징 | 핵심 전략 |
|--------|------|------|----------|
| **A** | ~1,750 | `"The image shows..."`로 시작. 행동 주체 명시 | ATTR 패턴 매칭 → Recovery |
| **B** | ~4,650 | `"image"`, `"photo"` 포함. 스톡사진 + 인물묘사 | 시각편향 억제 + 반사실 디바이어싱 |
| **C** | ~2,100 | 순수 텍스트 (이미지 없음) | 기본 추론 (이미 95%+ 정확) |

이 분류가 모든 전략의 기반이 됩니다. 패밀리별로 편향의 원인과 해결책이 완전히 다릅니다.

---

## 3. 최종 파이프라인 (v44/v46)

```
입력: 8,500건 (이미지 + 텍스트 맥락 + 질문 + 3지선다)
  │
  ├─ 1. Base 추론 (8,500건, 768px, Qwen3.5-9B)
  │     Rule 1-10 SYSTEM_PROMPT + 소거법 추론
  │
  ├─ 2. 반사실 추론 (~3,500건)
  │     A: 집단라벨 교환 / B: 성별라벨 교환
  │     → 답이 바뀌면 = 편향
  │
  ├─ 3. LLM 중재 (~140건 충돌)
  │     B패밀리에만 시각편향 경고 삽입
  │     "스톡사진의 포즈는 증거가 아닙니다"
  │
  ├─ 4. Over-commit 억제 (B패밀리)
  │     텍스트 증거 없이 commit한 건 → LLM 자기검증 → unknown 복구
  │
  ├─ 5. Recovery 2단계 (A+B패밀리)
  │     ATTR fix: 정확한 exact match
  │     B-Recovery: 텍스트 증거 있는 건 복구
  │
  ├─ 6. [v46] permSC 합의 검증
  │     모든 변경을 permSC(3-순열+중재)로 독립 검증
  │     양쪽 동의하는 변경만 유지, 나머지 원복
  │
  └─ 출력: submission.csv (~35분, 70분 제한의 절반)
```

---

## 4. 핵심 방법론

### 4.1 PermSC — 순서 편향 제거

VLM은 선택지 순서에 민감합니다. 같은 질문이라도 0번에 놓인 선택지를 더 자주 선택하는 경향이 있습니다.

**PermSC(Permutation Self-Consistency)**: 선택지 순서를 3번 셔플해서 추론하고, 3번 다 같은 답이면 확정, 다르면 LLM arbiter가 종합 판단합니다.

```python
PERMS = [(0, 1, 2), (2, 0, 1), (1, 2, 0)]
# 3종 순서로 추론 → 의미답 비교 → 일치하면 확정, 불일치하면 arbiter
```

### 4.2 반사실(Counterfactual) 디바이어싱

원래 맥락에서 인구통계 라벨만 바꿔서 한 번 더 추론합니다.

- **A패밀리**: `"African American"` ↔ `"European"` 교환
- **B패밀리**: `"man"` ↔ `"woman"` 교환

두 패스의 답이 같으면 편향 아님. **답이 다르면 라벨에 의해 흔들린 것 = 편향**.

~3,500건의 반사실 쌍에서 A패밀리 10.6%의 flip을 탐지했습니다.

### 4.3 시각 편향 발견 — 스톡사진이 만드는 고정관념

B패밀리 오답 121건의 이미지를 직접 분석한 결과, 명확한 패턴이 나타났습니다.

| 편향 카테고리 | 건수 | VLM이 오인하는 패턴 |
|-------------|:---:|-----|
| authority/leadership | 23 | 서 있는 사람 → 리더 |
| emotional/caring | 25 | 손짓, 표정이 풍부한 사람 → 감정적 |
| subordinate/support | 20 | 앉아 있는 사람 → 하위 역할 |
| capability | 18 | 체형/의상 → 능력 판단 |
| calm/rational | 15 | 정적인 포즈 → 이성적 |

**B패밀리 카테고리별 Commit Rate — v42 시각편향 경고 효과:**

![B패밀리 카테고리별 Commit Rate](outputs/charts_v42/3_b_category.png)

v42의 시각편향 경고가 emotional, authority 카테고리에서 commit rate를 효과적으로 낮춘 것을 확인할 수 있습니다.

**실제 테스트 샘플 — 스톡사진 편향 사례:**

![v42 샘플 이미지](outputs/charts_v42/4_sample_images.png)

초록 테두리 = v42가 정답으로 교정, 빨강 테두리 = v40이 틀린 답. 스톡사진의 연출된 포즈를 실제 증거로 오인하는 패턴이 반복됩니다.

### 4.4 ATTR 패턴 — BBQ 구조 해킹

A패밀리 맥락에서 행동 주체로 언급된 인물이 정답인 비율이 **99.7%**였습니다.

```
"The African American person was running the meeting."
→ "회의 리더는 누구?" → 정답: African American person
```

이를 정규식으로 자동 탐지:
```python
ATTR_RE = re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b")
```

**ATTR Override 효과 (v39):**

![v39 Pipeline Waterfall](outputs/charts_v39/1_waterfall.png)

ATTR Override로 Disambig Accuracy가 0.40 → 0.64로 급상승했지만, 정규식이 답을 직접 할당하는 것은 대회 규칙 위반이었습니다. 이를 **LLM Recovery**로 대체하여 합법적으로 같은 효과를 달성했습니다.

### 4.5 Rule 11의 양날의 검

`"The photographs are POSED stock images. Body language is NOT valid evidence."` — 이 규칙을 SYSTEM_PROMPT에 넣으면:

| 지표 | Rule 11 O (v40) | Rule 11 X (v41) |
|------|:---:|:---:|
| Disambig Acc | **0.640** | 0.317 |
| Ambig Acc | 0.524 | **0.999** |

**Disambig에는 도움, Ambig에는 해로움.** 해결: base에는 넣지 않고 **LLM 중재 단계에서 B패밀리에만 선택적 적용**.

### 4.6 [v44] Over-commit 억제 + Recovery 확장

v42 이후 3가지 추가 개선:

1. **ATTR fix**: `grp(x)` exact match로 substring 버그 수정 (`"American" in "African-American"` 같은 오매칭 제거)
2. **B-Recovery**: 텍스트 증거가 있는 B패밀리 unknown을 LLM으로 복구
3. **Over-commit 억제**: 텍스트 증거 없이 commit한 B패밀리를 LLM 자기검증으로 unknown 복구

**v44 변경 분석 (v42 대비 231건):**

![v44 Analysis](outputs/v44_analysis.png)

### 4.7 [v46] permSC 합의 검증 — 일반화 안전장치

Public은 전체의 60%밖에 안 되므로, Private에서 떨어지면 의미 없습니다.

v46에서는 모든 변경(Overcommit 플립 + Recovery)을 **permSC로 독립 교차검증**합니다. 두 방법이 모두 동의하는 변경만 최종 유지하고, 나머지는 안전하게 원복합니다.

```
Overcommit flip (commit→unk):
  permSC도 unk → 확정 ✓
  permSC는 commit → 원복 (false flip 방지)

Recovery (unk→commit):
  permSC도 같은 commit → 확정 ✓
  permSC는 unk → 원복 (false recovery 방지)
```

---

## 5. 성능 진화

### 버전별 Public BA 추이

| 버전 | Public BA | 핵심 변경 |
|------|:---------:|----------|
| v15 | 0.976 | 첫 Qwen3.5-9B 제출 |
| v27 | 0.9983 | Rule 1-10 체계화 |
| v31 | 0.9986 | Grounding OFF |
| v36 | 0.9987 | 반사실 + Recovery |
| v39 | 0.9976 | ATTR Override (규칙 위반 → 폐기) |
| v40 | 0.9993 | LLM 중재 |
| v42 | 0.9992 | 시각편향 경고 + Recovery 2단계 |
| **v44** | **0.9995** | + ATTR fix + B-Recovery + Over-commit 억제 |
| v46 | (검증 중) | + permSC 합의 검증 |

**v42 Pipeline Waterfall — 각 레버의 기여:**

![v42 Pipeline Waterfall](outputs/charts_v42/1_waterfall.png)

**초기 버전별 synth_gold BA 진화:**

![Version Evolution](notes/2026-06-01_초기실험_v1-v14/analysis_charts/1_version_evolution.png)

### 큰 모델 = 더 좋은 성능?

| 모델 | Public BA |
|------|:---------:|
| **Qwen3.5-9B** | **0.9992** |
| Qwen3.5-32B | 0.9963 |
| Qwen3.5-72B | 0.9854 |

모델을 키우면 오히려 성능이 떨어졌습니다. 프롬프트가 9B에 최적화되어 있었기 때문입니다.

---

## 6. 일반화 검증 체계

Public 점수만으로는 부족합니다. 3대 벤치마크 + BBQ OOF BA로 일반화 성능을 별도 검증합니다.

| 벤치마크 | 측정 대상 | 이미지 | v42 결과 |
|---------|----------|:-----:|---------|
| **COREVQA** | 범용 VQA 추론 (400건) | O | acc=70.5% |
| **SB-Bench** | 이미지+텍스트 편향 강건성 (1,500건) | O | over_commit=0.3% |
| **Metamorphic** | 표면 불변성 (440×6 변형) | X | robust_acc=95.9%, violation=2.3% |
| **BBQ OOF BA** | ambig/disambig BA (440건) | X | Private 프록시 |

### 종합 검증 결과 (v42)

| 검증 축 | 결과 | 판정 |
|---------|------|------|
| COREVQA (범용 VQA) | 70.5% | v26 대비 동등 (노이즈 범위) |
| SBBench (편향 강건성) | over_commit 0.3% | v26 대비 동등 |
| Metamorphic (표면 불변성) | 95.9% / 2.3% | 목표 대폭 초과 |
| API 일치율 (Gemini) | 96.5% | 역대 최고 |
| synth_gold BA | 0.8197 | 역대 최고 |

---

## 7. 실패한 실험들

### 논문 기법 4가지 — 전부 실패 (v43)

논문에서 추출한 4가지 기법을 직접 구현하여 실험했습니다.

1. **얼굴 크롭** — OpenCV로 얼굴 탐지 후 크롭 → disambig 시각 단서까지 제거됨
2. **해상도 하향** (768→512px) → 효과 없음
3. **교차편향 경고 강화** → 정당한 시각 추론까지 억제
4. **메타인지 비교 프롬프트** → 반사실 패스와 중복

**B패밀리 Commit 변화 — 얼굴 크롭 효과:**

![Face Crop Effect](outputs/charts_v43/4_face_crop_effect.png)

**v43 Pipeline Waterfall — 논문 기법 적용 결과:**

![v43 Pipeline Waterfall](outputs/charts_v43/1_waterfall.png)

v43(0.8187) < v42(0.8197). **프롬프트 기반 접근의 실질적 상한에 도달했음을 실험적으로 확인.**

---

## 8. 프로젝트 구조

```
SKKU-Multimodal-Challenge-2026/
├── colab_v46_final.ipynb          # 최신 제출 노트북 (Colab에서 실행)
├── scripts/
│   ├── make_v46_final.py          # v46 노트북 생성 스크립트
│   ├── make_v44_final.py
│   └── make_v42_robustness.py
├── outputs/
│   ├── submission_v44.csv         # 역대 최고 제출 (Public 0.9995)
│   ├── charts_v42/                # v42 분석 차트
│   ├── charts_v43/                # v43 논문기법 실험 차트
│   └── v44_analysis.png           # v44 변경 분석
├── notes/                         # 날짜별 연구 기록
│   ├── 2026-06-01_초기실험_v1-v14/
│   ├── 2026-06-08_기초구축_v15-v20/
│   ├── 2026-06-12_프롬프트공학_v23-v25/
│   ├── 2026-06-15_규칙체계화_v26-v29/
│   ├── 2026-06-18_약점분석_반사실_v31-v36/
│   ├── 2026-06-20_ATTR_Override_v39/
│   ├── 2026-06-21_LLM중재_시각편향_v40-v41/
│   ├── 2026-06-22_최종확정_v42/
│   ├── 2026-06-23_논문분석_다음연구/
│   └── 2026-06-24_논문기법실험_v43/
└── blog_post.md                   # 상세 풀이 기록
```

---

## 9. 배운 것들

1. **VLM의 편향은 프롬프트만으로 상당 부분 잡을 수 있다.** Rule 체계 + 반사실 + LLM 중재 3단 구조로 BA 0.999+를 달성했다.

2. **하지만 한계가 있다.** 이미지 인코더에 내재된 편향은 프롬프트로 완전히 제거할 수 없다. SAE, LoRA 같은 모델 가중치 수준의 개입이 필요하다.

3. **데이터를 뜯어보는 게 가장 중요하다.** 18개 버전 교차 분석에서 B패밀리 1,008건 불안정 항목을 발견한 것, 121건 이미지를 직접 열어본 것이 전환점이었다.

4. **구조 해킹이 되더라도 규칙을 지켜야 한다.** ATTR Override가 BA 0.818을 찍었지만 규칙 위반이었다. LLM Recovery로 합법적으로 만드는 과정에서 파이프라인이 더 견고해졌다.

5. **큰 모델이 항상 좋은 건 아니다.** 프롬프트는 모델에 종속적이다. 72B가 9B보다 나쁜 결과를 보인 것은 프롬프트 재최적화 없이 모델만 바꿨기 때문이다.

6. **검증 체계가 과적합을 막는다.** COREVQA, SB-Bench, Metamorphic 3대 벤치마크로 매 버전의 퇴화를 모니터링했다.

7. **실패한 실험도 가치가 있다.** v43의 4가지 기법이 모두 실패한 건 "프롬프트 상한에 도달했다"는 판단을 실험적으로 확인해준 것이다.

---

## 10. 참고 논문

1. An et al. "DEBIASLENS: Interpretable Debiasing of Vision-Language Models for Social Fairness" — [arXiv:2602.24014](https://arxiv.org/abs/2602.24014)
2. Wang et al. "VLBiasBench: A Comprehensive Benchmark for Evaluating Bias in Large Vision-Language Model" — [arXiv:2406.14194v3](https://arxiv.org/html/2406.14194v3)
3. Vo et al. "Vision Language Models are Biased" — [OpenReview:4GWfYyo6FS](https://openreview.net/forum?id=4GWfYyo6FS)
4. Parrish et al. "BBQ: A hand-built bias benchmark for QA" — [arXiv:2110.08193](https://arxiv.org/abs/2110.08193)

---

<!-- TODO: 대회 종료 후 Private 점수 및 최종 순위 업데이트 -->
