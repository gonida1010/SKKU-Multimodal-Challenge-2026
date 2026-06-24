# Phase 7: 논문 기법 실험 (v43) — 효과 없음 확인
> 날짜: 2026-06-24

## 목적
06-23에 분석한 3편의 외부 논문에서 추출한 4가지 기법을 v42 파이프라인에 적용하여 성능 개선 가능성 검증.

## 참고 논문

| 논문 | 저자 | 링크 |
|------|------|------|
| **DEBIASLENS**: Interpretable Debiasing of VLMs for Social Fairness | An et al. (KAIST AI + Copenhagen + NVIDIA), 2026.02 | [arXiv:2602.24014](https://arxiv.org/abs/2602.24014) |
| **VLBiasBench**: A Comprehensive Benchmark for Evaluating Bias in Large VLM | Wang et al. (ICT CAS + Peking Univ), 2024 | [arXiv:2406.14194v3](https://arxiv.org/html/2406.14194v3) |
| **VLMs are Biased** | Vo et al., NeurIPS 2025 Datasets and Benchmarks | [OpenReview](https://openreview.net/forum?id=4GWfYyo6FS) |

## 적용한 4가지 기법

### 기법 1: B패밀리 얼굴 크롭 (DEBIASLENS)
- **근거**: DEBIASLENS Table 5 — SB-Syn-Crop(얼굴 크롭) 학습 데이터가 gender bias 감소에 효과적
- **구현**: OpenCV Haar Cascade로 B패밀리 이미지에서 얼굴 탐지 -> 가장 큰 얼굴 + 75% 패딩 크롭
- **의도**: 포즈/자세/체형 등 시각 편향 단서를 물리적으로 제거

### 기법 2: B패밀리 해상도 하향 (DEBIASLENS + VLBiasBench)
- **근거**: 이미지 인코더가 편향 주요 원인 (DEBIASLENS), 오픈소스 모델 텍스트 과의존 (VLBiasBench)
- **구현**: B패밀리 이미지 768px -> 512px (얼굴 크롭 후 또는 전체 축소)
- **의도**: 이미지 인코더에 전달되는 시각 정보량 축소 -> 편향 기여 감소

### 기법 3: 교차편향(Intersectional) 경고 강화 (DEBIASLENS)
- **근거**: DEBIASLENS Table 8 — 교차(Gender x Age x Race) 타겟 -11.4% > 단일 Gender -8.0%
- **구현**: bias_warn에 `apparent age, body type, clothing style, grooming` 추가
- **의도**: 성별뿐 아니라 나이/체형/의상에 의한 편향도 동시 억제

### 기법 4: 메타인지 비교 프롬프트 (VLBiasBench)
- **근거**: VLBiasBench Table IV — "Compared with [집단]" 비교 프레이밍이 숨겨진 편향 노출 (delta=-0.097)
- **구현**: LLM 중재 프롬프트에 "Would your answer change if the people belonged to different demographic groups?" 추가
- **의도**: 모델 스스로 자기 편향을 인지하도록 유도

## 실험 결과

### 성능 비교
| 지표 | v42 (기존 최선) | v43 (논문 기법) | 차이 |
|------|----------------|----------------|------|
| **synth_gold BA** | **0.8197** | 0.8187 | **-0.0010** |
| Ambig Acc | 0.999 | 0.999 | 0 |
| Disambig Acc | **0.641** | 0.639 | **-0.002** |
| 실행 시간 | ~35분 | ~28.5분 | -6.5분 |

### B패밀리 Commit 변화
| 버전 | B Commit | B Commit Rate |
|------|----------|---------------|
| base | 1059 | 22.8% |
| v42 | 1099 | 23.6% |
| **v43** | **1047** | **22.5%** |

- v43이 v42보다 B commit 52건 감소
- v43 vs v42 diff 155건 중 151건이 B패밀리

### Family별 Commit 변화
```
A: commit 511 -> 1041/1750 (+530)  # recovery에 의한 것, v42와 동일
B: commit 1047 -> 1047/4652 (+0)   # 변화 없음
C: commit 1999 -> 1999/2098 (+0)   # 변화 없음
```

### LLM 중재 상세
- 충돌 141건 (B패밀리 82건)
- 58건 변경, 이 중 B->unk 16건 (교차편향+메타인지 효과)
- Recovery: 530/532건 (99.6%) — v42의 534/535 (99.8%)보다 약간 낮음

## 실패 원인 분석

### 1. 얼굴 크롭이 역효과
- 얼굴 크롭 + 512px가 B패밀리 이미지 정보를 과도하게 제거
- disambig에서 맞아야 할 시각 단서(인물 식별 등)까지 손실
- base 시점에서 이미 B commit이 1059 -> 1047로 12건 감소 (정답 포함)

### 2. 교차편향 경고 과잉 억제
- age/body type/clothing 경고가 정당한 시각 추론까지 억제
- B->unk 16건 중 일부는 정답이 commit이어야 할 disambig 건

### 3. 메타인지 프롬프트의 한계
- VLBiasBench 논문은 "비교 프레이밍이 편향을 노출한다"고 보고
- 하지만 우리 파이프라인은 이미 반사실 패스로 편향 탐지 중
- 메타인지 질문은 이미 작동하는 메커니즘과 중복

### 4. 핵심 교훈
> **프롬프트 기반 접근의 상한에 도달했다는 기존 판단(06-23) 재확인.**
> 
> 논문들이 제시한 기법의 핵심(SAE로 뉴런 비활성화, 모델 가중치 수정)은
> 대회 제약(70분, 모델 수정 불가) 내에서 적용 불가.
> 적용 가능한 범위(이미지 전처리, 프롬프트 변형)로는 v42를 넘지 못함.

## 결론
- **v42가 최종 제출 1순위 유지**
- 4가지 논문 기법 모두 우리 파이프라인에서 유의미한 개선 없음
- 프롬프트/전처리 수준의 변형으로는 추가 개선 불가 확인

## 포함 파일
### notebooks/
- `colab_v43_paper_techniques.ipynb` — v43 실험 노트북
- `make_v43_notebook.py` — 노트북 생성 스크립트

### outputs/
- `submission_v43.csv` — v43 제출 파일 (Colab 실행 후 복사 필요)

### charts/
- `1_waterfall.png` — Pipeline Waterfall (v43 vs v42 vs v40 vs v31)
- `2_diff_analysis.png` — v43 vs v42/v40/v31 diff 분석
- `3_b_category.png` — B패밀리 카테고리별 Commit Rate
- `4_face_crop_effect.png` — 얼굴 크롭 효과 (B Commit 변화)
- `5_sample_images.png` — v43 vs v42 불일치 샘플 (30건)
