# 논문 분석 + 다음 연구 방향
> 날짜: 2026-06-23

## 배경
v42(최종 확정) 이후, "프롬프트 기반 접근의 상한"을 넘어설 제3의 방법이 있는지 외부 논문을 조사.

---

## 조사한 논문 3편

### 1. DEBIASLENS (KAIST AI + Copenhagen + NVIDIA, 2026.02)
- **논문**: "Interpretable Debiasing of Vision-Language Models for Social Fairness"
- **방법**: SAE(Sparse Autoencoder)로 VLM 인코더 내부 "social neuron" 찾아 비활성화
- **결과**: CLIP Max Skew 9~16% 감소, InternVL2 성별 불균형 40~50% 감소, SBBench 83.83%→89.49%
- **파일**: `DEBIASLENS_논문요약.md`

### 2. VLBiasBench (ICT CAS + Peking Univ, 2024)
- **논문**: "A Comprehensive Benchmark for Evaluating Bias in Large Vision-Language Model"
- **방법**: 128K 합성 이미지, 9+2개 편향 카테고리, 개방형(VADER 감정) + 폐쇄형(정확도) 이원화 평가
- **결과**: 오픈소스 모델은 텍스트 과의존, ambig에서 편향 확대, Gemini가 ambig에서 우수
- **파일**: `VLBiasBench_논문요약.md`

### 3. VLMs are Biased (NeurIPS 2025 D&B Track)
- **논문**: "Vision Language Models are Biased"
- **방법**: VLM이 시각 입력 무시하고 사전 지식에 의존하는 문제 측정
- **결과**: counting accuracy 평균 17.05%, "double-check" 프롬프트는 +6점 밖에 안 됨
- **파일**: `VLMs_are_Biased_논문요약.md`

---

## 논문에서 추출한 적용 가능 기법

### 1순위: B패밀리 이미지 얼굴 크롭 ← DEBIASLENS
- **근거**: DEBIASLENS Table 3에서 SB-Syn-Crop(얼굴 크롭)이 gender bias 감소에 효과적
- **우리 문제**: Phase 5에서 발견한 B 과잉commit 원인 = 포즈/자세/몸짓
- **방법**: B패밀리 이미지에서 얼굴만 detect & crop → 포즈 정보 물리적 제거
- **구현**: face detection(MTCNN/RetinaFace) → 얼굴 bbox crop → 768px 리사이즈
- **위험**: disambig에서 몸짓이 실제 단서인 경우 (BBQ 스톡사진이라 가능성 낮음)

### 2순위: 메타인지 비교 프롬프트 ← VLBiasBench
- **근거**: Table IV에서 "Compared with [다른 집단]" 프레이밍이 숨겨진 편향 노출 (Δ=-0.097)
- **방법**: LLM 중재에서 "만약 이 사람들의 성별/인종이 바뀌면 답이 바뀌겠는가?" 직접 질문
- **차이점**: 우리 반사실은 라벨 교환 후 재추론. 이건 모델에게 직접 자기 편향을 물어보는 것.
- **구현**: DEBIAS_LLM 프롬프트에 한 줄 추가

### 3순위: 교차편향(Intersectional) 경고 강화 ← DEBIASLENS
- **근거**: Table 8에서 Gender×Age×Race 동시 타겟 = -11.4% (단일 Gender = -8.0%)
- **방법**: B 시각편향 경고에 "age, body type, clothing style"도 추가
- **구현**: 기존 bias_warn 문자열 확장

### 4순위: B패밀리 이미지 해상도 하향 ← DEBIASLENS + VLBiasBench
- **근거**: 이미지 인코더가 편향 주요 원인 + 오픈소스 텍스트 과의존
- **방법**: B패밀리만 768px → 384px/512px
- **위험**: B disambig에서 이미지가 도움이 되는 경우 손해

---

## 종합 결론

| 관점 | 결론 |
|------|------|
| 프롬프트 기반 | v42가 상한. 논문들도 프롬프트의 한계 확인. |
| 이미지 전처리 | 얼굴 크롭(1순위)이 가장 유망. 논문 실험 근거 있음. |
| 모델 수정 | SAE/LoRA 불가능 (대회 제약). |
| 큰 모델 | 프롬프트 재최적화 없이 모델만 키우면 역효과. |

**v42 = 최종 1순위 확정.** 추가 실험은 1~3순위 기법을 v43으로 테스트 가능.
