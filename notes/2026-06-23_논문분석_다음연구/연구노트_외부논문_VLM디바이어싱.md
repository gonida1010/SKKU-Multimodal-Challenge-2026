# 외부 논문 조사: VLM 디바이어싱 기법 (2026-06-23)

> 목적: v42 이후 추가 개선 가능성 탐색. 현재 파이프라인(프롬프트 기반)을 넘어선 제3의 방법이 있는지 조사.

---

## 1. DEBIASLENS (KAIST AI + Copenhagen + NVIDIA, 2026.02)

**핵심 아이디어**: Sparse Autoencoder(SAE)로 VLM 인코더 내부의 "social neuron"을 찾아 비활성화하여 편향 제거.

### 방법
1. VLM 인코더(이미지/텍스트) 마지막 레이어에 SAE를 부착, FairFace 등 얼굴 데이터로 학습
2. SAE 활성화 패턴에서 특정 인구통계(성별/나이/인종)에만 반응하는 "social neuron" 식별
3. 추론 시 해당 뉴런을 비활성화(γ=0), 원본 feature와 가중합(α=0.6)으로 디바이어싱

### 주요 결과
- CLIP: Max Skew 9~16% 감소
- InternVL2-8B: 성별 불균형 40~50% 감소, 일반 성능 4~10점 하락
- **SBBench(BBQ 기반)**: InternVL2 83.83% → 89.49% (BBQ bias probing accuracy)
- top-1 뉴런만 비활성화해도 전체 뉴런 비활성화와 유사 효과 → 최소 개입으로 최대 효과

### 우리 프로젝트 적용 가능성: **낮음**
- SAE 학습에 110K 에폭 필요 (별도 GPU 자원)
- Qwen3.5-9B의 비전 인코더에 SAE 부착하려면 vLLM 추론 파이프라인 수정 필요
- 대회 규칙("최종 답변은 LLM이 생성한 텍스트"): 모델 가중치 변경이 허용되는지 불확실
- 시간 제약 70분 이내에 SAE 학습은 불가능 (사전 학습 필요)

### 우리가 가져갈 수 있는 인사이트
1. **이미지 인코더가 편향의 주요 원인** (Weng et al. [82] 인용: "image features are the primary contributors to bias"). → 우리의 시각편향 경고가 올바른 방향
2. **해상도가 높을수록 이미지 인코더 디바이어싱이 중요** → 768px이 적절한 절충
3. **프롬프트 엔지니어링 기준선**: 논문에서 "be mindful that people should not be judged based on their race, gender, age..."를 테스트. 효과 있지만 모델 레벨보다 약함. → 우리의 Rule 11과 유사한 접근
4. **α=0.6 트레이드오프**: 디바이어싱과 일반 성능 사이 항상 트레이드오프 존재

---

## 2. 시각적 배경 제거 (Background Removal/Blurring)

**핵심 아이디어**: VLM이 이미지의 배경/주변 환경에 의해 편향된 답변을 생성. 배경을 블러/마스킹하여 중심 객체에만 집중하게 유도.

### 우리 프로젝트 적용 가능성: **낮음~중간**
- BBQ 데이터셋 이미지는 CrowdHuman 군중사진 + 스톡사진
- 우리 연구에서 확인한 편향 원인은 **배경이 아니라 사람의 포즈/자세/표정** (Phase 5 시각편향 분석)
  - 서 있는 사람 → authority, 앉아 있는 사람 → subordinate
  - 손짓/표정 풍부 → emotional
  - 프레젠테이션 중 → leader
- 배경 제거는 이 문제를 해결하지 못함
- 오히려 사람 영역만 크롭하면 포즈/표정이 더 부각될 수 있음

### 실험 가능한 변형
- 이미지 해상도를 극단적으로 낮추기 (384px → 256px) → 시각 정보 약화 → ambig에서 더 많이 unknown
- 하지만 C패밀리(이미지 없음)에 영향 없고, A패밀리 disambig에서 핵심 문장 인식이 약해질 위험

---

## 3. 프롬프트 기반 디바이어싱 (Prompt Engineering)

**핵심 아이디어**: "이미지 세부 사항에만 집중하라", "단계별로 다시 확인하라" 등의 지시를 추가.

### 우리 프로젝트 적용 가능성: **이미 최대한 활용 중**

현재 v42 파이프라인에서 사용 중인 프롬프트 디바이어싱 기법:

| 기법 | 구현 위치 | 효과 |
|------|---------|------|
| Rule 1-10 (소거법, 증거 기반 추론) | SYSTEM_PROMPT | 기본 정확도 확보 |
| Rule 11 역할매핑 ("running the meeting = chair") | SYSTEM_PROMPT | C패밀리 board chair +5건 |
| 반사실 추론 (집단/성별 라벨 교환) | CF pass | 편향 탐지 핵심 |
| LLM 중재 + B 시각편향 경고 | DEBIAS_LLM | B 과잉commit 억제 |
| Recovery 강화 ("Abstaining is WRONG") | LLM_RECOVERY 1단계 | 95.8% 회수 |
| Binary choice ("ONLY two answers") | LLM_RECOVERY 2단계 | 추가 20건 회수 |

**DEBIASLENS 논문의 프롬프트 엔지니어링 기준선과 비교**: 논문에서 사용한 단일 지시문("be mindful...")보다 우리 파이프라인이 훨씬 정교하고 다단계. 이미 프롬프트 기반 접근의 상한에 근접해 있다고 판단.

---

## 4. 기타 참조 기법

### 반사실 디바이어싱 (Counterfactual Debiasing)
- Howard et al. (2024): "Uncovering bias in large VLMs at scale with counterfactuals"
- **우리 프로젝트**: v35부터 반사실 패스 도입, v40에서 LLM 중재로 발전. 이미 핵심 파이프라인.

### 인과 매개 분석 (Causal Mediation Analysis)
- Weng et al. (2024): 이미지 feature가 VLM 편향의 주요 원인
- **시사점**: 이미지를 아예 제거하면 편향이 줄지만 성능도 하락. 우리의 모달리티 ablation 실험(A패밀리 이미지 의존 16.4%)과 일치.

### LoRA Fine-Tuning
- Girrbach et al. (2025): LoRA가 LVLM 디바이어싱에서 가장 유망
- **적용 가능성**: Qwen3.5-9B에 BBQ 편향 데이터로 LoRA 튜닝 가능하지만, 대회에서 별도 fine-tuning 데이터/자원 필요. 현재 인프라에서 비현실적.

---

## 5. 종합 평가

### 우리가 이미 쓰고 있는 것
- 프롬프트 기반 디바이어싱 (Rule 1-11, 시각편향 경고) ✓
- 반사실 추론 (CF pass + LLM 중재) ✓
- Recovery (강화 프롬프트 + binary choice) ✓

### 적용 가능하지만 효과 불확실한 것
- 이미지 전처리 변형 (해상도 조절, 크롭 전략)
- 더 공격적인 시각편향 경고 프롬프트 변형

### 현실적으로 불가능한 것 (대회 제약)
- SAE 기반 뉴런 레벨 디바이어싱 (DEBIASLENS)
- LoRA fine-tuning
- 모델 가중치 변경/pruning

### 결론
**v42가 프롬프트 기반 접근의 실질적 상한에 도달했다.** 추가 개선은 가능하지만 marginal (1-2건 수준). 근본적 돌파구는 모델 레벨 개입(SAE, LoRA)이 필요하나 대회 제약상 불가.

---

## 참고 논문

1. An et al. "Interpretable Debiasing of Vision-Language Models for Social Fairness" (arXiv:2602.24014, 2026.02) — DEBIASLENS
2. Howard et al. "Uncovering bias in large VLMs at scale with counterfactuals" (arXiv:2405.20152, 2024)
3. Weng et al. "Images speak louder than words: Understanding and mitigating bias in VLM from a causal mediation perspective" (EMNLP 2024)
4. Girrbach et al. "Revealing and reducing gender biases in VLAs" (ICLR 2025)
5. Gerych et al. "BendVLM: Test-time debiasing of vision-language embeddings" (NeurIPS 2024)
6. Parrish et al. "BBQ: A hand-built bias benchmark for QA" (arXiv:2110.08193, 2021) — 대회 데이터 원본
