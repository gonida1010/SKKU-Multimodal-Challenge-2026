# DEBIASLENS: Interpretable Debiasing of Vision-Language Models for Social Fairness

> An et al. (KAIST AI + Copenhagen + NVIDIA), arXiv:2602.24014, 2026.02

## 핵심 아이디어
SAE(Sparse Autoencoder)로 VLM 인코더(이미지/텍스트) 마지막 레이어에서 **"social neuron"**을 찾아 비활성화하여 편향 제거.

## 방법론 (3단계)

### Stage 1: SAE Training
- VLM 인코더의 post-residual feature에 SAE 부착
- FairFace(86K 얼굴), CelebA, Cocogender 등으로 학습
- Matryoshka variant, top-k(=20) sparsity, expansion factor 8
- **110,000 에폭** 학습 (별도 GPU 자원 필요)

### Stage 2: Social Neuron Probing
- SAE 활성화 패턴에서 특정 인구통계(성별/나이/인종)에만 반응하는 뉴런 식별
- effectiveness criterion: τ 비율 이상의 샘플에서 non-zero 활성화
- 그룹 특이적 뉴런 = Ng = Eg \ U¬g (해당 그룹에서만 활성화, 다른 그룹에선 비활성)
- **top-1 뉴런만 비활성화해도 전체 비활성화와 유사 효과**

### Stage 3: Social Neuron-Modulated Inference
- 추론 시 social neuron 활성화를 γ=0으로 설정 (비활성화)
- 원본 feature v와 SAE decoded feature v̂의 가중합: v' = (1-α)·v + α·v̂
- **α=0.6**: 일반 성능과 편향 제거의 균형점
- **α=1.0**: 최대 편향 제거 (일반 성능 일부 하락)

## 실험 결과

### CLIP (VLM) — T2I Retrieval
| 모델 | 방법 | Gender Max Skew | 변화 |
|------|------|----------------|------|
| ViT-B/16 | Baseline | 14.1 | — |
| ViT-B/16 | DEBIASLENS(I) α=0.6 | ~12.0 | -15% |
| ViT-B/16 | DEBIASLENS(I) α=1.0 | ~10.5 | -25% |

### InternVL2-8B (LVLM) — VQA
| 방법 | Gender Disproportion | 일반 성능(MME) |
|------|---------------------|---------------|
| Baseline | ~0.62 | 1440 |
| DEBIASLENS α=0.6 | ~0.50 (-19%) | 1454 (-0.97%) |
| Full Fine-Tuning | ~0.62 | 1440 |
| LoRA | ~0.57 | 1440 |
| Pruning (0.05) | ~0.50 | 1268 (-12%) |

→ DEBIASLENS가 **가장 낮은 트레이드오프** (편향 감소 대비 일반 성능 하락 최소)

### SBBench (BBQ 기반) — 우리 대회와 가장 관련
| 방법 | Train Data | Probing Data | Gender Acc | Age Acc |
|------|-----------|-------------|------------|---------|
| InternVL2-8B (baseline) | — | — | 83.83 (Rule) / 85.97 (Phi) | 43.11 / 50.35 |
| DEBIASLENS | FairFace | FairFace α=0.6 | 86.68 / 88.39 | 47.52 / 52.54 |
| DEBIASLENS | FairFace | FairFace α=1.0 | **87.87 / 89.49** | **48.51 / 53.77** |
| DEBIASLENS | SB-Syn-Crop | SB-Syn-Crop | 84.71 / 87.07 | 45.55 / 51.51 |

### Neuron Specificity (Table 2)
- Gender neuron 비활성화 → Gender bias만 감소, Age/Race bias 미변화
- Age neuron 비활성화 → Age bias만 감소
- **뉴런이 monosemantic** (단일 개념 인코딩) 확인

### Intersectional Fairness (Table 8)
| 타겟 속성 | Gender Skew↓ | Age Skew↓ | Race Skew↓ |
|----------|-------------|----------|-----------|
| Gender only | -8.0% | -5.6% | -1.3% |
| Age only | -8.1% | -18.0% | -3.1% |
| Gender×Age×Race | **-11.4%** | **-19.5%** | **-12.0%** |

→ **교차(intersectional) 타겟팅이 단일 속성보다 훨씬 효과적**

### Computational Cost (Table 7)
| 방법 | 파라미터(M) | GPU 시간 | 오버헤드(ms) | Trade-off↑ |
|------|-----------|---------|------------|-----------|
| Full FT | 6979.58 | 0.02 | 310 | 1.29 |
| LoRA | 301.99 | 0.32 | 311 | 1.30 |
| Pruning(0.05) | — | 0.00 | 355 | 1.53 |
| Prompt Engin. | — | 0.00 | 317 | 1.35 |
| **DEBIASLENS** | 1.57 | 0.01 | 323 | **가장 좋은 트레이드오프** |

### α Trade-off (Table 4)
| α | ImgNette↑ | FairFace↓ | MME↑ | VLA↓ |
|---|---------|---------|------|------|
| 0.0 | 99.5 | 18.8 | 1440 | 0.62 |
| 0.4 | 99.0 | 16.2 | 1496 | 0.53 |
| 0.6 | 97.5 | 14.2 | 1454 | 0.50 |
| 1.0 | 59.1 | 10.6 | 1152 | — |

→ α↑ = 편향↓ but 일반 성능↓. BBQ 스타일에서는 α=1.0이 더 좋음.

## 우리 프로젝트 적용 가능성

### 직접 적용 불가
- SAE 학습: 110K 에폭 (별도 GPU + 시간)
- Qwen3.5-9B 비전 인코더에 SAE 부착: vLLM 파이프라인 수정 필요
- 대회 규칙: 모델 가중치 변경 허용 여부 불확실
- 70분 제한 내 SAE 학습 불가능

### 간접 적용 가능 (인사이트 활용)
1. **이미지 인코더가 편향 주요 원인** → 우리 시각편향 경고 방향 맞음
2. **SB-Syn-Crop(얼굴 크롭)이 효과적** → B패밀리 이미지 얼굴 크롭 전처리
3. **교차 디바이어싱이 더 효과적** → 성별뿐 아니라 나이/체형/의상도 경고
4. **α=1.0이 BBQ에서 더 좋음** → 시각편향 경고를 더 공격적으로 가능
5. **top-1 뉴런 = 최소 개입 최대 효과** → 가장 핵심적인 편향 단서 하나를 정확히 타겟

## 참고
- GitHub: (비공개)
- 평가 데이터: FairFace, VLAGenderBias, SBBench
- 테스트 모델: CLIP(ViT-B/16, ViT-L/14@336), LLaVA-1.5-7B, InternVL2-8B
