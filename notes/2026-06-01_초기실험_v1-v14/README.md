# Phase 0-1: 초기 실험 (v1~v14)
> 기간: 2026-06-01 ~ 06-05

## 요약
- Qwen3.5-9B 기반 VQA 파이프라인 초기 설계
- 다양한 모델 테스트: Qwen3-VL-8B, Gemma4-E4B
- 추론 전략 실험: few-shot, elimination, PermSC, image-gating, commit-arbiter
- COREVQA 일반화 벤치마크 설계 및 측정
- 강건성 데이터셋(SBBench, VisualHard) 평가 환경 구축

## 핵심 발견
- Qwen3.5-9B가 최적 모델로 확정 (타 모델 대비 안정적)
- 768px 이미지 해상도가 속도/정확도 균형점
- thinking 모드 OFF가 더 안정적 (v18에서 확정)
- PermSC(선택지 순서 셔플 3회) 도입 → 순서 편향 제거

## 포함 파일
### notebooks/ (20개)
- `colab_v1_legacy.ipynb` ~ `colab_v14_commit-arbiter.ipynb`: 버전별 실험 노트북
- `colab_research_corevqa.ipynb`: COREVQA 일반화 벤치마크
- `colab_robustness.ipynb`, `colab_sbbench_eval.ipynb` 등: 강건성 평가

### outputs/ (11개)
- 초기 제출 CSV: 0.9756~0.9963 점수대
- `diag_invariance.csv`: 불변성 진단 결과

### analysis_charts/
- 버전 진화, 파이프라인 워터폴, 커밋 히트맵 등 6장
