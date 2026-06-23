# Vision Language Models are Biased

> Vo et al., NeurIPS 2025 Datasets and Benchmarks Track
> OpenReview: https://openreview.net/forum?id=4GWfYyo6FS

## 핵심 아이디어
VLM이 **시각 입력을 무시하고 사전 지식(prior knowledge)에 의존**하여 틀린 답을 내는 문제를 체계적으로 측정.

## 실험 결과
- **7개 도메인에서 counting accuracy 평균 17.05%** (매우 낮음)
- 테스트 도메인: 동물, 브랜드, 체스, 보드게임, 광학 착각, 패턴 그리드
- **"double-check" 프롬프트**: 평균 **+6점** 밖에 안 됨
- **"이미지에만 집중하라" 프롬프트**: 역시 미미한 효과

## 의미
- VLM은 이미지를 "보는 것"보다 "아는 것"에 의존
- 단순 프롬프트("다시 확인하라", "이미지에 집중하라")는 거의 효과 없음
- **구조화된 프롬프트**(우리의 Rule 체계, Recovery 2단계)가 필요한 이유

## 우리 프로젝트와의 관련성
- **직접 관련**: 낮음 (이 논문은 social bias가 아니라 counting/identification bias)
- **간접 확인**: 
  - 단순 "double-check" = +6점 vs 우리 Recovery 1단계("Abstaining is WRONG") = +8%p → 구조화된 프롬프트가 핵심
  - VLM의 사전 지식 의존 = 우리 연구에서 확인한 "이미지는 고정관념 억제제" 역할과 일치

## 참고
- GitHub: https://github.com/anvo25/vlms-are-biased
- Dataset: https://huggingface.co/datasets/anvo25/vlms-are-biased
