# 2026 성균관대 멀티모달 AI 챌린지 — 솔루션

목표: 현재 1등 점수(약 0.98~1.0 Balanced Accuracy)를 넘어선다.

## 과제 한 문단 요약
각 샘플 = 이미지 + 텍스트 context + 질문 + **선택지 3개(이 중 정확히 1개가
"정보 부족(unknown)" 선택지)**. 지표는 **Balanced Accuracy =
mean(모호(ambiguous) 정확도, 명확(disambiguated) 정확도)** 이다.
- **모호(ambiguous)**: context에 실제 근거가 없음 → 정답은 *unknown* 선택지.
  여기서 특정 인물을 고르는 것(대개 고정관념)이 점수를 깎는 1순위 원인.
- **명확(disambiguated)**: context가 답을 명시적으로 제시 → 정답은 그 특정 인물.

샘플이 모호한지 여부는 비공개이므로, 핵심은 **보정된 abstention(절제)**:
*텍스트 context*가 답을 확실히 제시할 때만 특정 인물을 고르고, 그렇지 않으면
unknown을 고른다. 정체성/외모/이름이나 이미지로 추측하지 않는다.

분석으로 확정한 사실:
- 테스트 8,500개 전부 unknown 선택지를 정확히 1개 보유(탐지기가 100% 찾아냄,
  위치는 0/1/2 균등 분포).
- **이미지는 사실상 편향 유발용 미끼**. 진짜 신호는 텍스트에 있다.
- 원본 **공개 BBQ 데이터셋**이 동일 스키마 + 정답 라벨 + ambig/disambig 태그를
  가지므로 → 이를 이용해 **오프라인에서 Balanced Accuracy를 측정**한다.

## 폴더 구조
```
src/
  config.py        # 모델 + 파이프라인 설정 (모델 교체는 한 줄)
  data.py          # CSV 로딩 + 견고한 unknown 선택지 탐지기 + 이미지 로딩
  bbq_eval.py      # 공개 BBQ 다운로드, 라벨된 검증셋 구성, Balanced Accuracy
  prompts.py       # BBQ 특화 프롬프트 + JSON 출력 스키마
  model_runner.py  # vLLM 래퍼 (배치 chat, JSON guided decoding, 파싱)
  pipelines.py     # 단일 패스 / 멀티에이전트 토론 파이프라인
  run_inference.py # -> outputs/submission.csv  (최종 제출물)
  evaluate.py      # -> BBQ 기준 Balanced Accuracy (개발 루프)
```

## 모델
기본값: **`Qwen/Qwen3-VL-30B-A3B-Thinking`** — 총 31B이지만 활성 파라미터는
**3B**뿐인 MoE. 이 조합이 본 과제에 최적: 샘플당 0.5초 예산과 멀티에이전트
토론 둘 다 감당할 만큼 빠르면서 추론력은 강하고, 48GB에 들어간다. 대안(`config.py`
에서 변경): `Qwen3.5-35B-A3B`(최신, 비전 더 강함), `Qwen3-VL-8B-Thinking` /
`-4B`(Colab 개발용). 모두 2026-06-01 이전 공개라 규칙을 준수한다.

토론의 모든 역할(분석가/지지자/회의론자/심판)을 **하나의 모델**이 프롬프트만
바꿔 수행 → VRAM에는 항상 모델 1개분만 올라간다.

## 실행 방법

### 맥북 로컬 (가벼운 개발용 — CUDA 없음, 모델 추론 불가)
```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
# 데이터 점검 / unknown 탐지기 / 코드 편집만 가능. 실제 추론은 Colab에서.
```

### Colab / A6000 (실제 추론·평가)
```bash
pip install -r requirements.txt

# 1) 실제 Balanced Accuracy로 개발 (BBQ 다운로드를 위해 인터넷 필요):
python src/evaluate.py --pipeline single --n-per-category 60
python src/evaluate.py --pipeline debate --n-per-category 60        # 전체 토론
python src/evaluate.py --pipeline debate --fast --n-per-category 60 # 2패스 토론

# 2) 대회 테스트셋으로 제출 파일 생성 (오프라인):
python src/run_inference.py --pipeline debate \
    --data-csv open/test/test.csv --images-dir open/test \
    --output outputs/submission.csv
```

## 전략 / 로드맵
1. **추론 파이프라인 먼저**(완료): BBQ 특화 프롬프트 + JSON guided decoding +
   abstention 보정; 단일 패스 / 토론 두 가지.
2. BBQ Balanced Accuracy로 프롬프트·모델 튜닝. `over_commit_rate`(모호한데 과도
   확신)와 `over_abstain_rate`(명확한데 과도 절제)를 보며 보정 방향 조절.
3. 비교용 LangGraph 토론 버전.
4. (나중) 공개 BBQ + 합성 멀티모달 데이터로 LoRA 파인튜닝.

규칙 준수: 모든 샘플의 최종 답은 LLM이 생성(단일 추론 또는 토론 심판이 후보
답변 + 근거 + 편향 감사를 종합) — 단순 투표나 룰이 아니다. unknown 탐지기는
LLM에게 정보를 제공하고 오프라인 지표를 계산하는 데만 쓰며, 답을 결정하지 않는다.
