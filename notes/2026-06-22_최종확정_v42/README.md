# Phase 6: 최종 확정 (v42) ★
> 날짜: 2026-06-22

## 요약
v41 기반에 2가지 핵심 변경을 적용하여 **최종 제출 1순위** 확정.

### 변경 1: B 중재 시각편향 경고
- base SYSTEM_PROMPT에는 넣지 않음 (ambig 0.999 보호)
- LLM 중재 단계에서만, B패밀리 충돌 건에 "POSED stock images" 경고 삽입
- 144건 B충돌 중 52건 B→unk 전환

```python
if fam(r['ctx'])=='B':
    bias_warn=("CRITICAL: The photographs are POSED stock images with actors. "
               "Physical posture, gestures, facial expressions, seating position, "
               "and body language are NOT valid evidence...")
```

### 변경 2: Recovery 2단계 (binary choice)
- 1차("Abstaining is WRONG") 실패 건에 2차("ONLY two answers") 적용
- v41 22건 미회수 → v42 **1건**만 미회수 → Recovery **99.8%**

```python
rec2_text=("There are ONLY two possible answers. 'Unknown' is NOT acceptable.\n\n"
           "The answer is EITHER:\n  [A] opt_a\n  [B] opt_b\n\n"
           "The context states: \"key_sent\"...")
```

## 최종 결과
| 지표 | v42 | 비교 |
|------|-----|------|
| **Public** | 0.9991667 | v40(0.9993) 대비 -0.0002 |
| **synth_gold BA** | **0.8197** | 역대 최고 (v41 0.8124 대비 +0.0073) |
| **Recovery** | **99.8%** | 534/535건 회수 |
| 실행 시간 | ~35분 | 70분 제한 내 |

## v42 vs v40 diff 분석
- 총 188건 차이 (B:119, A:67, C:2)
- API(Gemini/GPT-4o) 합의 분석: v42가 API와 더 많이 합의 (75 vs 55건)

## 포함 파일
### notebooks/ — colab_v42.ipynb
### outputs/ — submission_v42.csv
### charts/ — waterfall, diff_analysis, b_category, sample_images (4장)
