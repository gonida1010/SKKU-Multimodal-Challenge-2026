# SKKU 멀티모달 챌린지 2026 — 마스터 인수인계 문서 (v17~v30 전체 통합)

작성: 2026-06-19. 이 문서 하나로 프로젝트 전체를 인수인계한다. 직전까지의 모든 세션(채점구조 역공학 → v27 제출 → v28/v29/v30 실패)을 통합했다.
**원칙: 사실과 수치만. 추측은 `[추정]` 표시. 모르는 건 모른다고 적는다.**

---

# PART 1. 대회 개요와 최종 목표

## 1-1. 대회
- **Dacon, 성균관대(SKKU) Multimodal AI Challenge 2026.**
- 과제: 이미지+텍스트 **BBQ 스타일 편향 3지선다.** 각 문항은 context(상황 문장) + question + 보기 3개(label 0/1/2). **보기 중 정확히 하나는 항상 "unknown" 류**("cannot be determined", "not enough information" 등).
- test 8,500개. 하루 제출 5회.
- 모델: **Qwen3.5-9B**, vLLM 서빙, 개발은 A100 Colab Pro.
- 핵심 구조: **permSC**(보기 순서 3순열 self-consistency) + arbiter(순서 흔들리면 종합).

## 1-2. 최종 목표 (절대 흔들지 말 것)
**비공개 Private 리더보드의 일반화 성능 1등.** Public 점수 최적화가 아니다.
- 사용자(박영곤) 확정 전략 = **"길 B": Public은 이미 충분히 올렸으니(0.99858), 점수에 안 보이는 Private 일반화에 베팅.**
- Public 19위지만 흔들리지 않기로 사용자가 명시 확정함. Public 천장 경쟁(C패밀리 텍스트 문제)은 Private 일반화(A패밀리 이미지 추론)와 다른 게임이다.

## 1-3. 대회 평가 환경 (FAQ로 확정)
- GPU: **RTX A6000 48GB** / Python 3.10 / CUDA 12.4 / PyTorch 2.6.0 / Ubuntu 20.04.
- **추론 시간 제한: Test 8,500개 ≈ 70분, Hidden 1,500개 ≈ 13분.**
- requirements.txt로 라이브러리 버전 조정 가능. CUDA/OS/Python/GPU는 변경 불가.
- **미해결 리스크: 현 파이프라인(permSC+witness+grounding+4단게이트)은 샘플당 vLLM 호출이 많다. Qwen3.5-9B가 A6000 70분 안에 드는지 미실측.** 개발은 A100 기준. 제출 전 반드시 A6000 시간 측정 필요.

## 1-4. 절대 제약 (위반 시 작업 무효)
1. **샘플 ID 하드코딩 금지.** 모든 수정은 일반화되는 규칙/코드여야 함.
2. **수동 CSV 편집 금지.** 손으로 라벨 고친 제출본(v22)은 최종 후보가 아니다.
3. **Public 전용 룰 금지. abstain(unknown 회피) 게이트 추가 금지.** commit precision만 올린다.
4. **이미지를 직접 보지 않고 판단 금지.** 멀티모달 대회다. 텍스트만 보고 판단했다가 분석이 두 번 뒤집힌 전례 있음.
5. **v27 원본 함수를 새로 짜지 말 것.** run_corevqa 등을 새로 작성하면 baseline 재현 실패 → 비교 무효(v30이 이걸로 망함).
6. **큰 파일 컨텍스트로 끌어오지 말 것**(2MB 진단 CSV, 8500행 base64, 870KB ipynb 전체). 셀만 추출하거나 Colab에서 집계. ← 이게 여러 세션 실패의 직접 원인.

---

# PART 2. 채점 구조 (프로브 실험으로 확정된 사실)

## 2-1. 지표
- **Balanced Accuracy = (acc_ambig + acc_disambig) / 2.** 1 unit = 1/12000.
- ambig = context가 모호해 정답이 "unknown"인 문항. disambig = 정답이 특정 보기인 문항.

## 2-2. Public/Private 분할 (프로브 실험 결과)
test 8,500 = **Public 5,000 + Private 3,500.** Private는 점수 안 보이고 대회 종료 후 공개.
- **Public 5,000** = C패밀리(~2,000, disambig, 오답당 -3units) + 무귀속 B(~2,900, ambig, 오답당 -2units)
- **Private 3,500** = A패밀리(~1,750) + 잔여 B(~1,750)

**프로브 실험**: A/B1/B2/C 패밀리에서 각 30개를 고의 오답으로 제출 →
- **C만 -90units(=30×3) 하락 → C는 Public disambig.**
- **A·B1·B2는 무변화 → Private 구역(현재 채점 안 됨).**
- 결론: **A패밀리 = Private의 핵심 표적.** 이미지를 봐야 푸는 샘플이 통째로 Private에 있음.

## 2-3. 패밀리 분류 정규식 (v27 노트북 셀11 원본) + 발견된 버그
```python
RE_A = re.compile(r"^\s*The image shows multiple people", re.I)   # ← 버그: A의 49%만 매칭
RE_IMGREF = re.compile(r"\b(image|photo|picture)\b", re.I)
fam = 'A' if RE_A.search(c) else ('B' if RE_IMGREF.search(c) else 'C')
```
**중대 발견**: 위 RE_A는 "multiple people"만 잡아 857개만 매칭. 실제 A패밀리는:
- `^The image shows multiple`: 857
- `^The image shows several`: 860 ← 놓침
- `^The image shows a`(classroom 등): 33 ← 놓침
- **정답 A패밀리 = `^\s*The image shows` = 1,750개** (프로브 기대값과 일치)
- v27 셀11이 A의 절반(893개)을 B로 오분류했음. 단 v27 제출 라벨에 영향 줬는지는 미확인. 올바른 1750 기준 v27 A합성BA 재계산 = 0.7445 (857 기준 0.747과 거의 동일).

## 2-4. unk_idx 판정 (셀1 원본)
```python
UNK = [31개 문구: "cannot be determined", "not enough information", "unknown", ...]
def find_unknown(answers):
    f = [any(p in a.lower() for p in UNK) for a in answers]
    if sum(f) == 1: return f.index(True)
    idx = [i for i, x in enumerate(f) if x]
    return min(idx, key=lambda i: len(answers[i])) if idx else None
```

---

# PART 3. A합성정답(synth_gold) — Private proxy 지표와 그 한계

A패밀리는 점수가 안 보이므로(Private), Private 성능을 버전 비교하려고 만든 **근사 정답 채점기**.

## 3-1. 정의 (셀11 원본)
```python
ATTR = re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b(.{0,120})")
# context에서 "An X person/man/woman ..." 패턴으로 행동 귀속된 집단 추출.
# hits = 보기 중 그 집단명 포함 인덱스. len(hits)==1 이어야 disambig 판정.
# 방향(어느 집단이 답인지): EV_NEG / Q_NEG / TRAIT_NEG 부호 로직.
```

## 3-2. 두 가지 한계 (둘 다 측정으로 확인됨)
1. **이미지 불일치를 못 봄**: synth_gold는 텍스트 귀속만 채점. "Latino in the center"가 정답이라 쳐도, 실제 사진엔 Latino가 없을 수 있음(BBQ가 CrowdHuman 군중사진을 무작위로 붙임). 따라서 **synth_gold가 높다고 좋은 게 아님 — 환각 commit도 정답으로 쳐줌.**
2. **방향 로직이 부정확**: v27 commit vs synth_gold를 disambig에서 비교하니 일치율 49.2%(nD=1041). context가 명시 귀속한 집단을 synth_gold가 반대로 판정하는 사례 다수. **즉 synth_gold는 정답지로 신뢰 불가.** 버전 간 "상대 비교"로만 참고.

## 3-3. A합성BA 추이 (버전 비교용, 절대값 신뢰 금지)
| 버전 | A합성BA | A disambig_acc |
|---|---|---|
| v18 | 0.667 | 0.334 |
| v24 | 0.721 | 0.441 |
| v25 | 0.771 | 0.543 |
| v27 | 0.747 | 0.494 |

**v27이 v25보다 낮은 건 후퇴 아님**: v27 grounding이 "이미지에서 식별 안 되는" 환각 commit 후보 79개를 기각했기 때문. synth_gold는 그걸 못 보고 점수를 깎음.

---

# PART 4. 버전별 전체 히스토리

## 4-1. v17~v19 (실패) — 방향 전환의 계기
- v17 contradiction gate(no/none/only 문장 재검), v19 context gate. 둘 다 **abstain을 늘리는 방향** → Public이 v18(0.99733) 밑으로. 
- **이 실패가 "점수가 1/12000 격자로만 움직인다"는 관찰 → 채점구조 역공학으로 전환을 낳음.** (PART 2가 그 산물.)
- 교훈: 이 대회에서 **abstain을 늘리면 손해.** commit precision만 올려야 함.

## 4-2. v20.1 (실패) — recovery 첫 시도
- unknown을 근거 있으면 commit하는 방향 전환. flip 92개 중 오답 섞임 → Public 0.9968.
- 원인: "근거 있음"을 느슨하게 정의. 행동 아닌 외모/소지품 묘사("태블릿 든 사람")를 근거로 commit. 순환논증 통과.

## 4-3. v23~v25 (성공) — 게이트 누적
| 버전 | Public | A합성BA | 핵심 추가 |
|---|---|---|---|
| v18 | 0.99733 | 0.667 | 이전 베이스라인 |
| v23 | 0.99742 | 0.700 | E2E 단일노트북, 듀얼루트(텍스트+이미지 witness) 재심, 행동증거 검증기(ACTION vs WEAK) |
| v24 | 0.99800 | 0.721 | 규칙8(텍스트 명시사실은 이미지가 못 뒤집음, C퇴행 방지) + 스테레오타입 누수 차단 게이트 |
| v25 | 0.99825 | 0.771 | 규칙9(특질 명시배정) + 2:1 다수결 + 확인패스(사실상 3:1 요구) |
- v25 검증: **사용자가 손으로 찾은 회수후보 32개를 v25 파이프라인이 하드코딩 없이 32/32 자력 재현.** (일반화 근거.)
- v22 = 손수정본(Public 0.99858). **원칙 위반이라 최종후보 제외.** 블로그에서도 의도적 배제.

## 4-4. v27 (현재 최선, 제출완료, Public 19위)
- 추가: 규칙10(집단 정체성은 이미지로 판별 불가) + **5번째 게이트 descriptor grounding.**
- `descriptor_grounding()` + `DESCRIPTOR_SYS`: commit 후보 보기("The Arab person in the black shirt")를 이미지에 대고 "정확히 한 사람만 매칭되나?" 물음. 모호하면 NO. "When in doubt, reply NO."
- **v27 풀런 결과**: flip 190개. grounding이 시각수식어 후보 231개 중 **79개 기각**(환각 차단). recovery 사유 분포: still_unknown 4140, FLIP 190, not_unanimous 122, descriptor_ungrounded 79, weak_evidence 70, evidence_fail 56, tiebreak_unconfirmed 32.
- BBQ 검증: BA 0.9791, ambig 오염 0, C퇴행 0.
- **Public 0.9985833333** (= 손수정 v22 동점, 코드로 달성한 최고. 리더보드 19위.)
- **미측정 항목**: v27의 SB over_commit(base 재사용 세션에서 guardrail 미호출로 누락). v25는 0.33%였음.

### v27 recovery 4단 게이트 (셀8)
1. permSC 합의(3순열 만장일치, 또는 2:1인데 이견이 unknown일 때만)
2. 행동증거 검증(인용 증거가 전부 ACTION이어야. WEAK면 기각)
3. 스테레오타입 누수 차단(_ev_in_ctx로 증거가 context에 있나 확인. 이미지 단독 증거면 텍스트 재질문, 그래도 commit하면 STEREOTYPE_LEAK 기각)
4. 확인 패스(2:1 후보는 이미지+witness 재질문 재확인)
+ 5단 descriptor grounding(위)

## 4-5. v28 (NO-GO) — "텍스트로 풀리는 A 회수상한" 측정
- 가설: v27이 unknown 남긴 A 중 텍스트만으로 안전 회수 가능한 수.
- 측정: 올바른 A=1750 기준 v27 미회수 A = 952개. 그중 synth_gold가 disambig로 본 것 = 251개. v27 게이트 분해(A∩귀속∩unknown∩집단명 보기 1매칭)=254개(descriptor_ungrounded 78, not_unanimous 76, still_unknown 67, tiebreak_unconfirmed 31, weak 2).
- **이미지 vLLM 카운트 + 육안 검증(15장)**: descriptor_ungrounded 78개 중 "집단=1명" 합의는 7개뿐. 그 7개 중 6개는 보기의 위치/옷 묘사가 실제 사람과 불일치(**v27이 옳게 막은 것**). 1개(TEST_2928 Arab "blue shirt")만 회색지대.
- **결론: 텍스트 회수 광맥은 거의 없음. v27 grounding이 막은 게 대부분 정당.** 추가 회수 여지 미미.

## 4-6. v29/v29b (NO-GO) — "정밀도 누수 차단" commit→unknown
- 가설: v27이 commit한 A 중 귀속 집단이 이미지에 2명 이상이면 모호 → unknown 복귀(ambig 정확도 상승 의도).
- v29(무조건 집단≥2 flip): 표적 796개, flip 434개. **문제: 묘사 없는 집단지칭 보기("The White person")까지 flip → context 귀속이 정답인데 망침. 15개 중 6개(~40%) 오류.**
- v29b(개인묘사 보기만 flip): DESC 정규식으로 위치/옷/안경 묘사 있는 보기만. 표적 439개, flip 271개. **제출함. Public 0.9985833333 (v27과 완전 동일 — 271개 전부 A패밀리라 Public 무영향, 8229행 v27 동일 확인).**
- **문제: flip 대상 대부분이 disambiguated context였음**(예: "A Caucasian person ignored..."에 명시 귀속). [추정] BBQ A패밀리는 context 귀속이 답을 정하므로 집단이 여러 명이어도 답은 그 집단 → flip이 정답을 unknown으로 망쳤을 가능성. Private 미공개라 확증 불가. 추가 버그: TEST_0486은 ATTR 집단(Black)≠commit 집단(White)인데 Black 카운트로 White를 flip(처리 안 된 케이스).

## 4-7. v30 (NO-GO) — COREVQA 분해 프롬프트 + 대회 이식
- COREVQA 768 로그(v27 기실행분) 오류 해부: 이미지패스 acc 0.7125, 텍스트 0.5775. False(1)를 True로 트는 편향(False 오답 74/284). 복합 부정/연언문("not a single","neither","only one")에서 취약.
- A/B/C 프롬프트 비교(400개): **측정 무효.** Claude가 A(기존)를 v27 실제 함수 run_corevqa로 안 부르고 SYS_A를 새로 작성 → baseline 재현 실패(A acc 35.5%, 알려진 0.7125와 불일치). B분해 56%도 v27 0.71보다 낮음.
- 대회 이식: v27 SYSTEM_PROMPT에 분해지침 추가(SYS_V30, "prefer unknown" 포함), base 추론 교체(게이트 미적용). **제출함. Public 0.9980833333 (하락).** [추정] unknown 선호가 과도한 abstain 유발.

---

# PART 5. 제출 기록 (Dacon Public)

| 제출 파일 | 제출일 | Public | 비고 |
|---|---|---|---|
| submission_v27_descriptor_grounding.csv | 06-16 | 0.9985833333 | **현재 최종후보, 19위** |
| submission_v29b_precision_gate.csv | 06-18 | 0.9985833333 | v27과 동일(A패밀리만 변경, Public 무영향) |
| submission_v30_decomp_base.csv | 06-18 | 0.9980833333 | 하락 |
- 상위권: Public 1.0 만점 다수(대회 토론서 큰 모델 사용 확인). 우리 v27 = 0.99858.

---

# PART 6. 반복된 실패 패턴 (다음 AI는 피할 것)

1. **단순 규칙(집단 카운트)으로 v27을 변경 → 개선 없음/하락**(v28,v29b). A패밀리는 context 귀속이 답을 정하는데, 이미지 인원수로 뒤집으려 함.
2. **v27 원본 함수 대신 새 프롬프트 작성 → baseline 재현 실패 → 비교 무효**(v30). 사용자가 "원본 가져오라" 반복 지시했으나 위반.
3. **이미지 직접 검증 생략하고 제출**(v29b, v30). v28만 육안 검증했고, 그게 유일하게 신뢰할 결론을 냄.
4. **큰 파일을 컨텍스트로 끌어와 토큰 소진 후 끊김**(여러 세션). download_file_content로 8500행 base64 받기 = 검증된 실패 경로.
5. **Colab 런타임 끊김**을 처음부터 재실행으로 대응 → 시간 낭비. base/witness 캐시(`base_preds_v27_*.csv`, `witness_v27_*.csv`)부터 확인할 것.

---

# PART 7. 우리가 연구로 확정한 핵심 지식 (가장 중요)

이 프로젝트의 진짜 자산. 다음 시도는 반드시 이 위에서 출발할 것.

1. **A패밀리(이미지 필수)=Private, C패밀리(텍스트 BBQ)=Public.** 프로브 실험으로 측정 확정. Public 점수는 A패밀리를 채점 안 하므로, A 개선은 Public에 안 보인다. 19위에 흔들릴 필요 없다.

2. **BBQ A패밀리의 정답은 context의 행동 귀속이 정한다.** "An Arab person did X"면 답은 이미지 인원수와 무관하게 Arab. 이미지는 그 집단이 화면에 있는지 확인하는 보조. → **이미지 인원수로 commit을 뒤집는 접근(v29)은 틀렸다.**

3. **BBQ는 CrowdHuman 군중사진을 무작위로 붙였다.** 보기의 시각 수식어("노란 셔츠", "왼쪽")가 실제 사진과 불일치하는 경우가 절반 이상. → v27 grounding(이미지에서 식별될 때만 commit)이 이 환각을 막는 게 정당. (육안 15장 검증으로 확인.)

4. **이 대회는 천장(1.0) 근처 정밀도 싸움.** flip 많이가 아니라 flip 대부분이 맞아야 이득. abstain 늘리기(v17/v19)·느슨한 근거(v20.1)·프롬프트 unknown 선호(v30) 모두 손해.

5. **synth_gold(A합성BA)는 절대값 신뢰 불가, 버전 비교용.** 이미지 불일치를 못 보고 방향 로직도 부정확(일치율 49%). 숫자만 좇으면 환각을 정답으로 착각한다.

6. **COREVQA = 순수 일반화 측정 프록시**(대회 제출과 분리). 대회측이 "Public과 비슷한 분포 + 멀티모달 일반화 중요"라 안내. 일반화 올리면 Public도 근소 상승하는 경향 관찰됨. 약점: 복합 부정/연언문에서 False를 True로 트는 편향.

7. **남은 개선 여지는 "A패밀리에서 v27이 놓쳤지만 이미지로 진짜 식별 가능한" 케이스**인데, v28 측정 결과 그 수가 매우 적음(78개 중 7개 후보, 그나마 6개는 v27이 옳게 막음). → 단순 규칙으로는 한계. 다음은 [추정] (a) 더 강한 VLM으로 이미지 식별력 자체를 올리거나, (b) grounding 게이트의 정밀도를 높이는 방향이 남음.

---

# PART 8. 자산 위치 (Google Drive: SKKU-Multimodal-Challenge-2026/)

## outputs/ (parentId 1x2_TB0HU-MvzsouWMWcVRFW82BzCnm9V)
- `submission_v27_descriptor_grounding.csv` (id 1cPNWm2iE3ks8esrwsNUeIMY9U_oqZSdC, 8500행) ← **현재 최선**
- `submission_v29b_precision_gate.csv` + `v29b_precision_gate_flips.csv` (271행 변경로그)
- `submission_v30_decomp_base.csv`
- `research_corevqa_ABC.csv` (id 1xA_NU-4RhpmhEWrUn-lnjOUuq1C0eQUw, 400행)
- `v28_count_probe.csv` (78행) + `v28_review_imgs/` (검증이미지 15 + _meta.csv)
- `v27_descriptor_grounding_diagnostics.csv` (id 1kelHe3SN1AfMoz3QX8oFvu6KCnKZaP94, 2MB, reason 컬럼. **직접 download 금지**)
- `g_suite_summary.csv` (id 1QUlDAw5F6AJzdh7BJDzLJOAsC7mzsgKo. **버그: COREVQA/VG/SB 줄 누락** — base 재사용 세션서 _GUARD_CACHE 미기록)
- `base_preds_v27_descriptor_grounding.csv`, `witness_v27_descriptor_grounding.csv` (재사용 캐시. 끊김 복구용)
- `corevqa_logs/corevqa_format_768.csv` (400행. 컬럼: image_id, image_path, statement, gold_label, pred_img, pred_txt, raw_output_img/txt, reasoning_img/txt, correct_img, correct_txt, auto_tags, image_size, resize_long_side, experiment_name)

## 노트북 (Drive 폴더 parentId 1DrrPolA5AIV0_HeqbPDaKZZ5JoeCebQz)
- `colab_v27_resumable.ipynb` (id 10f_LuZcAAVmk0I2ZiMDbVoxAY_NVTOky, 13셀, 870KB. **read_file_content 금지 — 셀만 추출**)
  - 셀 구조: 0 설치 / 1 엔진+load_img+find_unknown / 2 헬퍼(SYSTEM_PROMPT, parse_answer, build_user_text, _sp) / 3~6 COREVQA(run_corevqa, SAMPLES, generate_with, load_image, to_url_jpeg, _reasoning, tag_statement) / 7 base추론 / 8 recovery 5단게이트 / 9 제출+진단저장 / 10 BBQ검증 / 11 종합+패밀리분류(RE_A/ATTR/synth_gold)
- `colab_v28_count_probe.ipynb` (IMG_ROOT 누락버그 → 측정셀 내 IMG_ROOT=TEST_DIR 명시 수정. load_img는 경로 틀려도 None 반환=조용히 실패하니 가드 필수)
- `colab_v29_precision_gate.ipynb`, `colab_v29b_precision_gate.ipynb`
- `colab_research_corevqa.ipynb` (8셀. PROJECT 변수 재시작 후 소실 → 쓰는 셀마다 자체 정의 가드 추가로 수정)

## 측정에 m24.pkl 불필요
m24.pkl은 Colab 로컬 자산이라 끊기면 소실됨(Drive에 없음). A패밀리 판정은 **모델 산물이 아니라 정규식**(PART 2-3)이므로, **test 원문 + submission CSV + 정규식**만으로 모든 측정 재현 가능. m24 복구에 매달리지 말 것.

---

# PART 9. 블로그 (참고)
- 티스토리 pak1010pak.tistory.com, 6편까지. 6편(채점구조 역공학~v25)은 작성 완료(`blog6.md` + 이미지 3장). 1인칭 연구일지 톤. v22 손수정본은 원칙상 본문 제외.

---

# PART 10. 작업 방식 (사용자 요구, 엄수)
- 깊게 추론. 추측과 측정 구분 표시. 모르는 내용 아는 척 금지 — 필요 데이터는 사용자에게 요청.
- **이미지를 직접 보고 판단.** 텍스트만으로 결론 금지.
- **v27 원본 함수 그대로 사용.** 새로 짜면 비교 무효.
- 큰 파일 컨텍스트 반입 금지. 셀 추출/Colab 집계로.
- 끊기면 캐시·저장본부터 확인. 처음부터 재실행 금지.
- Colab 비용 신경쓰지 말 것(사용자 명시). 제대로 된 검증 우선.
- **분석만 반복하지 말고 제출 가능한 산출물을 낼 것.**
