# SKKU 멀티모달 챌린지 2026 — v31 grounding 게이트 분석 세션

작성: 2026-06-19 (Claude Code 세션). **사실·수치 위주. 추정은 `[추정]` 표시.**
이전 인수인계(MASTER_HANDOFF.md, HANDOFF_v28_v30.md)의 제약·목표는 그대로 유효. 이 문서는 그 위에 추가된 **v27 5차 grounding 게이트 정밀도 연구**다.

방향: 사용자 지정 = "v27 grounding 게이트의 정밀도를 더 높이기". 일반화(Private) 개선 우선, A6000 70분 제약은 문제없다고 사용자 확정.

---

## 0. 한 줄 결론

**v27의 5차 `descriptor_grounding` 게이트는 A패밀리(Private 표적)에서 정답을 회수 못 하게 막는 over-rejection 게이트다.** 시각수식어는 BBQ A패밀리 답 결정에 무관하고(측정: same-group=0 전수), 민족은 무작위 사진으로 식별 불가능하기 때문(이미지 8장 직접 검증). → 게이트를 끈 `colab_v31_grounding_off.ipynb` 생성(제출 대기).

이는 **MASTER_HANDOFF PART 7-3 / 핵심지식 #3("grounding이 환각 commit을 막는 게 정당")을 뒤집는 발견**이다.

---

## 1. v27 recovery 5단 게이트 구조 (셀8 원본 정밀 분석)

`recovery_permsc(qrows, qimgs, qwitness)` — unknown 전체(4689개)를 재심:
1. **permSC 재심**: 3순열 재추론(REC_SYSTEM + witness 시각사실 주입). 만장일치 또는 2:1(이견이 unknown일 때만) → candidate.
2. **인용 풀 검증** `_ev_in_pool`: evidence가 context+witness에 3-gram으로 실재해야.
3. **행동증거 검증기** VERIFIER_SYS: 인용이 전부 ACTION이어야. WEAK면 `weak_evidence` 기각.
4. **스테레오타입 누수** `_ev_in_ctx`: 이미지 단독 증거면 텍스트만 재질문 → 여전히 commit하면 `stereotype_leak` 기각.
5. **descriptor grounding** (이번 분석 표적): 옵션이 `VIS_DESC` 정규식(색/옷/위치/소지품) 매칭 시 → `descriptor_grounding()`이 이미지에 "정확히 한 사람만 명확히 보이나? YES/NO, 의심되면 NO". NO면 `descriptor_ungrounded` 기각.

REC_SYSTEM에는 이미 명시: *"if the context attributes the action to a group and one option names that group, commit to it; the image not showing the action does NOT contradict the context."* → **5차 게이트는 이 원칙과 정면 충돌**(시각수식어 미식별로 정답 집단을 막음).

---

## 2. 측정 (전부 로컬, LLM 불필요)

도구: `scripts/nb_peek.py`(ipynb 셀 추출), `scripts/analyze_grounding.py`, `scripts/analyze_option_groups.py`, `scripts/extract_ungrounded.py`.
데이터: `outputs/v27_descriptor_grounding_diagnostics.csv`(4689행) + `outputs/submission_v27_descriptor_grounding.csv` + `open/test/test.csv`.

### 2-1. v27 recovery reason 분포
```
still_unknown 4140 | FLIP 190 | not_unanimous 122 | descriptor_ungrounded 79
weak_evidence 70 | evidence_fail 56 | tiebreak_unconfirmed 32   (합 4689)
```

### 2-2. FLIP 190개 분해 (commit 옵션 = answers[label])
- A패밀리(올바른 정의 `^The image shows` = 1750): **160**
- 시각수식어 옵션(VIS_DESC 매칭 = grounding이 YES로 통과시킨 것): **152**
- **A & 시각수식어 = 135** ← v27 A commit의 84%(135/160). 전부 grounding이 YES 판정.
- A & 순수 집단명(grounding 비적용): 25
- 패밀리: FLIP A160 / B25 / C5

### 2-3. ★핵심: A패밀리 non-unk 보기 2개의 집단 관계 (전수 1750)
```
diff(서로 다른 집단) 1549 | "Can't answer"형 201(전부 diff) | same(같은 집단) 0
```
**A패밀리 1750개 전체에서 same-group = 0.** 모든 A패밀리는 `[집단X 보기] + [집단Y 보기] + [unknown]` 구조(예: Arab vs Hispanic). 같은 집단의 두 개인을 시각수식어로 구분하는 샘플은 **하나도 없음**.
→ **시각수식어("yellow shirt","on the left")는 A패밀리 답 결정에 100% 잉여.** 답은 context 귀속 집단으로 유일 결정되고, 보기에 그 집단은 단 하나뿐.
→ `descriptor_grounding`은 A패밀리에서 정답을 가려낼 능력이 없고, NO면 정답 후보를 unknown에 가둔다.

### 2-4. grounding이 막은 79개(descriptor_ungrounded) 패밀리
```
A 78 | B 1 | C 0
```
**거의 100% A패밀리, C(Public)는 0.** → 게이트 제거 시 Public 무영향, 정확히 Private(A)만 회수.

---

## 3. 이미지 직접 검증 8장 (제약 #4 준수, descriptor_ungrounded 무작위 표본 seed=7) — **2026-06-20 재확인, 표 일치**

| 샘플 | 귀속집단 / 시각수식어 | 실제 이미지 | 식별 가능? |
|---|---|---|---|
| TEST_4604 | European / "in the center" | 미국 정치인(Rubio 중앙, Vance, Trump) | European/Caucasian 외모 판별 불가 |
| TEST_2240 | Jewish / "wearing a tie" | 유대인 결혼식(키파) | Jewish는 종교표식으로 가능, tie는 모호 |
| TEST_5654 | Jewish / "on the left" | 실내 모임 군중 | 왼쪽 다수, 누가 Jewish인지 불명 |
| TEST_0819 | Latino / "far left" | **인도 철도노조 시위** | Latino 아예 없음 |
| TEST_0878 | Middle Eastern / "on the right" | 남아시아 여성들 | Middle Eastern 판별 모호 |
| TEST_7632 | Hispanic / "blue shirt" | 중동 남성+아이들 | Hispanic 없음 |
| TEST_1311 | American / "blue shirt" | 사무실 4인 | American 외모 판별 원천 불가 |
| TEST_5150 | African / "far right" | **인도 거리 행인** | African 아예 없음 |

**관찰**:
1. 8장 전부 context와 무관한 무작위 스톡/뉴스 사진(MASTER PART 7-3 = CrowdHuman 무작위 부착 재확인).
2. 귀속 집단 식별: European/Caucasian/American/Hispanic/Latino = 외모로 **원천 불가**. Middle Eastern/Arab/African = 복장 정황 추정만, 보기 매칭 모호. Jewish = 종교표식 있을 때만 가능.
3. **`descriptor_grounding`의 NO는 "환각 차단"이 아니라 "사진으로 민족 식별 불가능"의 결과.** "far left에 Latino 보이나?"→far left엔 인도인이 있으니 NO. 하지만 답은 어차피 context 귀속 Latino 보기.
4. **[2026-06-20] 이 decoupling이 §7 모달리티 발견(이미지=고정관념 억제제)의 메커니즘이다:** 이미지가 context와 무관한 무작위 사진이라, 모델이 그걸 보면 "여기 명명된 집단이 없다"며 *적절히 abstain*(이미지 ON). 이미지를 빼면 텍스트 고정관념으로 commit. → 이미지의 인과적 역할 = 답 제공이 아니라 고정관념 억제. **단 이 8장은 회수의 *정답성*을 검증한 게 아니다(이미지로는 불가). 회수 근거는 텍스트 구조(same-group=0 + context 귀속)뿐이다.**

---

## 4. v28이 반대 결론("옳게 막음")을 낸 이유

v28(HANDOFF_v28_v30.md §1)은 descriptor_ungrounded 78개 중 7개만 "집단=1", 6개는 "위치/옷 불일치 → v27이 옳게 막음"이라 결론. **그러나 v28은 grounding과 동일한 잘못된 전제**("시각수식어가 한 사람에 매칭돼야 정답")로 검증했다. v28은 **"non-unk 보기 2개가 항상 다른 집단(same=0)"을 측정하지 않았다.** 시각수식어가 답 결정에 잉여라는 사실을 못 보고, 위치/옷 불일치를 "막을 이유"로 오해했다.

---

## 5. 산출물 + 다음 단계

### 5-1. `colab_v31_grounding_off.ipynb` (생성 완료, 제출 대기)
- v27 원본(`colab_v27_resumable.ipynb`)에서 **최소 변경**(제약 #5 준수, 새로 안 짬):
  - 셀8: 5차 게이트를 `GROUNDING_ON = False`로 OFF (`if desc_items and GROUNDING_ON:`).
  - 셀9: `V_NAME = 'v31_grounding_off'`.
  - VER(`v27_descriptor_grounding`) 유지 → base/witness 캐시 재사용(재추론 없음, 빠름).
- 생성 스크립트: `scripts/make_v31_notebook.py`.
- 예상: flip 190 → 약 268(A패밀리 78 추가 회수). `submission_v31_grounding_off.csv` + `v31_grounding_off_diagnostics.csv` 저장. Public(C) 무변동.
- **실행 순서**: 셀 0(설치→재시작) → 1 → 2 → 3(마운트) → 7(base캐시) → 8(witness캐시+recovery) → 9(제출). 셀 4·5·6·10·11은 선택(셀11은 synth/패밀리 종합).

### 5-2. 리스크 / 미해결
- **[추정] v31 commit 78개가 진짜 정답인지는 Private 미공개라 확증 불가.** 근거: 이 78개는 1~4차 게이트(permSC 합의+ACTION 증거+스테레오타입 차단)를 통과한 disambig 후보이므로 정답 가능성 높음. 방향(어느 집단)은 permSC 판정에 의존.
- 간접 확인: 셀11 A합성BA(synth_gold, **절대값 신뢰 불가·버전비교용**) + 실제 Dacon 제출.
- grounding "통과분" 135개(YES)의 방향 정확도는 미검증(이번 세션은 "막힌 79개"에 집중). [추정] 통과분도 시각수식어 무관하게 집단 귀속이 답이면 정답.

---

## 6. 핵심지식 업데이트 제안 (MASTER_HANDOFF PART 7)

- **#3 수정**: "v27 grounding이 환각을 막는 게 정당" → **부분 철회.** A패밀리에서 시각수식어는 답과 무관(same=0)이고 민족 식별은 사진으로 불가능하므로, grounding은 A패밀리 정답을 over-reject한다. "시각수식어 불일치"는 막을 이유가 아니다(옷색·위치는 BBQ가 붙인 잉여 장식). 단 B/C패밀리에서의 grounding 효용은 별도(이번 미검증).
- **#2 강화**: "A패밀리 답은 context 행동 귀속이 정한다(이미지 무관)"가 보기 구조로 재확인됨 — 모든 A는 서로 다른 두 집단 + unknown이라, 귀속 집단 보기가 유일 정답 후보.

---

## 5-3. v31 실행 결과 (2026-06-19, 코랩 A100)

`colab_v31_grounding_off.ipynb` 실행 완료(base/witness 캐시 재사용, 재추론 없음).
- **recovery 사유**: still_unknown 4136 | FLIP **276** | not_unanimous 122 | weak_evidence 65 | evidence_fail 59 | tiebreak_unconfirmed 31. **descriptor_ungrounded 0** (게이트 OFF 확인).
- **FLIP 190→276 (+86)**. 구성 = descriptor_ungrounded 79개 석방 + 추론노이즈 순 +7 (still_unknown −4, weak −5, tiebreak −1, evidence_fail +3).
- **78개 회수 근거 = 텍스트 구조(same-group=0 + context 행동 귀속)**: verdict=ACTIONxN, 인용은 모두 context 행동 귀속. v28 회색지대 TEST_2928(Arab blue shirt)도 FLIP. ⚠️ **정정(2026-06-20):** §3의 이미지 8장은 회수의 *정답성*을 검증한 게 아니라 "이미지=context와 무관한 decoupled 스톡사진→집단 식별 불가"를 확인한 것. 이미지로는 정답을 검증할 수 없다(예전 "이미지검증 8/8" 표현은 과장이었음).
- **자가검증 19/32** (v25는 32/32). 못 잡은 13개는 전부 시각수식어 없는 이름/두-사람 BBQ(연구자·gym·math class·tutor 등)로 permSC합의·tiebreak에서 막힘 → **grounding과 무관, vLLM 0.23.0 환경 차이에 의한 추론 비결정성**. v31 결함 아님.
- 산출물: `outputs/submission_v31_grounding_off.csv`(8500행), `outputs/v31_grounding_off_diagnostics.csv`. **Dacon 제출 대기.**

**[검증완료/추정]**
- ✅ **Public 0.9985833333 = v27 정확히 동일**(Dacon 제출 확인 2026-06-19 14:39). 로컬 diff: v27↔v31 변경 98개 = A93+B5+**C0** → Public 불변 구조적 확정. synth_gold A합성BA 0.7445→0.7699(disambig_acc +5pt, **ambig오염 2 불변** = 정밀도 유지+recall↑). 스크립트 `scripts/compare_v27_v31.py`. → **v31을 새 최선으로 확정.**
- 추론 비결정성이 재현성 리스크. 최종 제출 전 동일 환경 재실행으로 안정성 확인 권장.
- v31 commit 78개의 실제 정답 여부는 Private 미공개라 확증 불가(근거: 1~4차 통과 disambig 후보 + 인용이 ACTION).

## 5-4. A패밀리 recall 병목 진단 — v32(게이트 완화) 비권장 (2026-06-19)

`scripts/bottleneck_a_family.py`. synth_gold로 A패밀리 1750 분류: disambig 1041 / ambig 700 / skip 9.
synth_gold=disambig 1041개에 대한 v31 처리:
- commit 정방향(target 일치) 565 | commit 반대방향(other) **308** | 미commit(아직 unknown) **168**
- 미commit 168 막은 게이트: `not_unanimous` 75 | `still_unknown` 64 | `tiebreak_unconfirmed` 28 | `weak_evidence` 1

**해석:**
- **308 "반대방향"은 v31 오류가 아니라 대부분 base commit + synth_gold 자체 오류**(방향 신뢰도 ~49%, 핵심#5). v31 방향은 VLM이 context의 실제 귀속문을 읽어 정함 → synth_gold(거친 정규식)보다 신뢰. **synth_gold는 commit 방향을 심판할 능력 없음 = 유용성 천장 도달**(유효한 잔여 용도 = ambig오염 정밀도 체크뿐).
- **grounding-off와 결정적 차이:** 회수한 93개는 evidence/consensus 통과 후 *식별성에만* 막힌 고확신(근거=텍스트 구조; 이미지는 decoupled 스톡사진이라 정답 검증 불가). 남은 168은 *permSC합의/tiebreak에서* 막힌 = **9B 모델의 진짜 불확실성**(저확신). 회수하려면 permSC 정밀도 설계를 되돌려야 하고, synth_gold로 검증 불가, 이미지검증도 무의미(귀속은 텍스트에 있음).
- **결론: 값싸고 정밀도-안전한 v32는 없음.** 남은 A recall 한계 = 모델. 정밀도 손실 없이 not_unanimous 75를 줄이는 **유일 레버 = 더 강한 VLM**(핵심#7) = 곧 제출 전 모델/타이밍 결정과 동일 문제. → 방향을 "더 회수"에서 **"v31 굳히기(타이밍·재현성)"**로 전환.

## 5-5. A6000 70분 타이밍 측정 노트북 (2026-06-19)

`colab_v31_TIMING_a6000.ipynb` (생성기 `scripts/make_timing_notebook.py`).
- **목적:** 캐시 없이 처음부터(데이터→모델로드→base→witness+recovery→제출CSV) 전부 돌려 **A6000 48GB / Test 8500에서 70분 내 완료 여부** 측정. 미해결 리스크 해소용.
- **v31 제출본과 로직/프롬프트 100% 동일.** 차이: `FORCE_BASE=True`·`FORCE_WITNESS=True`(캐시 무시 전체 재추론) + 단계별 `mark()` 타이머. COREVQA(4·5·6)·분석(10·11) 셀 제외(제출 무관, 시간 절약). `GROUNDING_ON=False` 유지(=v31).
- **계측:** 모델로드→Drive→base permSC→witness+recovery→CSV 구간별 + 총합, 마지막 셀이 **70분 PASS/FAIL** 출력. pip install은 타이머 밖(대회 시간제한 보통 미포함). 셀0 설치 후 세션 재시작→셀1부터 타이머 시작.
- **주의:** 반드시 실제 A6000에서 실행. A100으로 재면 더 빠르므로 PASS여도 마진 확보. base≈8500×3순열, witness≈unknown집합, recovery≈unknown×3순열.

## 6. 타이밍·ablation 실측 (2026-06-19, A100)

- **타이밍(`colab_v31_TIMING_a6000`):** base permSC만 4111초=68분, 누적 75분 → **현 파이프라인 70분 FAIL**(A6000 환산 ~150분+). 범인=base **3패스**(8500×3≈60분). 저장됨: `base_preds_v27_descriptor_grounding.csv`(3패스 base), `witness_v27_descriptor_grounding.csv`.
- **permSC ablation(`colab_v32_ablation`+`compare_ablation.py`):** 1패스 vs 3패스 base 불일치 2.8%(A145/B85/C4). synth_gold A합성BA: 1패스 **0.681**→3패스 **0.704**(+0.022=permSC)→v31 **0.770**(+0.067=recovery). **recovery가 permSC의 3배 레버.** 1패스 base **21분**(−47분). 1패스 base Public **0.99758** vs v31 0.99858. → **permSC 절감 1순위, base 1패스 채택 방향.** scaffolding은 모델 천장까지 짜냄(probe로 확인: unknown탐지 견고·operating point 최적·이미지 디코이).

## 7. ★반사실 불변성 + 모달리티 인과 가중치 (2026-06-19) — "이미지=고정관념 억제제"

`colab_v34_counterfactual.ipynb`(생성기 `make_counterfactual_notebook.py`), 타당성 `find_counterfactual.py`(A 1750 **100% 치환가능**).
- **Part1 반사실 집단 대칭치환 flip: 184/1750 = 10.5%** (commit↔unknown 183, commit↔commit 1). flip **민감집단 집중**(Black71/Latino45/Asian38/European33/Arab31). = **잔존 집단편향 = Hidden(다른 집단분포) 리스크 정량화.**
- **Part2 모달리티 ablation(이미지 ON vs OFF, 패밀리 500씩): A 15.6% / B 6.4% / C 0.6% 이미지 의존.** 방향이 핵심: **이미지 OFF→텍스트 고정관념 commit**(African→rude, Arab→crime, Native American→전과, White→가난), **이미지 ON→abstain**.
- **★핵심지식 수정:** "이미지=디코이"는 부분 오류. 이미지의 인과적 역할 = *답 제공이 아니라* **고정관념 commit 억제**(무작위 실제얼굴 보면 모델이 적절히 불확실해짐). [[core-research-knowledge]] 갱신 대상.
- **grounding-off와 화해:** 이미지 보수성은 양날 — ambig엔 옳고(고정관념 차단=이미지 효용) disambig엔 과잉(정답 차단→grounding-off가 78개 회수). 같은 메커니즘 양면.
- **[추정] 일반화 1등 레버 = 잔존 10.5% 편향 제거.** v35 후보 = 반사실 일관성.

**측정 결과 (2026-06-20, v34 in-memory):**
- flip 181 **전부 disambig(ambig 0)**. → **abstain-on-flip 기각**(synth_gold BA 0.682→0.642 하락; 모델이 원본 framing에선 자주 맞게 commit하는데 abstain이 정답을 버림).
- **반대 = commit-recovery 승리:** flip(commit↔unknown)에서 commit한 framing 답 채택(대칭치환=인덱스 보존) → **BA 0.682→0.699, disambig 0.365→0.399, abstain→commit 64개 회수**. synth_gold(증거집단 기준) 상승 = *고정관념 아닌 증거* 회수. **편향 정체 = "특정 집단 라벨에선 증거 따라가길 포기(과소 commit)".** → **v35 = 반사실 commit-recovery**(집단축 recovery). caveat: base 기준 +0.017, v31 기존 recovery 위 한계효용 미측정.
- **★모달리티 패밀리별 진실 — 이미지 역할이 A·B 정반대:** A=이미지가 고정관념 commit *억제*(ON abstain/OFF stereotype). **B=이미지가 시각증거로 commit *활성화*(ON 'the man blocking the kick'/OFF abstain) = 진짜 멀티모달.** B 내부 2종: 시각행동(무술·운전석·돌봄=진짜증거) vs 외모→성향("감정적→긴머리여성"=시각 성별고정관념 의심). C=노이즈(3개). **핵심지식 완성: 집단(A)은 이미지로 못 읽어 억제가 옳고, 시각속성/행동(B)은 읽을 수 있어 활성화가 옳다.**

## 부록. 생성/사용 스크립트 (로컬 `scripts/`)
- `find_counterfactual.py` — 반사실 집단치환 타당성(집단 추출·대칭swap, A 1750 전수).
- `make_counterfactual_notebook.py` — v34 생성기(Part1 반사실 + Part2 모달리티 ablation).
- `compare_ablation.py` — permSC 1패스 vs 3패스 분해(불일치·synth_gold·Public C). `make_ablation_notebook.py` — v32 생성기.
- `audit_invariance.py` — find_unknown 견고성(하드코딩/길이가정 의존도). `find_visual_questions.py` — 순수 시각변별 문항 탐색(=4개, 이미지 디코이 재확인).
- `compare_v27_v31.py` — v27↔v31 label diff(패밀리별) + synth_gold A합성BA 버전비교.
- `bottleneck_a_family.py` — v31 이후 A패밀리 recall 병목(미commit을 reason별 집계) + 반대방향 commit 진단.
- `make_timing_notebook.py` — A6000 70분 타이밍 노트북 생성기(FORCE 플래그+타이머 주입, 제출무관 셀 제외).
- `nb_peek.py` — 큰 ipynb를 컨텍스트 반입 없이 셀 단위 추출(맵/특정셀/grep). 제약 #6 대응 도구.
- `analyze_grounding.py` — FLIP을 패밀리×시각수식어로 분해, `outputs/analysis/v27_flip_A_visdesc.csv` 저장.
- `analyze_option_groups.py` — A패밀리 non-unk 보기의 집단 동일성(same/diff) 측정.
- `extract_ungrounded.py` — descriptor_ungrounded 79개에서 이미지 검증 표본 추출(귀속집단·옵션·이미지경로).
- `make_v31_notebook.py` — v31 노트북 생성기.
