# v32 ablation 노트북 생성: permSC(3패스) vs 1패스 정확도/시간 비교.
# 셀 0,1,2,3은 v31 원본 재사용(설치/모델/헬퍼/Drive). 새 셀로 1패스 base + 비교만 추가.
import json

SRC = r"C:\Users\pak10\Downloads\colab_v31_grounding_off.ipynb"
OUT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v32_ablation.ipynb"
nb = json.load(open(SRC, encoding="utf-8"))
S = nb["cells"]
def body(i): return "".join(S[i]["source"])
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": s.splitlines(keepends=True)}
def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}

INTRO = """# 🔬 v32 ablation — permSC(3패스) vs 1패스 : 정확도 vs 시간

**가설:** 68분짜리 3패스 self-consistency의 효용은 작다. 직전 타이밍런 로그에서 `arbiter 종합: 506/8500` = **3패스가 실제로 바꾼 건 6%뿐**(94%는 1패스로도 동일). 이걸 숫자로 검증한다.

**무엇을 재나**
1. **1패스 base**를 새로 추론(~23분 예상) → 3패스 base(직전 런이 Drive에 저장)와 비교.
2. 1패스 vs 3패스 **불일치 %**(패밀리별) + **synth_gold A합성BA**(둘 다).
3. 1패스 base **소요 시간**.

**결과 해석:** 불일치가 작고 synth_gold가 비슷하면 → permSC 3패스는 버려도 됨(=base 68→23분, 70분 예산 확보). 그 예산으로 더 강한 모델 가능.

**실행:** 셀0 설치 → 세션 재시작 → 순서대로. **전제:** 직전 타이밍런이 `outputs/base_preds_v27_descriptor_grounding.csv`(3패스)를 Drive에 저장해둔 상태(로그의 'base 저장 완료'). 없으면 셀 C가 알려줌.

**산출물:** `outputs/submission_v32_1pass_base.csv` — recovery 없는 1패스 base. Public에 제출해 v31(0.99858)과 비교하면 "permSC+recovery가 Public을 올리긴 하나"를 직접 확인.
"""

DATA = r"""# ===== A: 데이터 + 768 이미지 로드 (base 추론 없이) =====
import os, time, zipfile, csv, json
from tqdm.auto import tqdm
from google.colab import drive
drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/SKKU-Multimodal-Challenge-2026'
os.makedirs(f'{PROJECT}/outputs', exist_ok=True)
ZIP = f'{PROJECT}/open.zip'
if not os.path.isdir('/content/open') and not os.path.isdir('/content/test'):
    with zipfile.ZipFile(ZIP) as z: z.extractall('/content')
TEST_DIR = next((c for c in ['/content/open/test', '/content/test'] if os.path.isdir(c)), None)
TEST_CSV = f'{TEST_DIR}/test.csv'
IMG_ROOT = TEST_DIR                      # load_img(셀1)이 참조하는 전역
rows, ids = [], []
with open(TEST_CSV, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        ans = json.loads(r['answers'])
        rows.append({'ctx': r.get('context',''), 'q': r.get('question',''),
                     'answers': ans, 'unk': find_unknown(ans), 'path': r['image_path']})
        ids.append(r['sample_id'])
print(f"테스트 {len(rows)}건")
_t = time.time()
images_768 = [load_img(r['path']) for r in tqdm(rows, desc='img768')]
print(f"이미지 768 로드 {time.time()-_t:.0f}s")
"""

ONEPASS = r"""# ===== B: 1패스 base 추론 (permSC 없음, 단일 greedy) — 시간 측정 =====
import time, csv
_t0 = time.time()
preds_1pass = run_single(rows, images_768)     # 셀2 정의: generate 1회 + parse
T_1PASS = time.time() - _t0
print(f"[1패스 base] {T_1PASS:.0f}s = {T_1PASS/60:.1f}분")
with open(f'{PROJECT}/outputs/base_preds_1pass.csv','w',newline='',encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['sample_id','label'])
    for i,p in zip(ids, preds_1pass): w.writerow([i,p])
_u1 = sum(preds_1pass[i]==rows[i]['unk'] for i in range(len(rows)))
print(f"1패스: unknown {_u1} / commit {len(rows)-_u1}")
"""

LOAD3 = r"""# ===== C: 3패스(permSC) base 로드 (직전 타이밍런이 저장한 캐시) =====
import csv, os
BASE3 = f'{PROJECT}/outputs/base_preds_v27_descriptor_grounding.csv'
assert os.path.exists(BASE3), f"3패스 base 캐시 없음: {BASE3}\n  -> 직전 타이밍런이 'base 저장 완료'까지 갔어야 함. 없으면 timing 노트북 셀7만 다시 실행."
id2b = {}
with open(BASE3, encoding='utf-8') as f:
    for r in csv.DictReader(f): id2b[r['sample_id']] = int(r['label'])
assert all(i in id2b for i in ids), "캐시가 현재 test와 불일치"
preds_3pass = [id2b[i] for i in ids]
_u3 = sum(preds_3pass[i]==rows[i]['unk'] for i in range(len(rows)))
print(f"3패스 base 로드: unknown {_u3} / commit {len(rows)-_u3}")
"""

COMPARE = r"""# ===== D: 비교 — 불일치% + synth_gold A합성BA + 시간 결론 =====
import re
from collections import Counter
RE_A  = re.compile(r"^\s*The image shows", re.I)
RE_IMG= re.compile(r"\b(image|photo|picture)\b", re.I)
def fam(c): return 'A' if RE_A.search(c) else ('B' if RE_IMG.search(c) else 'C')

diff = [i for i in range(len(rows)) if preds_1pass[i] != preds_3pass[i]]
dfam = Counter(fam(rows[i]['ctx']) for i in diff)
print(f"=== 1패스 vs 3패스(permSC) 불일치: {len(diff)}/{len(rows)} ({len(diff)/len(rows)*100:.1f}%) ===")
print("  패밀리별 불일치:", dict(dfam))
print(f"  (직전 런 arbiter=506/8500=6.0% 와 비교: 불일치가 이 부근이면 permSC 효용 작음)")

ATTR = re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b(.{0,120})")
EV_NEG = re.compile(r"struggl|difficult|fail|lost|steal|stole|interrupt|yell|rude|late|cheat|lie|lied|could not|couldn t|cannot|debt|fell apart|shoplift|arrest|mock|complain|trouble")
Q_NEG = re.compile(r"less likely|not |unlikely|least ")
TRAIT_NEG = re.compile(r"lie|steal|poor|criminal|rude|fail|violen|less educated|unemploy|suspicious|emotional|weak|lazy|cheat|trouble|danger")
def synth_gold(ctx, q, ans, unk):
    m = ATTR.search(' ' + str(ctx))
    if not m: return ('ambig', unk)
    grp = m.group(1).strip().lower()
    hits = [i for i,a in enumerate(ans) if grp in a.lower() and i != unk]
    if len(hits) != 1: return ('skip', None)
    tgt = hits[0]; other = [i for i in range(3) if i not in (tgt, unk)][0]
    ev = bool(EV_NEG.search(m.group(2).lower())); ql = str(q).lower()
    return ('disambig', tgt if ((bool(TRAIT_NEG.search(ql))==ev) != bool(Q_NEG.search(ql))) else other)

A_idx = [i for i in range(len(rows)) if fam(rows[i]['ctx'])=='A']
def synth_ba(preds):
    okD=nD=okA=nA=0
    for i in A_idx:
        t,g = synth_gold(rows[i]['ctx'], rows[i]['q'], rows[i]['answers'], rows[i]['unk'])
        if t=='skip' or g is None: continue
        if t=='ambig': nA+=1; okA+=(preds[i]==g)
        else: nD+=1; okD+=(preds[i]==g)
    return ((okA/max(1,nA))+(okD/max(1,nD)))/2, okD/max(1,nD), okA, nA
print("\n=== synth_gold A합성BA (A패밀리, 버전비교용 프록시) ===")
for nm, pr in [("1패스", preds_1pass), ("3패스", preds_3pass)]:
    b,d,oa,na = synth_ba(pr)
    print(f"  {nm}: A합성BA={b:.4f}  disambig_acc={d:.4f}  ambig정답={oa}/{na}")

print("\n=== 시간 결론 ===")
print(f"  1패스 base: {T_1PASS/60:.1f}분  |  3패스 base(직전 측정): ~68분")
print(f"  절감: ~{68 - T_1PASS/60:.0f}분  (A6000 환산 ×~1.6)")
print("  -> 불일치 작고 synth_gold 비슷하면: permSC 버리고 base 1패스 채택 = 70분 예산 확보")
"""

SAVE = r"""# ===== E: 1패스 base 제출본 저장 (Public 직접 비교용) =====
import csv
OUT = f'{PROJECT}/outputs/submission_v32_1pass_base.csv'
with open(OUT,'w',newline='',encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['sample_id','label'])
    for i,p in zip(ids, preds_1pass): w.writerow([i,p])
print("저장:", OUT)
print("이걸 Dacon에 제출 -> Public이 v31(0.9985833333)과 같으면: permSC+recovery는 Public에 무의미(=전부 Private용).")
print("낮으면: 그 차이가 permSC+recovery가 Public에서 버는 실점수.")
"""

cells = [md(INTRO), code(body(0)), code(body(1)), code(body(2)), code(body(3)),
         code(DATA), code(ONEPASS), code(LOAD3), code(COMPARE), code(SAVE)]
nb["cells"] = cells
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("생성:", OUT, "| 셀", len(cells), "개")
