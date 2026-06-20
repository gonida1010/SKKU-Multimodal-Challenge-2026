# v35 = lean counterfactual-debiased 제출본 생성.
# base=1패스 run_single(전체) + A집단치환/B성별치환 통합규칙. permSC/witness/recovery 없음(70분 예산).
# 규칙: 원본==치환 유지 | commit↔unknown→commit채택(증거회수,A) | commit↔commit→abstain(고정관념제거,B).
import json
SRC = r"C:\Users\pak10\Downloads\colab_v31_grounding_off.ipynb"
OUT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v35_cf_debias.ipynb"
nb = json.load(open(SRC, encoding="utf-8")); S = nb["cells"]
def body(i): return "".join(S[i]["source"])
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}

INTRO = """# 🎯 v35 — 반사실 디바이어싱 제출본 (집단축 A + 성별축 B 통합)

**새 레버(scaffolding 아님):** 정체성(집단/성별)을 대칭 치환해도 답이 흔들리면 = 모델이 증거 아닌 prior 사용. 그걸 교정.

**통합 규칙** (원본 답 po, 치환 답 pc; 대칭치환이라 인덱스 보존):
- `po==pc` → 유지 (불변=신뢰)
- `commit↔unknown` → **commit 채택** (한 framing에 증거 있음 = 회수). A 집단축. *측정: synth_gold +0.017*
- `commit↔commit`(서로 다름) → **abstain** (순수 정체성 고정관념 = 증거 없음). B 성별축. *모호문항 오답 제거 → ambig_acc↑*

**파이프라인:** base = **1패스 run_single 전체**(permSC/witness/recovery 제거 → 70분 예산 충족). A패밀리=집단 대칭치환, B패밀리 M-vs-F=성별 대칭치환. C·기타=원본 유지.

**실행:** 셀0 설치→재시작→순서대로. 1패스 base(전체) + 1패스 치환(A·B 대상). ~40분(A100). 끝에 **v31 대비 synth_gold + label diff** 출력.
**산출:** `outputs/submission_v35_cf_debias.csv`.
"""

DATA = r"""# ===== A: 데이터 + 768 이미지 (전체) =====
import os, zipfile, csv, json, time
from tqdm.auto import tqdm
from google.colab import drive
drive.mount('/content/drive')
PROJECT='/content/drive/MyDrive/SKKU-Multimodal-Challenge-2026'
os.makedirs(f'{PROJECT}/outputs',exist_ok=True)
ZIP=f'{PROJECT}/open.zip'
if not os.path.isdir('/content/open') and not os.path.isdir('/content/test'):
    with zipfile.ZipFile(ZIP) as z: z.extractall('/content')
TEST_DIR=next((c for c in ['/content/open/test','/content/test'] if os.path.isdir(c)), None)
IMG_ROOT=TEST_DIR
rows, ids = [], []
with open(f'{TEST_DIR}/test.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        ans=json.loads(r['answers'])
        rows.append({'ctx':r['context'],'q':r['question'],'answers':ans,'unk':find_unknown(ans),'path':r['image_path']})
        ids.append(r['sample_id'])
print(f"테스트 {len(rows)}건")
t=time.time(); images=[load_img(r['path']) for r in tqdm(rows,desc='img768')]; print(f"이미지 {time.time()-t:.0f}s")
"""

CFBUILD = r"""# ===== B: 반사실 생성 (A 집단치환 / B 성별치환) — 이미지는 base와 공유 =====
import re
RE_A=re.compile(r"^\s*The image shows",re.I); RE_IMG=re.compile(r"\b(image|photo|picture)\b",re.I)
def fam(c): return 'A' if RE_A.search(c) else ('B' if RE_IMG.search(c) else 'C')
OPT_GRP=re.compile(r"^\s*(?:The|An?)\s+(.+?)\s+(?:person|man|woman|individual|people|guy|lady)\b",re.I)
def grp(o):
    m=OPT_GRP.search(o); return m.group(1).strip() if m else None
MALE=re.compile(r"\b(man|men|male|boy|gentleman|guy|father|son|brother|husband)\b",re.I)
FEMALE=re.compile(r"\b(woman|women|female|girl|lady|mother|daughter|sister|wife)\b",re.I)
def g_of(o):
    m=bool(MALE.search(o)); f=bool(FEMALE.search(o)); return 'M' if(m and not f)else('F' if(f and not m)else'-')
def ssw(text,a,b):
    t=re.sub(rf'\b{re.escape(a)}\b','\x00',text,flags=re.I); t=re.sub(rf'\b{re.escape(b)}\b',a,t,flags=re.I); return t.replace('\x00',b)
GP=[('woman','man'),('women','men'),('female','male'),('girl','boy'),('lady','gentleman'),
    ('mother','father'),('daughter','son'),('sister','brother'),('wife','husband'),('she','he'),('her','his')]
def gsw(t):
    for a,b in GP: t=ssw(t,a,b)
    return t

cf_type=[None]*len(rows); cf_rows=[]; cf_map=[]
for k,r in enumerate(rows):
    a=r['answers']; unk=r['unk']; non=[i for i in range(len(a)) if i!=unk]
    if len(non)!=2: continue
    f=fam(r['ctx'])
    if f=='A':
        g0,g1=grp(a[non[0]]),grp(a[non[1]])
        if not g0 or not g1 or g0.lower()==g1.lower(): continue
        if not (re.search(rf'\b{re.escape(g0)}\b',r['ctx'],re.I) and re.search(rf'\b{re.escape(g1)}\b',r['ctx'],re.I)): continue
        sc=ssw(r['ctx'],g0,g1); sa=[ssw(o,g0,g1) for o in a]; cf_type[k]='A'
    elif f=='B':
        if set([g_of(a[non[0]]),g_of(a[non[1]])])!={'M','F'}: continue
        sc=gsw(r['ctx']); sa=[gsw(o) for o in a]; cf_type[k]='B'
    else:
        continue
    cf_rows.append({'ctx':sc,'q':r['q'],'answers':sa,'unk':find_unknown(sa)}); cf_map.append(k)
cf_imgs=[images[k] for k in cf_map]
print(f"반사실 대상: A {cf_type.count('A')} + B {cf_type.count('B')} = {len(cf_map)} / 8500")
"""

RUN = r"""# ===== C: 추론 (base 전체 + 반사실) =====
import time
t=time.time(); base=run_single(rows, images);     print(f"base(8500) {time.time()-t:.0f}s = {(time.time()-t)/60:.1f}분")
t=time.time(); cf  =run_single(cf_rows, cf_imgs);  print(f"반사실({len(cf_rows)}) {time.time()-t:.0f}s")
"""

RULE = r"""# ===== D: 통합 규칙 적용 + 제출 저장 =====
import csv
cf_pred={cf_map[j]:cf[j] for j in range(len(cf_map))}
cf_unk ={cf_map[j]:cf_rows[j]['unk'] for j in range(len(cf_map))}
final=list(base)
n_rec=n_abs=n_inv=0
for k in range(len(rows)):
    if k not in cf_pred: continue
    po,pc,uo,uc=base[k],cf_pred[k],rows[k]['unk'],cf_unk[k]
    if po==pc: n_inv+=1; continue
    if po!=uo and pc==uc: continue                       # 원본 commit, 치환 abstain -> 원본 유지
    if po==uo and pc!=uc: final[k]=pc; n_rec+=1          # 원본 abstain, 치환 commit -> 회수(A)
    elif po!=uo and pc!=uc and po!=pc: final[k]=uo; n_abs+=1  # commit↔commit -> abstain(B)
print(f"불변 {n_inv} | commit-recovery(A) {n_rec} | 성별abstain(B) {n_abs} | base와 변경 {n_rec+n_abs}")
OUT=f'{PROJECT}/outputs/submission_v35_cf_debias.csv'
with open(OUT,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['sample_id','label'])
    for i,p in zip(ids,final): w.writerow([i,p])
print("저장:",OUT)
"""

CMP = r"""# ===== E: v31 대비 비교 (synth_gold A + label diff) =====
import csv, re
V31=f'{PROJECT}/outputs/submission_v31_grounding_off.csv'
p31={r['sample_id']:int(r['label']) for r in csv.DictReader(open(V31,encoding='utf-8'))} if os.path.exists(V31) else None
ATTR=re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b(.{0,120})")
EV_NEG=re.compile(r"struggl|difficult|fail|lost|steal|stole|interrupt|yell|rude|late|cheat|lie|lied|could not|couldn t|cannot|debt|fell apart|shoplift|arrest|mock|complain|trouble")
Q_NEG=re.compile(r"less likely|not |unlikely|least ")
TRAIT_NEG=re.compile(r"lie|steal|poor|criminal|rude|fail|violen|less educated|unemploy|suspicious|emotional|weak|lazy|cheat|trouble|danger")
def sg(ctx,q,a,unk):
    m=ATTR.search(' '+str(ctx))
    if not m: return ('ambig',unk)
    g=m.group(1).strip().lower(); h=[i for i,x in enumerate(a) if g in x.lower() and i!=unk]
    if len(h)!=1: return ('skip',None)
    tgt=h[0]; oth=[i for i in range(3) if i not in (tgt,unk)][0]
    ev=bool(EV_NEG.search(m.group(2).lower())); ql=str(q).lower()
    return ('disambig', tgt if ((bool(TRAIT_NEG.search(ql))==ev)!=bool(Q_NEG.search(ql))) else oth)
RE_A=re.compile(r"^\s*The image shows",re.I)
A=[k for k in range(len(rows)) if RE_A.search(rows[k]['ctx'])]
def ba(pred_by_id):
    okA=okD=nA=nD=0
    for k in A:
        r=rows[k]; t,g=sg(r['ctx'],r['q'],r['answers'],r['unk'])
        if t=='skip' or g is None: continue
        p=pred_by_id[ids[k]]
        if t=='ambig': nA+=1; okA+=(p==g)
        else: nD+=1; okD+=(p==g)
    return ((okA/max(1,nA))+(okD/max(1,nD)))/2, okA/max(1,nA), okD/max(1,nD)
final_by_id={ids[k]:final[k] for k in range(len(rows))}
base_by_id={ids[k]:base[k]  for k in range(len(rows))}
print("=== synth_gold A합성BA ===")
b,a,d=ba(base_by_id);  print(f"  base(1패스)  : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
b,a,d=ba(final_by_id); print(f"  v35(디바이어싱): BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
if p31:
    b,a,d=ba(p31); print(f"  v31(기존최선) : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
    diff=sum(1 for k in range(len(rows)) if final[k]!=p31[ids[k]])
    from collections import Counter
    fc=Counter('A' if RE_A.search(rows[k]['ctx']) else ('B' if re.search(r'\b(image|photo|picture)\b',rows[k]['ctx'],re.I) else 'C') for k in range(len(rows)) if final[k]!=p31[ids[k]])
    print(f"  v35 vs v31 label diff {diff} | 패밀리 {dict(fc)} (C=Public 영향)")
print("※ v35가 v31보다 BA↑면: 새 레버가 무거운 scaffolding을 이김(+70분 예산도 충족).")
"""

cells=[md(INTRO), code(body(0)), code(body(1)), code(body(2)), code(body(3)),
       code(DATA), code(CFBUILD), code(RUN), code(RULE), code(CMP)]
nb["cells"]=cells
json.dump(nb, open(OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("생성:", OUT, "| 셀", len(cells))
