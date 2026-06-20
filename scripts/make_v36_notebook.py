# v36 = v35(반사실 디바이어싱) + v31(recovery) 합본.
# Pipeline: 1패스 base → 1패스 CF → 디바이어싱 규칙 → recovery(잔여 unknown)
# 시간: ~21min(base) + ~9min(CF) + ~22min(recovery) ≈ 52min A6000 (70분 예산 충족)
import json

SRC = r"C:\Users\pak10\Downloads\colab_v31_grounding_off.ipynb"
OUT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v36_cf_recovery.ipynb"

nb = json.load(open(SRC, encoding="utf-8")); S = nb["cells"]
def body(i): return "".join(S[i]["source"])
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}

# Cell 8 = recovery. Split: function definitions vs execution block.
cell8 = body(8)
_marker = '\nimport csv as _csv'
_split = cell8.index(_marker)
RECOVERY_FUNCS = cell8[:_split].rstrip() + '\n'

INTRO = """# v36 — 반사실 디바이어싱 + Recovery 통합본

**v35의 새 레버(반사실 디바이어싱) + v31의 최대 레버(recovery)를 합친 최종 파이프라인.**

**파이프라인:**
1. **base 1패스** run_single 전체 8500 (768px, ~21min A6000)
2. **반사실 1패스** A 집단치환 + B 성별치환 대상 (~3500, ~9min)
3. **디바이어싱 규칙** 적용 (invariant=유지, commit↔unknown=회수, commit↔commit=abstain)
4. **Recovery** 잔여 unknown → 1024px witness + recovery_permsc 3패스 + 다단 게이트 (~22min)

**예산:** ~52min A6000 / ~35min A100. 70분 충족.
**산출:** `outputs/submission_v36_cf_recovery.csv`

**실행:** 셀0 설치→재시작→순서대로.
"""

DATA = r"""# ===== A: 데이터 + 768 이미지 =====
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
t=time.time(); images=[load_img(r['path']) for r in tqdm(rows,desc='img768')]; print(f"이미지 로드 {time.time()-t:.0f}s")
"""

CFBUILD = r"""# ===== B: 반사실 생성 (A 집단치환 / B 성별치환) =====
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

RUN = r"""# ===== C: 추론 (base 1패스 + 반사실 1패스) =====
import time
t0=time.time(); base=run_single(rows, images); print(f"base(8500) {time.time()-t0:.0f}s = {(time.time()-t0)/60:.1f}분")
t0=time.time(); cf=run_single(cf_rows, cf_imgs); print(f"반사실({len(cf_rows)}) {time.time()-t0:.0f}s")
"""

DEBIAS = r"""# ===== D: 반사실 디바이어싱 규칙 =====
cf_pred={cf_map[j]:cf[j] for j in range(len(cf_map))}
cf_unk={cf_map[j]:cf_rows[j]['unk'] for j in range(len(cf_map))}
debiased=list(base)
n_rec=n_abs=n_inv=0
for k in range(len(rows)):
    if k not in cf_pred: continue
    po,pc,uo,uc=base[k],cf_pred[k],rows[k]['unk'],cf_unk[k]
    if po==pc: n_inv+=1; continue
    if po!=uo and pc==uc: continue
    if po==uo and pc!=uc: debiased[k]=pc; n_rec+=1
    elif po!=uo and pc!=uc and po!=pc: debiased[k]=uo; n_abs+=1
print(f"디바이어싱: 불변 {n_inv} | commit-recovery {n_rec} | abstain {n_abs} | 변경 {n_rec+n_abs}")
unk_mask=[debiased[i]==rows[i]['unk'] for i in range(len(rows))]
print(f"recovery 대상(디바이어싱 후 unknown): {sum(unk_mask)} / {len(rows)}")
"""

RECOVERY_EXEC = r"""# ===== F: Recovery 실행 (1024px witness + permSC 3패스 + 다단 게이트) =====
import time
from tqdm.auto import tqdm
VER='v36_cf_recovery'
unk_idx_list=[i for i in range(len(rows)) if unk_mask[i]]
print(f"recovery 대상: {len(unk_idx_list)}건 (디바이어싱 후 잔여 unknown)")
u_rows=[rows[i] for i in unk_idx_list]
t0=time.time()
u_imgs=[load_img_hires(rows[i]['path']) for i in tqdm(unk_idx_list,desc='img1024')]
print(f"1024 이미지 {time.time()-t0:.0f}s")
t0=time.time(); u_wit=witness_pass(u_rows,u_imgs); print(f"witness {time.time()-t0:.0f}s")
t0=time.time(); local_flips,rec_diag=recovery_permsc(u_rows,u_imgs,u_wit)
flips={unk_idx_list[j]:p for j,p in local_flips.items()}
from collections import Counter
print(f"recovery {time.time()-t0:.0f}s | flip {len(flips)}건 | 사유:", Counter(d['reason'] for d in rec_diag))
"""

FINAL_CMP = r"""# ===== G: 최종 저장 + v31 비교 =====
import csv, re, os
final=list(debiased)
for i,p in flips.items(): final[i]=p
n_total=n_rec+n_abs+len(flips)
print(f"총 변경: 디바이어싱 {n_rec+n_abs}(rec={n_rec},abs={n_abs}) + recovery {len(flips)} = {n_total}")
OUT=f'{PROJECT}/outputs/submission_v36_cf_recovery.csv'
with open(OUT,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['sample_id','label'])
    for i,p in zip(ids,final): w.writerow([i,p])
print("저장:",OUT)

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
base_by_id={ids[k]:base[k] for k in range(len(rows))}
debiased_by_id={ids[k]:debiased[k] for k in range(len(rows))}
print("=== synth_gold A합성BA ===")
b,a,d=ba(base_by_id);     print(f"  base(1패스)    : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
b,a,d=ba(debiased_by_id); print(f"  디바이어싱only : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
b,a,d=ba(final_by_id);    print(f"  v36(최종)      : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
if p31:
    b,a,d=ba(p31); print(f"  v31(기존최선)  : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
    diff=sum(1 for k in range(len(rows)) if final[k]!=p31[ids[k]])
    from collections import Counter
    fc=Counter('A' if RE_A.search(rows[k]['ctx']) else ('B' if re.search(r'\b(image|photo|picture)\b',rows[k]['ctx'],re.I) else 'C') for k in range(len(rows)) if final[k]!=p31[ids[k]])
    print(f"  v36 vs v31 label diff {diff} | 패밀리 {dict(fc)}")
print()
print("v36 = 디바이어싱(새 레버) + recovery(기존 최대 레버) 합본.")
print("BA가 v31보다 높으면 = 새 레버가 기존 위에 순수 가치 추가(+70분 충족).")
"""

cells = [
    md(INTRO),
    code(body(0)),         # 셀0: pip install
    code(body(1)),         # 셀1: model/imports
    code(body(2)),         # 셀2: helpers (run_single, PERMS, etc.)
    code(body(3)),         # 셀3: drive mount
    code(DATA),            # 셀A: 데이터 + 768 이미지
    code(CFBUILD),         # 셀B: 반사실 생성
    code(RUN),             # 셀C: 1패스 base + CF
    code(DEBIAS),          # 셀D: 디바이어싱 규칙 → unk_mask
    code(RECOVERY_FUNCS),  # 셀E: recovery 함수 정의 (v31 cell8 함수부)
    code(RECOVERY_EXEC),   # 셀F: recovery 실행
    code(FINAL_CMP),       # 셀G: 저장 + 비교
]

nb["cells"] = cells
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("생성:", OUT, "| 셀", len(cells))
