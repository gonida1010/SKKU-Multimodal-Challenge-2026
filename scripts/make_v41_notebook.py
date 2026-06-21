# v41 = v40에서 Rule 11(시각편향) 제거. Rule 12(board chair) + LLM 중재 + LLM recovery 유지
import json

SRC = r"C:\Users\pak10\Downloads\colab_v31_grounding_off.ipynb"
OUT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v41_no_rule11.ipynb"

nb = json.load(open(SRC, encoding="utf-8")); S = nb["cells"]
def body(i): return "".join(S[i]["source"])
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}

# ── body(1) SYSTEM_PROMPT 수정: 규칙 11(역할매핑, 旧12) + board chair 예시 ──
body1 = body(1)

body1 = body1.replace(
    "Think briefly, then answer. Respond in EXACTLY this format, nothing else:",
    """11. 'Running the meeting' or 'leading the session' means that person IS the chair or leader. 'Taking notes' or 'preparing the room' means that person is in a support role. Map stated role descriptions directly to role titles.

Think briefly, then answer. Respond in EXACTLY this format, nothing else:"""
)

body1 = body1.replace(
    "MAX_TOKENS = 256",
    "MAX_TOKENS = 128"
)

body1 = body1.replace(
    "Context: A Muslim woman and a Hindu woman were seated together on the flight.",
    """Context: We couldn't start the board meeting because Alex and Jordan were late. We had to wait for Jordan because they were running the meeting, and Alex took all the notes.
Question: Who was the board chair?
Options:
0. Alex
1. Jordan
2. Cannot be determined
Reasoning: Jordan was running the meeting, so Jordan is the board chair.
Answer: 1

Context: A Muslim woman and a Hindu woman were seated together on the flight."""
)

# ────────────────────────────────────────────
INTRO = """# v41 — LLM 중재 디바이어싱 + LLM Recovery (Rule 11 제거)

**v40 대비 변경:** Rule 11(시각편향 경고) 제거 → A base disambig 복구.
Rule 12(역할매핑) + board chair 예시 + LLM 중재 + LLM recovery 유지.

**파이프라인:**
1. **base 1패스** 8500건 (768px, ~21min)
2. **반사실 1패스** A집단치환 + B성별치환 (~3500, ~9min)
3. **LLM 중재 디바이어싱** 충돌 시 LLM 재판단 (~1min)
4. **A-family LLM recovery** 잔여 unknown에 강화 프롬프트 재질문 (~2min)
5. **분석 시각화** (차트 4장)

**시간:** ~33min A6000.
**산출:** `outputs/submission_v41_no_rule11.csv` + 분석 차트.
**실행:** 셀0 설치→재시작→순서대로.
"""

# ────────────────────────────────────────────
DATA = r"""# ===== 데이터 + 768px 이미지 =====
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

# ────────────────────────────────────────────
CFBUILD = r"""# ===== 반사실 생성 (A 집단치환 / B 성별치환) =====
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

# ────────────────────────────────────────────
RUN = r"""# ===== 추론 (base + 반사실) =====
import time
T_START=time.time()
t0=time.time(); base=run_single(rows, images); print(f"base(8500) {time.time()-t0:.0f}s = {(time.time()-t0)/60:.1f}분")
t0=time.time(); cf=run_single(cf_rows, cf_imgs); print(f"반사실({len(cf_rows)}) {time.time()-t0:.0f}s")
"""

# ────────────────────────────────────────────
DEBIAS_LLM = r"""# ===== LLM 중재 디바이어싱 =====
cf_pred={cf_map[j]:cf[j] for j in range(len(cf_map))}
cf_unk={cf_map[j]:cf_rows[j]['unk'] for j in range(len(cf_map))}
debiased=list(base)
n_inv=0; conflicts=[]
for k in range(len(rows)):
    if k not in cf_pred: continue
    po,pc=base[k],cf_pred[k]
    if po==pc: n_inv+=1; continue
    uo,uc=rows[k]['unk'],cf_unk[k]
    if po!=uo and pc==uc: continue
    conflicts.append(k)

print(f"디바이어싱: 불변 {n_inv} | 유지(base committed,CF unk) {sum(1 for k in range(len(rows)) if k in cf_pred and base[k]!=rows[k]['unk'] and cf_pred.get(k)==cf_unk.get(k) and base[k]!=cf_pred.get(k))} | 충돌→LLM중재 {len(conflicts)}")

if conflicts:
    t0=time.time()
    med_convs=[]
    for k in conflicts:
        r=rows[k]; po,pc=base[k],cf_pred[k]
        ans_o=r['answers'][po] if 0<=po<len(r['answers']) else '?'
        ans_c=r['answers'][pc] if 0<=pc<len(r['answers']) else '?'
        med_text=(f"Two inference passes on the same question produced different answers:\n"
                  f"- Pass 1 (original context): {ans_o}\n"
                  f"- Pass 2 (group labels swapped): {ans_c}\n\n"
                  f"If answers change when group labels swap, the original may be biased. "
                  f"Decide based ONLY on explicitly stated actions/facts in the context.\n\n"
                  + build_user_text(r['ctx'],r['q'],r['answers']))
        uc=[]
        if images[k] is not None:
            uc.append({"type":"image_url","image_url":{"url":to_url(images[k])}})
        uc.append({"type":"text","text":med_text})
        med_convs.append([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":uc}])
    try:
        med_outs=llm.chat(med_convs,_sp(0.0),use_tqdm=True,chat_template_kwargs={"enable_thinking":False})
    except Exception:
        med_outs=llm.chat(med_convs,_sp(0.0),use_tqdm=True)
    n_changed=0
    for k,o in zip(conflicts,med_outs):
        new_p=parse_answer(o.outputs[0].text,rows[k]['answers'],rows[k]['unk'])
        if new_p!=base[k]: n_changed+=1
        debiased[k]=new_p
    print(f"LLM 중재 완료: {len(conflicts)}건, {n_changed}건 base 대비 변경 | {time.time()-t0:.0f}s")

unk_mask=[debiased[i]==rows[i]['unk'] for i in range(len(rows))]
n_a_unk=sum(1 for i in range(len(rows)) if unk_mask[i] and fam(rows[i]['ctx'])=='A')
print(f"전체 unknown {sum(unk_mask)} | A패밀리 unknown {n_a_unk}")
"""

# ────────────────────────────────────────────
LLM_RECOVERY = r"""# ===== A-family LLM Recovery =====
import re, time
t0=time.time()
ATTR_RE=re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b")
recovery_targets=[]
for k in range(len(rows)):
    if not unk_mask[k]: continue
    if fam(rows[k]['ctx'])!='A': continue
    m=ATTR_RE.search(' '+str(rows[k]['ctx']))
    if not m: continue
    g=m.group(1).strip().lower()
    h=[i for i,x in enumerate(rows[k]['answers']) if g in x.lower() and i!=rows[k]['unk']]
    if len(h)!=1: continue
    recovery_targets.append(k)

print(f"A-family recovery 대상: {len(recovery_targets)}건")

final=list(debiased)
if recovery_targets:
    rec_convs=[]
    for k in recovery_targets:
        r=rows[k]; ctx=r['ctx']
        m=ATTR_RE.search(' '+str(ctx))
        key_sent=""
        if m:
            full=' '+str(ctx)
            dot_pos=full.rfind('.',0,m.start())
            if dot_pos<0: dot_pos=0
            end=full.find('.',m.end())
            if end<0: end=len(full)
            key_sent=full[dot_pos:end+1].strip()

        rec_text=(f"The context DEFINITELY contains a sentence attributing a specific action to one identified person. "
                  f"Abstaining is WRONG here — you MUST commit.\n\n"
                  f"Key sentence: \"{key_sent}\"\n\n"
                  f"The person named in this sentence performed the stated action. "
                  f"Identify them and commit to the matching option.\n\n"
                  + build_user_text(r['ctx'],r['q'],r['answers']))
        uc=[]
        if images[k] is not None:
            uc.append({"type":"image_url","image_url":{"url":to_url(images[k])}})
        uc.append({"type":"text","text":rec_text})
        rec_convs.append([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":uc}])
    try:
        rec_outs=llm.chat(rec_convs,_sp(0.0),use_tqdm=True,chat_template_kwargs={"enable_thinking":False})
    except Exception:
        rec_outs=llm.chat(rec_convs,_sp(0.0),use_tqdm=True)
    n_recovered=0
    for k,o in zip(recovery_targets,rec_outs):
        new_p=parse_answer(o.outputs[0].text,rows[k]['answers'],rows[k]['unk'])
        if new_p!=rows[k]['unk']:
            final[k]=new_p
            n_recovered+=1
    print(f"LLM recovery: {n_recovered}/{len(recovery_targets)}건 committed | {time.time()-t0:.0f}s")
else:
    print("recovery 대상 없음")

print(f"잔여 unknown: {sum(1 for i in range(len(rows)) if final[i]==rows[i]['unk'])}")
from collections import Counter
for f_name in ['A','B','C']:
    before=sum(1 for i in range(len(rows)) if fam(rows[i]['ctx'])==f_name and debiased[i]!=rows[i]['unk'])
    after=sum(1 for i in range(len(rows)) if fam(rows[i]['ctx'])==f_name and final[i]!=rows[i]['unk'])
    total=sum(1 for i in range(len(rows)) if fam(rows[i]['ctx'])==f_name)
    print(f"  {f_name}: commit {before}->{after}/{total} (+{after-before})")
"""

# ────────────────────────────────────────────
FINAL_CMP = r"""# ===== 최종 저장 + 비교 =====
import csv, re, os, time
elapsed=(time.time()-T_START)/60
print(f"총 소요: {elapsed:.1f}분")
VER='v41_no_rule11'
OUT=f'{PROJECT}/outputs/submission_{VER}.csv'
with open(OUT,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['sample_id','label'])
    for i,p in zip(ids,final): w.writerow([i,p])
print("저장:",OUT)

V31=f'{PROJECT}/outputs/submission_v31_grounding_off.csv'
p31={r['sample_id']:int(r['label']) for r in csv.DictReader(open(V31,encoding='utf-8'))} if os.path.exists(V31) else None
V39=f'{PROJECT}/outputs/submission_v39_attr_override.csv'
p39={r['sample_id']:int(r['label']) for r in csv.DictReader(open(V39,encoding='utf-8'))} if os.path.exists(V39) else None
V40=f'{PROJECT}/outputs/submission_v40_llm_debias.csv'
p40={r['sample_id']:int(r['label']) for r in csv.DictReader(open(V40,encoding='utf-8'))} if os.path.exists(V40) else None

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
RE_A2=re.compile(r"^\s*The image shows",re.I)
A_idx=[k for k in range(len(rows)) if RE_A2.search(rows[k]['ctx'])]
def ba(pred_by_id):
    okA=okD=nA=nD=0
    for k in A_idx:
        r=rows[k]; t,g=sg(r['ctx'],r['q'],r['answers'],r['unk'])
        if t=='skip' or g is None: continue
        p=pred_by_id.get(ids[k])
        if p is None: continue
        if t=='ambig': nA+=1; okA+=(p==g)
        else: nD+=1; okD+=(p==g)
    return ((okA/max(1,nA))+(okD/max(1,nD)))/2, okA/max(1,nA), okD/max(1,nD)
final_by_id={ids[k]:final[k] for k in range(len(rows))}
base_by_id={ids[k]:base[k] for k in range(len(rows))}
debiased_by_id={ids[k]:debiased[k] for k in range(len(rows))}
print("\n=== synth_gold A합성BA ===")
b,a,d=ba(base_by_id);     print(f"  base(1패스)    : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
b,a,d=ba(debiased_by_id); print(f"  +LLM디바이어싱 : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
b,a,d=ba(final_by_id);    print(f"  v41(최종)      : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
if p31:
    b,a,d=ba(p31); print(f"  v31(기존최선)  : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
if p40:
    b40,a40,d40=ba(p40); print(f"  v40(Public1등) : BA={b40:.4f} ambig={a40:.3f} disambig={d40:.3f}")
if p39:
    b39,a39,d39=ba(p39); print(f"  v39(ATTR)      : BA={b39:.4f} ambig={a39:.3f} disambig={d39:.3f}")
if p40:
    diff40=sum(1 for k in range(len(rows)) if final[k]!=p40[ids[k]])
    from collections import Counter
    fc40=Counter(fam(rows[k]['ctx']) for k in range(len(rows)) if final[k]!=p40[ids[k]])
    print(f"  v41 vs v40 label diff {diff40} | 패밀리 {dict(fc40)}")
if p31:
    diff31=sum(1 for k in range(len(rows)) if final[k]!=p31[ids[k]])
    fc31=Counter(fam(rows[k]['ctx']) for k in range(len(rows)) if final[k]!=p31[ids[k]])
    print(f"  v41 vs v31 label diff {diff31} | 패밀리 {dict(fc31)}")
"""

# ────────────────────────────────────────────
ANALYSIS = r"""# ===== 분석 시각화 =====
import subprocess, os, random
subprocess.run(['apt-get','install','-y','-qq','fonts-nanum'], capture_output=True)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import numpy as np
from collections import Counter, defaultdict

fp='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(fp): fm.fontManager.addfont(fp); plt.rcParams['font.family']='NanumGothic'
plt.rcParams['axes.unicode_minus']=False
CHART_DIR=f'{PROJECT}/outputs/charts_v41'
os.makedirs(CHART_DIR, exist_ok=True)

ba_base,am_base,di_base = ba(base_by_id)
ba_deb,am_deb,di_deb = ba(debiased_by_id)
ba_fin,am_fin,di_fin = ba(final_by_id)
ba_31,am_31,di_31 = ba(p31) if p31 else (0,0,0)
ba_40,am_40,di_40 = ba(p40) if p40 else (0,0,0)
ba_39,am_39,di_39 = ba(p39) if p39 else (0,0,0)

# ── Chart 1: Pipeline Waterfall ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
stages=['base','+ LLM debias','+ recovery','v41','v40(0.9993)','v31']
for ax, vals, title in [
    (axes[0],[ba_base,ba_deb,ba_fin,ba_fin,ba_40,ba_31],'Balanced Accuracy'),
    (axes[1],[am_base,am_deb,am_fin,am_fin,am_40,am_31],'Ambig Accuracy'),
    (axes[2],[di_base,di_deb,di_fin,di_fin,di_40,di_31],'Disambig Accuracy')]:
    colors=['#78909C','#FF9800','#2196F3','#4CAF50','#9C27B0','#F44336']
    bars=ax.bar(range(len(stages)),vals,color=colors,edgecolor='white',linewidth=2)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2.,v-0.008,f'{v:.4f}',ha='center',va='top',fontsize=8,fontweight='bold',color='white')
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages,fontsize=8,rotation=15)
    ax.set_title(title,fontsize=12,fontweight='bold')
fig.suptitle('v41 Pipeline Waterfall (Rule 11 제거)',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(f'{CHART_DIR}/1_waterfall.png',dpi=150,bbox_inches='tight'); plt.show()

# ── Chart 2: v41 vs v40 vs v31 diff ──
if p40 and p31:
    cats_fam=defaultdict(lambda:{'v40_u2c':0,'v40_c2u':0,'v31_u2c':0,'v31_c2u':0})
    for k in range(len(rows)):
        f=fam(rows[k]['ctx']); u=rows[k]['unk']; p41=final[k]
        if p40 and p41!=p40[ids[k]]:
            if p40[ids[k]]==u and p41!=u: cats_fam[f]['v40_u2c']+=1
            elif p40[ids[k]]!=u and p41==u: cats_fam[f]['v40_c2u']+=1
        if p41!=p31[ids[k]]:
            if p31[ids[k]]==u and p41!=u: cats_fam[f]['v31_u2c']+=1
            elif p31[ids[k]]!=u and p41==u: cats_fam[f]['v31_c2u']+=1
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fams=['A','B','C']; x=np.arange(3); w=0.2
    for ax_i,(ref,label) in enumerate([('v40','v41 vs v40'),('v31','v41 vs v31')]):
        u2c_key=f'{ref}_u2c'; c2u_key=f'{ref}_c2u'
        axes[ax_i].bar(x-w/2,[cats_fam[f][u2c_key] for f in fams],w,label='unk->commit',color='#4CAF50')
        axes[ax_i].bar(x+w/2,[cats_fam[f][c2u_key] for f in fams],w,label='commit->unk',color='#F44336')
        for xi in range(3):
            for di,key in enumerate([u2c_key,c2u_key]):
                v=cats_fam[fams[xi]][key]
                if v>0: axes[ax_i].text(xi+(di-0.5)*w,v+1,str(v),ha='center',fontsize=9,fontweight='bold')
        axes[ax_i].set_xticks(x); axes[ax_i].set_xticklabels(fams,fontsize=11)
        axes[ax_i].set_title(label,fontsize=12,fontweight='bold'); axes[ax_i].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f'{CHART_DIR}/2_diff_analysis.png',dpi=150,bbox_inches='tight'); plt.show()

# ── Chart 3: B-family commit rate by question category ──
import re as _re
Q_CATS={'authority':_re.compile(r'leader|authorit|charge|boss|lead',_re.I),
        'emotional':_re.compile(r'emotional|overwhelm|feeling|comfort|caring|nurtur|support|crying',_re.I),
        'subordinate':_re.compile(r'note|assistant|subordinate|co-host|secretary',_re.I),
        'capability':_re.compile(r'struggle|less capable|weaker|difficult|weight|skill|tech|knowledge',_re.I),
        'calm/rational':_re.compile(r'calm|composed|rational|manage|maintain|distract|focus',_re.I)}
cat_data=defaultdict(lambda:{'n':0,'base_c':0,'v40_c':0,'v31_c':0})
RE_IMG2=_re.compile(r'\b(image|photo|picture)\b',_re.I)
for k in range(len(rows)):
    if fam(rows[k]['ctx'])!='B': continue
    q=rows[k]['q']
    cat='other'
    for cn,rx in Q_CATS.items():
        if rx.search(q): cat=cn; break
    cat_data[cat]['n']+=1
    if base[k]!=rows[k]['unk']: cat_data[cat]['base_c']+=1
    if final[k]!=rows[k]['unk']: cat_data[cat]['v41_c']+=1
    if p31 and p31[ids[k]]!=rows[k]['unk']: cat_data[cat]['v31_c']+=1

fig,ax=plt.subplots(figsize=(12,6))
cats_sorted=sorted(cat_data.items(),key=lambda x:-x[1]['n'])
names=[c for c,_ in cats_sorted]; x=np.arange(len(names)); w=0.25
base_r=[d['base_c']/max(1,d['n']) for _,d in cats_sorted]
v41_r=[d['v41_c']/max(1,d['n']) for _,d in cats_sorted]
v31_r=[d['v31_c']/max(1,d['n']) for _,d in cats_sorted] if p31 else [0]*len(names)
ax.bar(x-w,base_r,w,label='base',color='#78909C',alpha=0.85)
ax.bar(x,v41_r,w,label='v41',color='#2196F3',alpha=0.85)
ax.bar(x+w,v31_r,w,label='v31',color='#4CAF50',alpha=0.85)
for xi in range(len(names)):
    ax.text(xi,max(base_r[xi],v41_r[xi],v31_r[xi])+0.02,f'n={cats_sorted[xi][1]["n"]}',ha='center',fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names,fontsize=9,rotation=20)
ax.set_ylabel('Commit Rate'); ax.set_ylim(0,1.1)
ax.set_title('B패밀리 질문 카테고리별 Commit Rate',fontsize=12,fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig(f'{CHART_DIR}/3_b_category.png',dpi=150,bbox_inches='tight'); plt.show()

# ── Chart 4: Sample images (v41 vs v40 diff, 30건) ──
if p40:
    from PIL import Image as PILImage
    diffs=[]
    for k in range(len(rows)):
        p_40_ref,p_41=p40[ids[k]],final[k]
        if p_40_ref==p_41: continue
        f=fam(rows[k]['ctx'])
        t,g=sg(rows[k]['ctx'],rows[k]['q'],rows[k]['answers'],rows[k]['unk']) if f=='A' else ('skip',None)
        verdict='skip'
        if g is not None and t!='skip':
            if p_41==g and p_40_ref!=g: verdict='v41'
            elif p_41!=g and p_40_ref==g: verdict='v40'
        diffs.append({'k':k,'p40':p_40_ref,'p41':p_41,'verdict':verdict,'fam':f})
    random.seed(42); random.shuffle(diffs)
    cats={'v41':[],'v40':[],'skip':[]}
    for d in diffs: cats[d['verdict']].append(d)
    samples=[]
    for cn in ['v41','v40','skip']:
        samples.extend(cats[cn][:10])
    random.shuffle(samples); samples=samples[:30]
    ncols,nrows=6,5
    fig=plt.figure(figsize=(30,27))
    gs=gridspec.GridSpec(nrows,ncols,hspace=0.55,wspace=0.25)
    for idx,d in enumerate(samples[:nrows*ncols]):
        if idx>=nrows*ncols: break
        ax=fig.add_subplot(gs[idx])
        r=rows[d['k']]
        try:
            from pathlib import Path
            im=PILImage.open(Path(IMG_ROOT)/r['path']); ax.imshow(im)
        except: ax.text(0.5,0.5,'X',ha='center',va='center',transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        bc={'v41':'#4CAF50','v40':'#F44336','skip':'#9E9E9E'}[d['verdict']]
        for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(3)
        tag={'v41':'OK:v41','v40':'OK:v40','skip':'?'}[d['verdict']]
        ax.set_title(f"{ids[d['k']]} [{d['fam']}|{tag}]",fontsize=7,fontweight='bold',color=bc)
        opts='\n'.join(f"{'>' if i==d['p41'] else ' '}{'*' if i==d['p40'] else ' '} [{i}]{a[:30]}" for i,a in enumerate(r['answers']))
        ax.set_xlabel(f">v41 *v40\n{opts}",fontsize=5,fontfamily='monospace')
    fig.suptitle(f'v41 vs v40 불일치 ({len(diffs)}건 중 {min(len(samples),30)}개)\n초록=v41정답 | 빨강=v40정답 | 회색=판정불가',
                 fontsize=13,fontweight='bold')
    plt.savefig(f'{CHART_DIR}/4_sample_images.png',dpi=100,bbox_inches='tight'); plt.show()

print(f"\n차트 저장: {CHART_DIR}/")
"""

# ────────────────────────────────────────────
cells = [
    md(INTRO),
    code(body(0)),          # 셀0: pip install
    code(body1),            # 셀1: model/imports + SYSTEM_PROMPT (수정됨)
    code(body(2)),          # 셀2: helpers (permSC 등)
    code(body(3)),          # 셀3: drive mount
    code(DATA),             # 셀A: 데이터
    code(CFBUILD),          # 셀B: 반사실 생성
    code(RUN),              # 셀C: 추론
    code(DEBIAS_LLM),       # 셀D: LLM 중재 디바이어싱
    code(LLM_RECOVERY),     # 셀E: A-family LLM recovery
    code(FINAL_CMP),        # 셀F: 저장 + 비교
    code(ANALYSIS),         # 셀G: 분석 차트
]

nb["cells"] = cells
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"생성: {OUT} | 셀 {len(cells)}")

# 검증
all_src = "".join(
    "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
    for c in cells
)
checks = [
    ("Rule11 제거됨", "POSED stock photographs" not in all_src),
    ("Rule12(역할매핑) 유지", "Running the meeting" in all_src),
    ("board chair 예시 유지", "Jordan was running the meeting" in all_src),
    ("LLM 중재 디바이어싱", "LLM 중재" in DEBIAS_LLM),
    ("ATTR regex 직접할당 없음", "final[k]=tgt" not in all_src),
    ("LLM recovery 있음", "LLM recovery" in LLM_RECOVERY),
    ("VER=v41", "v41_no_rule11" in FINAL_CMP),
    ("v40 비교 있음", "v40" in FINAL_CMP),
    ("차트 v41", "charts_v41" in ANALYSIS),
    ("T_START 있음", "T_START" in RUN),
    ("API 주석 없음", "API 분석 근거" not in all_src),
    ("총 셀 수=12", len(cells) == 12),
]
print(f"\n검증 {sum(1 for _,v in checks if v)}/{len(checks)}:")
for name, ok in checks:
    print(f"  {'✓' if ok else '✗'} {name}")
