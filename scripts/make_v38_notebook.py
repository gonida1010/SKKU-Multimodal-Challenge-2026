# v38 = v36 개선: A패밀리 only recovery(70분 fit) + 내장 분석 시각화.
# 차트 분석에서 발견: B패밀리 74% unknown(미개척), A commit rate v31≈v36.
# recovery를 A에만 집중 → synth_gold 동일 + 70분 충족.
import json

SRC = r"C:\Users\pak10\Downloads\colab_v31_grounding_off.ipynb"
OUT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v38_final.ipynb"

nb = json.load(open(SRC, encoding="utf-8")); S = nb["cells"]
def body(i): return "".join(S[i]["source"])
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}

cell8 = body(8)
_split = cell8.index('\nimport csv as _csv')
RECOVERY_FUNCS = cell8[:_split].rstrip() + '\n'

# ────────────────────────────────────────────
INTRO = """# v38 — 반사실 디바이어싱 + A패밀리 Recovery (70분 제출본)

**파이프라인:**
1. **base 1패스** run_single 8500 (768px, ~21min)
2. **반사실 1패스** A집단치환 + B성별치환 (~3500, ~9min)
3. **디바이어싱 규칙** (invariant=유지 | commit-recovery=회수 | commit-commit=abstain)
4. **A패밀리 Recovery** 잔여 A unknown → 1024px witness + permSC 3패스 + 다단 게이트 (~17min)
5. **분석 시각화** (자동 차트 생성)

**시간:** ~48min A6000. 70분 충족.
**산출:** `outputs/submission_v38_final.csv` + 분석 차트 4장.
**실행:** 셀0 설치→재시작→순서대로.
"""

# ────────────────────────────────────────────
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

# ────────────────────────────────────────────
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

# ────────────────────────────────────────────
RUN = r"""# ===== C: 추론 (base 1패스 + 반사실 1패스) =====
import time
T_START=time.time()
t0=time.time(); base=run_single(rows, images); print(f"base(8500) {time.time()-t0:.0f}s = {(time.time()-t0)/60:.1f}분")
t0=time.time(); cf=run_single(cf_rows, cf_imgs); print(f"반사실({len(cf_rows)}) {time.time()-t0:.0f}s")
"""

# ────────────────────────────────────────────
DEBIAS = r"""# ===== D: 반사실 디바이어싱 규칙 =====
cf_pred={cf_map[j]:cf[j] for j in range(len(cf_map))}
cf_unk={cf_map[j]:cf_rows[j]['unk'] for j in range(len(cf_map))}
debiased=list(base)
n_rec=n_abs=n_inv=0
debias_abstained=set()
for k in range(len(rows)):
    if k not in cf_pred: continue
    po,pc,uo,uc=base[k],cf_pred[k],rows[k]['unk'],cf_unk[k]
    if po==pc: n_inv+=1; continue
    if po!=uo and pc==uc: continue
    if po==uo and pc!=uc: debiased[k]=pc; n_rec+=1
    elif po!=uo and pc!=uc and po!=pc: debiased[k]=uo; n_abs+=1; debias_abstained.add(k)
print(f"디바이어싱: 불변 {n_inv} | commit-recovery {n_rec} | abstain {n_abs} | 변경 {n_rec+n_abs}")
unk_mask=[debiased[i]==rows[i]['unk'] for i in range(len(rows))]
n_a_unk=sum(1 for i in range(len(rows)) if unk_mask[i] and fam(rows[i]['ctx'])=='A')
print(f"전체 unknown {sum(unk_mask)} | A패밀리 unknown {n_a_unk} (← recovery 대상)")
"""

# ────────────────────────────────────────────
RECOVERY_EXEC = r"""# ===== F: A패밀리 Recovery (1024px witness + permSC 3패스 + 다단 게이트) =====
import time
from tqdm.auto import tqdm
VER='v38_final'
unk_idx_list=[i for i in range(len(rows)) if unk_mask[i] and fam(rows[i]['ctx'])=='A']
print(f"A패밀리 recovery 대상: {len(unk_idx_list)}건")
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

# ────────────────────────────────────────────
FINAL_CMP = r"""# ===== G: 최종 저장 + v31 비교 =====
import csv, re, os, time
final=list(debiased)
for i,p in flips.items(): final[i]=p
n_total=n_rec+n_abs+len(flips)
elapsed=(time.time()-T_START)/60
print(f"총 변경: 디바이어싱 {n_rec+n_abs}(rec={n_rec},abs={n_abs}) + recovery {len(flips)} = {n_total}")
print(f"총 소요: {elapsed:.1f}분")
OUT=f'{PROJECT}/outputs/submission_v38_final.csv'
with open(OUT,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['sample_id','label'])
    for i,p in zip(ids,final): w.writerow([i,p])
print("저장:",OUT)

# --- synth_gold ---
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
b,a,d=ba(debiased_by_id); print(f"  디바이어싱only : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
b,a,d=ba(final_by_id);    print(f"  v38(최종)      : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
if p31:
    b,a,d=ba(p31); print(f"  v31(기존최선)  : BA={b:.4f} ambig={a:.3f} disambig={d:.3f}")
    diff=sum(1 for k in range(len(rows)) if final[k]!=p31[ids[k]])
    from collections import Counter
    fc=Counter(fam(rows[k]['ctx']) for k in range(len(rows)) if final[k]!=p31[ids[k]])
    print(f"  v38 vs v31 label diff {diff} | 패밀리 {dict(fc)}")
"""

# ────────────────────────────────────────────
ANALYSIS = r"""# ===== H: 분석 시각화 (차트 4장 자동 생성) =====
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
CHART_DIR=f'{PROJECT}/outputs/charts_v38'
os.makedirs(CHART_DIR, exist_ok=True)

# ── Chart 1: Pipeline Waterfall ──
ba_base,am_base,di_base = ba(base_by_id)
ba_deb,am_deb,di_deb = ba(debiased_by_id)
ba_fin,am_fin,di_fin = ba(final_by_id)
ba_31,am_31,di_31 = ba(p31) if p31 else (0,0,0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
stages=['base','+ debias','+ recovery','v38','v31']
for ax, vals, title in [
    (axes[0],[ba_base,ba_deb,ba_fin,ba_fin,ba_31],'Balanced Accuracy'),
    (axes[1],[am_base,am_deb,am_fin,am_fin,am_31],'Ambig Accuracy'),
    (axes[2],[di_base,di_deb,di_fin,di_fin,di_31],'Disambig Accuracy')]:
    colors=['#78909C','#FF9800','#2196F3','#4CAF50','#F44336']
    bars=ax.bar(range(len(stages)),vals,color=colors,edgecolor='white',linewidth=2)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2.,v-0.008,f'{v:.4f}',ha='center',va='top',fontsize=9,fontweight='bold',color='white')
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages,fontsize=9)
    ax.set_title(title,fontsize=12,fontweight='bold')
fig.suptitle('v38 Pipeline Waterfall',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(f'{CHART_DIR}/1_waterfall.png',dpi=150,bbox_inches='tight'); plt.show()

# ── Chart 2: v38 vs v31 Diff Analysis ──
if p31:
    cats_fam=defaultdict(lambda:{'u2c':0,'c2u':0,'c2c':0})
    sg_verdict={'v38_win':0,'v31_win':0,'skip':0}
    for k in range(len(rows)):
        p_31,p_38=p31[ids[k]],final[k]; u=rows[k]['unk']
        if p_31==p_38: continue
        f=fam(rows[k]['ctx'])
        if p_31==u and p_38!=u: cats_fam[f]['u2c']+=1
        elif p_31!=u and p_38==u: cats_fam[f]['c2u']+=1
        else: cats_fam[f]['c2c']+=1
        if f=='A':
            t,g=sg(rows[k]['ctx'],rows[k]['q'],rows[k]['answers'],u)
            if t=='skip' or g is None: sg_verdict['skip']+=1
            elif p_38==g and p_31!=g: sg_verdict['v38_win']+=1
            elif p_38!=g and p_31==g: sg_verdict['v31_win']+=1
            else: sg_verdict['skip']+=1
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    fams=['A','B','C']; x=np.arange(3); w=0.25
    axes[0].bar(x-w,[cats_fam[f]['u2c'] for f in fams],w,label='unk→commit',color='#4CAF50')
    axes[0].bar(x,[cats_fam[f]['c2u'] for f in fams],w,label='commit→unk',color='#F44336')
    axes[0].bar(x+w,[cats_fam[f]['c2c'] for f in fams],w,label='commit→commit',color='#FF9800')
    for xi in range(3):
        for di,key in enumerate(['u2c','c2u','c2c']):
            v=cats_fam[fams[xi]][key]
            if v>0: axes[0].text(xi+(di-1)*w,v+1,str(v),ha='center',fontsize=8,fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(fams,fontsize=11)
    axes[0].set_title('v38 vs v31 Label Diff',fontsize=12,fontweight='bold'); axes[0].legend(fontsize=8)
    sv=sg_verdict
    axes[1].pie([sv['v38_win'],sv['v31_win'],sv['skip']],
                labels=[f"v38 win ({sv['v38_win']})",f"v31 win ({sv['v31_win']})",f"skip ({sv['skip']})"],
                colors=['#4CAF50','#F44336','#9E9E9E'],autopct='%1.0f%%',startangle=90)
    axes[1].set_title('A패밀리 synth_gold 판정',fontsize=12,fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{CHART_DIR}/2_diff_analysis.png',dpi=150,bbox_inches='tight'); plt.show()

# ── Chart 3: Identity Group Commit Rate (top 25) ──
if p31:
    grp_data=defaultdict(lambda:{'n':0,'v31c':0,'v38c':0})
    for k in A_idx:
        r=rows[k]; non=[i for i in range(3) if i!=r['unk']]
        if len(non)!=2: continue
        seen=set()
        for ni in non:
            m=OPT_GRP.search(r['answers'][ni])
            if m:
                g=m.group(1).strip()
                if g.lower() not in seen:
                    seen.add(g.lower())
                    grp_data[g]['n']+=1
                    if p31[ids[k]]!=r['unk']: grp_data[g]['v31c']+=1
                    if final[k]!=r['unk']: grp_data[g]['v38c']+=1
    top=sorted(grp_data.items(),key=lambda x:-x[1]['n'])[:25]
    fig,ax=plt.subplots(figsize=(16,7))
    names=[g for g,_ in top]; x=np.arange(len(names)); w=0.35
    v31r=[d['v31c']/max(1,d['n']) for _,d in top]
    v38r=[d['v38c']/max(1,d['n']) for _,d in top]
    ns=[d['n'] for _,d in top]
    ax.bar(x-w/2,v31r,w,label='v31',color='#4CAF50',alpha=0.85)
    ax.bar(x+w/2,v38r,w,label='v38',color='#2196F3',alpha=0.85)
    for xi in range(len(names)):
        diff=v38r[xi]-v31r[xi]
        c='#E91E63' if diff>0.01 else('#FF9800' if diff<-0.01 else'#9E9E9E')
        ax.text(xi,max(v31r[xi],v38r[xi])+0.02,f'n={ns[xi]}',ha='center',fontsize=6,color=c)
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=50,ha='right',fontsize=7)
    ax.set_ylabel('Commit Rate'); ax.set_ylim(0,1.1)
    ax.set_title('A패밀리 정체성 집단별 Commit Rate (v31 vs v38)',fontsize=12,fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(f'{CHART_DIR}/3_group_commit.png',dpi=150,bbox_inches='tight'); plt.show()

# ── Chart 4: Sample Images (A패밀리 v38≠v31, 30건) ──
if p31:
    from PIL import Image as PILImage
    diffs=[]
    for k in A_idx:
        p_31,p_38=p31[ids[k]],final[k]
        if p_31==p_38: continue
        t,g=sg(rows[k]['ctx'],rows[k]['q'],rows[k]['answers'],rows[k]['unk'])
        verdict='v38' if(g is not None and p_38==g and t!='skip') else('v31' if(g is not None and p_31==g and t!='skip') else 'skip')
        diffs.append({'k':k,'p31':p_31,'p38':p_38,'verdict':verdict})
    random.seed(42); random.shuffle(diffs)
    cats={'v38':[],'v31':[],'skip':[]}
    for d in diffs: cats[d['verdict']].append(d)
    samples=[]
    for cn in ['v38','v31','skip']:
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
        bc={'v38':'#4CAF50','v31':'#F44336','skip':'#9E9E9E'}[d['verdict']]
        for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(3)
        tag={'v38':'OK:v38','v31':'OK:v31','skip':'?'}[d['verdict']]
        ax.set_title(f"{ids[d['k']]} [{tag}]",fontsize=7,fontweight='bold',color=bc)
        opts='\n'.join(f"{'>' if i==d['p38'] else ' '}{'*' if i==d['p31'] else ' '} [{i}]{a[:30]}" for i,a in enumerate(r['answers']))
        ax.set_xlabel(f">v38 *v31\n{opts}",fontsize=5,fontfamily='monospace')
    fig.suptitle(f'A패밀리 v38 vs v31 불일치 ({len(diffs)}건 중 {min(len(samples),30)}개)\n초록=v38정답 | 빨강=v31정답 | 회색=판정불가',
                 fontsize=13,fontweight='bold')
    plt.savefig(f'{CHART_DIR}/4_sample_images.png',dpi=100,bbox_inches='tight'); plt.show()

print(f"\n차트 저장: {CHART_DIR}/")
"""

# ────────────────────────────────────────────
cells = [
    md(INTRO),
    code(body(0)),         # 셀0: pip install
    code(body(1)),         # 셀1: model/imports
    code(body(2)),         # 셀2: helpers
    code(body(3)),         # 셀3: drive mount
    code(DATA),            # 셀A: 데이터 + 768 이미지
    code(CFBUILD),         # 셀B: 반사실 생성
    code(RUN),             # 셀C: 1패스 base + CF
    code(DEBIAS),          # 셀D: 디바이어싱 규칙 → unk_mask
    code(RECOVERY_FUNCS),  # 셀E: recovery 함수 정의
    code(RECOVERY_EXEC),   # 셀F: A패밀리 recovery 실행
    code(FINAL_CMP),       # 셀G: 저장 + synth_gold 비교
    code(ANALYSIS),        # 셀H: 분석 시각화 4장
]

nb["cells"] = cells
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("생성:", OUT, "| 셀", len(cells))
