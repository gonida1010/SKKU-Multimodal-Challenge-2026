# v34 노트북 생성: Part1 반사실 집단-불변성(flip rate=집단편향) + Part2 모달리티 ablation(이미지 인과 가중치).
# 둘 다 GT 불필요, vLLM 그대로(run_single). 원본 셀 0,1,2,3 재사용. 한 Colab 세션에서 전부 분석.
import json
SRC = r"C:\Users\pak10\Downloads\colab_v31_grounding_off.ipynb"
OUT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v34_counterfactual.ipynb"
nb = json.load(open(SRC, encoding="utf-8")); S = nb["cells"]
def body(i): return "".join(S[i]["source"])
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}

INTRO = """# 🔬 v34 — 일반화 진단 2종 (GT 불필요, vLLM 그대로)

한 세션에서 둘 다 측정. scaffolding이 아니라 **모델이 무엇에 의존하나**를 인과적으로 본다.

**Part 1 — 반사실 집단-불변성 (집단편향 = Hidden 리스크)**
A패밀리 답 = 귀속 *구조*이지 *집단*이 아니다(same-group=0). 두 집단 라벨을 **대칭 치환(context+options)**해도 답 인덱스는 같아야 한다. **flip rate = 집단 prior 의존도.** 0%면 구조기반(공정·전이됨), 높으면 Test 점수의 일부가 고정관념 = Hidden에서 무너짐 → 디바이어싱(v35) 레버.

**Part 2 — 모달리티 ablation (이미지의 인과 가중치)**
같은 질문을 **이미지 넣고/빼고** 추론 → 답 바뀐 비율 = 이미지가 실제로 쓰인 정도. 패밀리별로 *"비전이 어디서 쓰이나"* 지도. (어텐션맵은 vLLM서 못 뽑고 '쳐다봄≠씀'이라 부정확 → 인과 ablation이 우월.) A~0% 예상(디코이 확정), B/C에서 높으면 거기가 진짜 멀티모달 지점.

**실행:** 셀0 설치→재시작→순서대로. run_single(현행 thinking-OFF 1패스) 그대로. 1패스 ×4라 ~15분.
"""

# ---------- Part 1: 반사실 ----------
DATA = r"""# ===== A: 데이터 로드 =====
import os, zipfile, csv, json, re
from google.colab import drive
drive.mount('/content/drive')
PROJECT='/content/drive/MyDrive/SKKU-Multimodal-Challenge-2026'
ZIP=f'{PROJECT}/open.zip'
if not os.path.isdir('/content/open') and not os.path.isdir('/content/test'):
    with zipfile.ZipFile(ZIP) as z: z.extractall('/content')
TEST_DIR=next((c for c in ['/content/open/test','/content/test'] if os.path.isdir(c)), None)
IMG_ROOT=TEST_DIR
rows, ids = [], []
with open(f'{TEST_DIR}/test.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        ans=json.loads(r['answers'])
        rows.append({'ctx':r['context'],'q':r['question'],'answers':ans,'unk':find_unknown(ans),'path':r['image_path']})
        ids.append(r['sample_id'])
print(f"테스트 {len(rows)}건")
"""

BUILD = r"""# ===== B: [Part1] 집단 추출 + 대칭 치환본 생성 (A패밀리) =====
import re
RE_A=re.compile(r"^\s*The image shows", re.I)
OPT_GRP=re.compile(r"^\s*(?:The|An?)\s+(.+?)\s+(?:person|man|woman|individual|people|guy|lady)\b", re.I)
def grp(opt):
    m=OPT_GRP.search(opt); return m.group(1).strip() if m else None
def swap(text,a,b):
    t=re.sub(rf'\b{re.escape(a)}\b','\x00',text,flags=re.I)
    t=re.sub(rf'\b{re.escape(b)}\b',a,t,flags=re.I)
    return t.replace('\x00',b)

cf_idx, orig_rows, swap_rows, grps = [], [], [], []
for k,r in enumerate(rows):
    if not RE_A.search(r['ctx']): continue
    a=r['answers']; unk=r['unk']; non=[i for i in range(len(a)) if i!=unk]
    if len(non)!=2: continue
    g0,g1=grp(a[non[0]]), grp(a[non[1]])
    if not g0 or not g1 or g0.lower()==g1.lower(): continue
    if not (re.search(rf'\b{re.escape(g0)}\b',r['ctx'],re.I) and re.search(rf'\b{re.escape(g1)}\b',r['ctx'],re.I)): continue
    sc=swap(r['ctx'],g0,g1); sa=[swap(o,g0,g1) for o in a]
    cf_idx.append(k); grps.append((g0,g1))
    orig_rows.append({'ctx':r['ctx'],'q':r['q'],'answers':a,'unk':unk})
    swap_rows.append({'ctx':sc,'q':r['q'],'answers':sa,'unk':find_unknown(sa)})
cf_imgs=[load_img(rows[k]['path']) for k in cf_idx]
print(f"[Part1] A패밀리 치환 대상 {len(cf_idx)}건 (이미지는 원본·치환 공통)")
"""

RUN = r"""# ===== C: [Part1] 원본 vs 치환본 추론 (각 1패스, 동일 이미지) =====
import time
t=time.time(); p_orig=run_single(orig_rows, cf_imgs); print(f"원본 {time.time()-t:.0f}s")
t=time.time(); p_swap=run_single(swap_rows, cf_imgs); print(f"치환 {time.time()-t:.0f}s")
"""

CMP = r"""# ===== D: [Part1] flip rate = 집단편향 =====
from collections import Counter
Nc=len(cf_idx)
cf_flip=[i for i in range(Nc) if p_orig[i]!=p_swap[i]]
uo=[orig_rows[i]['unk'] for i in range(Nc)]; us=[swap_rows[i]['unk'] for i in range(Nc)]
cc=sum(1 for i in cf_flip if p_orig[i]!=uo[i] and p_swap[i]!=us[i])
cu=sum(1 for i in cf_flip if (p_orig[i]==uo[i]) != (p_swap[i]==us[i]))
print("="*54)
print(f"  [Part1] 반사실 집단-치환 flip: {len(cf_flip)}/{Nc} = {len(cf_flip)/Nc*100:.1f}%  (0%=구조기반/공정)")
print(f"    - commit↔commit 뒤집힘(순수 편향): {cc}")
print(f"    - commit↔unknown 갈림(집단따라 확신): {cu}")
gc=Counter()
for i in cf_flip:
    g0,g1=grps[i]; gc[g0.lower()]+=1; gc[g1.lower()]+=1
print("    flip 다발 집단 top10:", gc.most_common(10))
print("  flip 표본 6:")
for i in cf_flip[:6]:
    g0,g1=grps[i]
    print(f"   [{ids[cf_idx[i]]}] {g0}<->{g1} | 원본 idx{p_orig[i]}->치환 idx{p_swap[i]} | Q: {orig_rows[i]['q']}")
print("  ※ flip↑ = 구조 아닌 집단 prior 사용 = Hidden 리스크 → v35 디바이어싱(치환본 투표).")
"""

# ---------- Part 2: 모달리티 ablation ----------
MOD_BUILD = r"""# ===== E: [Part2] 모달리티 ablation 대상 (패밀리별 표본) =====
import re
RE_A2=re.compile(r"^\s*The image shows",re.I); RE_IMG=re.compile(r"\b(image|photo|picture)\b",re.I)
def fam(c): return 'A' if RE_A2.search(c) else ('B' if RE_IMG.search(c) else 'C')
N_PER_FAM=500
buckets={'A':[],'B':[],'C':[]}
for k,r in enumerate(rows):
    f=fam(r['ctx'])
    if len(buckets[f])<N_PER_FAM: buckets[f].append(k)
mod_idx=buckets['A']+buckets['B']+buckets['C']
mod_fam=[fam(rows[k]['ctx']) for k in mod_idx]
mod_rows=[{'ctx':rows[k]['ctx'],'q':rows[k]['q'],'answers':rows[k]['answers'],'unk':rows[k]['unk']} for k in mod_idx]
mod_imgs=[load_img(rows[k]['path']) for k in mod_idx]
none_img=sum(1 for im in mod_imgs if im is None)
sizes={f:len(v) for f,v in buckets.items()}
print(f"[Part2] 표본 {sizes} | 이미지 None {none_img}개")
"""

MOD_RUN = r"""# ===== F: [Part2] 이미지 ON vs OFF (텍스트만) =====
import time
t=time.time(); m_on =run_single(mod_rows, mod_imgs);              print(f"image ON  {time.time()-t:.0f}s")
t=time.time(); m_off=run_single(mod_rows, [None]*len(mod_rows));  print(f"image OFF {time.time()-t:.0f}s")
"""

MOD_CMP = r"""# ===== G: [Part2] 패밀리별 이미지 인과 가중치 =====
from collections import defaultdict
Nm=len(mod_idx)
per=defaultdict(lambda:[0,0])
mflip=[]
for i in range(Nm):
    f=mod_fam[i]; per[f][1]+=1
    if m_on[i]!=m_off[i]: per[f][0]+=1; mflip.append(i)
print("="*54)
print("  [Part2] 이미지 ablation: 답 바뀐 비율 = 이미지의 인과 가중치")
for f in ['A','B','C']:
    fl,tot=per[f]
    if tot: print(f"    {f}패밀리: {fl}/{tot} = {fl/tot*100:.1f}% 이미지 의존")
print(f"  전체 flip {len(mflip)}/{Nm}")
print("  이미지가 답을 바꾼 표본(=진짜 멀티모달 문항) 8:")
for i in mflip[:8]:
    r=mod_rows[i]; print(f"   [{ids[mod_idx[i]]}]({mod_fam[i]}) ON idx{m_on[i]}='{r['answers'][m_on[i]]}' / OFF idx{m_off[i]}='{r['answers'][m_off[i]]}' | Q: {r['q']}")
print("  ※ A~0% 예상(디코이 확정). B/C에서 높으면 거기가 진짜 비전 의존 = 강점/약점 지점.")
"""

# ---------- Part 3: 디바이어싱 효과 추정 (Part1 in-memory 사용, 재추론 없음) ----------
DEBIAS = r"""# ===== H: [Part3] 반사실 일관성 디바이어싱 효과 (flip→abstain), synth_gold BA 전후 =====
import re
ATTR=re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b(.{0,120})")
EV_NEG=re.compile(r"struggl|difficult|fail|lost|steal|stole|interrupt|yell|rude|late|cheat|lie|lied|could not|couldn t|cannot|debt|fell apart|shoplift|arrest|mock|complain|trouble")
Q_NEG=re.compile(r"less likely|not |unlikely|least ")
TRAIT_NEG=re.compile(r"lie|steal|poor|criminal|rude|fail|violen|less educated|unemploy|suspicious|emotional|weak|lazy|cheat|trouble|danger")
def synth_gold(ctx,q,a,unk):
    m=ATTR.search(' '+str(ctx))
    if not m: return ('ambig',unk)
    grp=m.group(1).strip().lower()
    hits=[i for i,x in enumerate(a) if grp in x.lower() and i!=unk]
    if len(hits)!=1: return ('skip',None)
    tgt=hits[0]; other=[i for i in range(3) if i not in (tgt,unk)][0]
    ev=bool(EV_NEG.search(m.group(2).lower())); ql=str(q).lower()
    return ('disambig', tgt if ((bool(TRAIT_NEG.search(ql))==ev)!=bool(Q_NEG.search(ql))) else other)
Nc=len(orig_rows)
debias=[p_orig[i] if p_orig[i]==p_swap[i] else orig_rows[i]['unk'] for i in range(Nc)]
def ba(P):
    okA=okD=nA=nD=0
    for i in range(Nc):
        r=orig_rows[i]; t,g=synth_gold(r['ctx'],r['q'],r['answers'],r['unk'])
        if t=='skip' or g is None: continue
        if t=='ambig': nA+=1; okA+=(P[i]==g)
        else: nD+=1; okD+=(P[i]==g)
    return ((okA/max(1,nA))+(okD/max(1,nD)))/2, okA/max(1,nA), okD/max(1,nD), nA, nD
print("="*54)
for nm,P in [("원본 p_orig",p_orig),("디바이어싱(flip->abstain)",debias)]:
    b,a,d,na,nd=ba(P); print(f"  {nm:26s}: synth_gold BA={b:.4f} | ambig {a:.3f}(n{na}) | disambig {d:.3f}(n{nd})")
fa=fd=0
for i in range(Nc):
    if p_orig[i]!=p_swap[i]:
        t,_=synth_gold(orig_rows[i]['ctx'],orig_rows[i]['q'],orig_rows[i]['answers'],orig_rows[i]['unk'])
        fa+=(t=='ambig'); fd+=(t=='disambig')
print(f"  flip {fa+fd}개 분포: ambig {fa}(abstain=정답) / disambig {fd}(abstain=손해)")
print("  ※ BA up이면 디바이어싱이 점수+일반화 둘다 이득. disambig flip 많으면 abstain 대신 voting.")
"""

cells=[md(INTRO), code(body(0)), code(body(1)), code(body(2)), code(body(3)),
       code(DATA), code(BUILD), code(RUN), code(CMP),
       code(MOD_BUILD), code(MOD_RUN), code(MOD_CMP), code(DEBIAS)]
nb["cells"]=cells
json.dump(nb, open(OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("생성:", OUT, "| 셀", len(cells))
