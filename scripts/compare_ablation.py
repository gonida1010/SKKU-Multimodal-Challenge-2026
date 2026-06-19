"""permSC(3패스) ablation 분해 (로컬):
 - 1패스 base vs 3패스 base 불일치(패밀리별) + synth_gold
 - Public 0.001 갭을 permSC vs recovery로 분해 (C패밀리 label 변화로 근사)
"""
import csv, json, re
from collections import Counter

TEST = r"C:\Users\pak10\Downloads\open\test\test.csv"
P1F  = "outputs/base_preds_1pass.csv"                       # 1패스 base
P3F  = "outputs/base_preds_v27_descriptor_grounding.csv"    # 3패스 permSC base
V31F = "outputs/submission_v31_grounding_off.csv"           # 3패스 base + recovery

RE_A = re.compile(r"^\s*The image shows", re.I)
RE_IMG = re.compile(r"\b(image|photo|picture)\b", re.I)
def fam(c): return 'A' if RE_A.search(c) else ('B' if RE_IMG.search(c) else 'C')
UNK = ["cannot be determined","can't be determined","can not be determined","cannot answer",
  "can't answer","cannot say","not enough info","not enough information","unknown","not known",
  "undetermined","not answerable","not possible to tell","cannot be inferred","insufficient",
  "not determinable","not enough"]
def find_unknown(a):
    f=[any(p in x.lower() for p in UNK) for x in a]
    if sum(f)==1: return f.index(True)
    idx=[i for i,x in enumerate(f) if x]
    return min(idx,key=lambda i:len(a[i])) if idx else None

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

T={}
for r in csv.DictReader(open(TEST,encoding="utf-8")):
    a=json.loads(r["answers"]); T[r["sample_id"]]={"ctx":r["context"],"q":r["question"],"a":a,"unk":find_unknown(a),"f":fam(r["context"])}
def load(p): return {r["sample_id"]:int(r["label"]) for r in csv.DictReader(open(p,encoding="utf-8"))}
P1,P3,V31=load(P1F),load(P3F),load(V31F)
IDS=list(T)

# 1) 1패스 vs 3패스 불일치
diff=[s for s in IDS if P1[s]!=P3[s]]
print(f"=== 1패스 base vs 3패스(permSC) base 불일치: {len(diff)}/{len(IDS)} ({len(diff)/len(IDS)*100:.1f}%) ===")
print("  패밀리별:", dict(Counter(T[s]['f'] for s in diff)))
for nm,P in [("1패스",P1),("3패스",P3)]:
    u=sum(P[s]==T[s]['unk'] for s in IDS)
    print(f"  {nm}: unknown {u} / commit {len(IDS)-u}")

# 2) synth_gold A합성BA
A=[s for s in IDS if T[s]['f']=='A']
def ba(P):
    okD=nD=okA=nA=0
    for s in A:
        t,g=synth_gold(T[s]['ctx'],T[s]['q'],T[s]['a'],T[s]['unk'])
        if t=='skip' or g is None: continue
        if t=='ambig': nA+=1; okA+=(P[s]==g)
        else: nD+=1; okD+=(P[s]==g)
    return ((okA/max(1,nA))+(okD/max(1,nD)))/2, okD/max(1,nD), okA, nA
print("\n=== synth_gold A합성BA (A패밀리, 프록시) ===")
for nm,P in [("1패스 base",P1),("3패스 base",P3),("v31(3패스+recovery)",V31)]:
    b,d,oa,na=ba(P); print(f"  {nm:22s}: A합성BA={b:.4f}  disambig={d:.4f}  ambig정답={oa}/{na}")

# 3) Public(C패밀리) 0.001 갭 분해
C=[s for s in IDS if T[s]['f']=='C']
d_permsc=[s for s in C if P1[s]!=P3[s]]      # permSC가 바꾼 C
d_recov =[s for s in C if P3[s]!=V31[s]]     # recovery가 바꾼 C
d_total =[s for s in C if P1[s]!=V31[s]]     # 1패스base vs 최종 v31
print(f"\n=== Public(C패밀리 {len(C)}개) label 변화 분해 ===")
print(f"  permSC가 바꾼 C   (1패스 vs 3패스): {len(d_permsc)}")
print(f"  recovery가 바꾼 C (3패스 vs v31) : {len(d_recov)}")
print(f"  합쳐서 바뀐 C     (1패스 vs v31) : {len(d_total)}")
print(f"  Public: 1패스 base=0.9975833333 / v31=0.9985833333 (갭 0.001)")
