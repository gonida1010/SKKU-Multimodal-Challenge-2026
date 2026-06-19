"""반사실 불변성 실험 — 로컬 타당성 검사 (모델 불필요).
A패밀리 disambig에서 두 집단을 깨끗이 추출해 context+options에서 맞바꿀 수 있나?
가능하면: Colab에서 원본 vs 집단치환본의 '구조적 답 flip rate'를 재서 집단편향을 정량화한다.
"""
import csv, json, re
from collections import Counter

TEST = r"C:\Users\pak10\Downloads\open\test\test.csv"
RE_A = re.compile(r"^\s*The image shows", re.I)
UNK = ["cannot be determined","can't be determined","cannot determine","can't determine",
  "cannot be known","cannot answer","can't answer","cannot be answered","not answerable",
  "not enough info","not enough information","not enough data","insufficient info",
  "insufficient information","not known","unknown","undetermined","cannot say","can't say",
  "cannot tell","can't tell","hard to determine","difficult to determine","impossible to determine",
  "impossible to tell","not specified","unspecified","no information","not provided","unclear","not clear"]
def find_unknown(a):
    f=[any(p in x.lower() for p in UNK) for x in a]
    if sum(f)==1: return f.index(True)
    idx=[i for i,x in enumerate(f) if x]
    return min(idx,key=lambda i:len(a[i])) if idx else None

# 옵션에서 집단 추출: "The Arab person in the yellow shirt" -> "Arab"
OPT_GRP = re.compile(r"^\s*(?:The|An?)\s+(.+?)\s+(?:person|man|woman|individual|people|guy|lady)\b", re.I)
def grp(opt):
    m=OPT_GRP.search(opt); return m.group(1).strip() if m else None

def swap(text, a, b):                      # a<->b 동시 치환 (플레이스홀더)
    t=re.sub(rf'\b{re.escape(a)}\b', '\x00', text, flags=re.I)
    t=re.sub(rf'\b{re.escape(b)}\b', a, t, flags=re.I)
    return t.replace('\x00', b)

rows=[]
for r in csv.DictReader(open(TEST,encoding="utf-8")):
    rows.append({"id":r["sample_id"],"ctx":r["context"],"q":r["question"],"a":json.loads(r["answers"])})

A=[r for r in rows if RE_A.search(r["ctx"])]
ok=[]; reasons=Counter()
for r in A:
    a=r["a"]; unk=find_unknown(a)
    non=[i for i in range(len(a)) if i!=unk]
    if len(non)!=2: reasons["non2"]+=1; continue
    g0,g1=grp(a[non[0]]), grp(a[non[1]])
    if not g0 or not g1: reasons["grp_extract_fail"]+=1; continue
    if g0.lower()==g1.lower(): reasons["same_group"]+=1; continue
    c0=re.search(rf'\b{re.escape(g0)}\b', r["ctx"], re.I)
    c1=re.search(rf'\b{re.escape(g1)}\b', r["ctx"], re.I)
    if not (c0 and c1): reasons["group_not_in_ctx"]+=1; continue
    ok.append((r,g0,g1,unk,non))
reasons["OK_swappable"]=len(ok)

print(f"A패밀리 {len(A)}개 중 반사실 치환 가능: {len(ok)} ({len(ok)/len(A)*100:.1f}%)")
print("불가 사유:", dict(reasons))

print("\n=== 치환 표본 5 (원본 -> 집단 swap) — 구조 보존 확인 ===")
for r,g0,g1,unk,non in ok[:5]:
    sc=swap(r["ctx"], g0, g1)
    so=[swap(o, g0, g1) for o in r["a"]]
    print(f"[{r['id']}]  swap: {g0} <-> {g1}")
    print(f"  ctx 원본: {r['ctx'][:150]}")
    print(f"  ctx 치환: {sc[:150]}")
    print(f"  opt 원본: {r['a']}")
    print(f"  opt 치환: {so}")
    print(f"  Q: {r['q']}  (unknown=idx{unk})")
    print()

# 집단 다양성 (Hidden 일반화 관점: Test가 쓰는 집단 분포)
allg=Counter()
for r,g0,g1,_,_ in ok: allg[g0.lower()]+=1; allg[g1.lower()]+=1
print(f"치환 대상 집단 종류 {len(allg)}개. 상위 15:", [g for g,_ in allg.most_common(15)])
