"""B패밀리 성별편향 감사 — 로컬 타당성 (모델 불필요).
A의 집단축 감사를 성별축으로. B패밀리 옵션이 '남 vs 여' 구조면 성별 대칭치환으로 flip(성별편향) 측정 가능.
"""
import csv, json, re
from collections import Counter

TEST = r"C:\Users\pak10\Downloads\open\test\test.csv"
RE_A = re.compile(r"^\s*The image shows", re.I)
RE_IMG = re.compile(r"\b(image|photo|picture)\b", re.I)
def fam(c): return 'A' if RE_A.search(c) else ('B' if RE_IMG.search(c) else 'C')
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

MALE   = re.compile(r"\b(man|men|male|boy|gentleman|guy|father|son|brother|husband|he|his|him)\b", re.I)
FEMALE = re.compile(r"\b(woman|women|female|girl|lady|mother|daughter|sister|wife|she|her|hers)\b", re.I)
def gender(opt):
    m=bool(MALE.search(opt)); f=bool(FEMALE.search(opt))
    if m and not f: return 'M'
    if f and not m: return 'F'
    if m and f: return 'MF'
    return '-'

# 성별 대칭치환 (명사 위주; 대소문자 보존은 단순화)
PAIRS=[('woman','man'),('women','men'),('female','male'),('girl','boy'),('lady','gentleman'),
       ('mother','father'),('daughter','son'),('sister','brother'),('wife','husband'),
       ('she','he'),('her','his')]  # 주의: her->his 는 소유격만 안전, 목적격 her 모호 -> 측정시 명사 중심 권장
def swap_gender(text):
    def rep(t,a,b):
        t=re.sub(rf'\b{a}\b','\x00',t,flags=re.I)
        t=re.sub(rf'\b{b}\b',a,t,flags=re.I)
        return t.replace('\x00',b)
    for a,b in PAIRS: text=rep(text,a,b)
    return text

rows=[]
for r in csv.DictReader(open(TEST,encoding="utf-8")):
    rows.append({"id":r["sample_id"],"ctx":r["context"],"q":r["question"],"a":json.loads(r["answers"])})
B=[r for r in rows if fam(r["ctx"])=='B']
print(f"B패밀리 총 {len(B)}건")

cat=Counter(); mvf=[]
for r in B:
    a=r["a"]; unk=find_unknown(a)
    non=[i for i in range(len(a)) if i!=unk]
    if len(non)!=2: cat['non2']+=1; continue
    g0,g1=gender(a[non[0]]), gender(a[non[1]])
    key=tuple(sorted([g0,g1]))
    if set([g0,g1])=={'M','F'}: cat['M_vs_F']+=1; mvf.append(r)
    elif g0==g1=='M' or g0==g1=='F': cat['same_gender']+=1
    elif g0=='-' and g1=='-': cat['no_gender(순수시각)']+=1
    else: cat['mixed/etc']+=1
print("B패밀리 옵션 성별 구조:", dict(cat))
print(f"\n=== 성별감사 가능(M vs F) {len(mvf)}건 표본 6 (치환 확인) ===")
for r in mvf[:6]:
    sc=swap_gender(r["ctx"]); so=[swap_gender(o) for o in r["a"]]
    print(f"[{r['id']}] Q: {r['q']}")
    print(f"  opt 원본: {r['a']}")
    print(f"  opt 치환: {so}")
    print(f"  ctx: {r['ctx'][:110]}")
    print()
