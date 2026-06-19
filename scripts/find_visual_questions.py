"""제3의 가설 핵심 검증 — '순수 시각 변별 문항'이 존재하나?
옵션 2개(비-unknown)가 집단도 이름도 아니고 *시각 속성만*이면 = 텍스트로 불가, 이미지로만 가능.
그런 문항에서 우리가 지금 abstain(unknown)하고 있다면 = 버리는 확정 점수.
GT 불필요. v31 제출본으로 현재 우리 처리(abstain/commit) 확인.
"""
import csv, json, re
from collections import Counter

TEST = r"C:\Users\pak10\Downloads\open\test\test.csv"
V31  = "outputs/submission_v31_grounding_off.csv"

RE_A = re.compile(r"^\s*The image shows", re.I)
RE_IMG = re.compile(r"\b(image|photo|picture)\b", re.I)
def fam(c): return 'A' if RE_A.search(c) else ('B' if RE_IMG.search(c) else 'C')

UNK = ["cannot be determined","can't be determined","cannot determine","can't determine",
  "cannot be known","cannot answer","can't answer","cannot be answered","not answerable",
  "not enough info","not enough information","not enough data","insufficient info",
  "insufficient information","not known","unknown","undetermined","cannot say","can't say",
  "cannot tell","can't tell","hard to determine","difficult to determine",
  "impossible to determine","impossible to tell","not specified","unspecified",
  "no information","not provided","unclear","not clear"]
def find_unknown(a):
    f=[any(p in x.lower() for p in UNK) for x in a]
    if sum(f)==1: return f.index(True)
    idx=[i for i,x in enumerate(f) if x]
    return min(idx,key=lambda i:len(a[i])) if idx else None

# 집단(민족/국적/종교/성별) 어휘 — 텍스트로 결정되는 옵션
GROUP = re.compile(r"\b(arab|hispanic|latino|latina|african|american|european|asian|jewish|jew|"
  r"muslim|christian|hindu|buddhist|catholic|black|white|caucasian|native|middle eastern|"
  r"indian|chinese|japanese|korean|mexican|irish|german|french|italian|russian|nigerian|"
  r"man|woman|male|female|boy|girl|transgender|gay|lesbian|guy|lady|gentleman|"
  r"young|old|elderly|teen|child|senior)\b", re.I)
# 시각 속성 — 이미지로만 결정되는 옵션 (옷/색/위치/자세/소지품)
VIS = re.compile(r"\b(wearing|shirt|jacket|hoodie|dress|hat|cap|beanie|tie|suit|scarf|coat|"
  r"glasses|bag|red|blue|green|yellow|orange|purple|pink|brown|gray|grey|"
  r"left|right|center|centre|middle|front|back|far|standing|sitting|seated|holding|"
  r"the person (in|on|with|wearing|holding|to)|the one (in|on|with|wearing))\b", re.I)
# 이름(고유명사) — 'NAME said' 류
NAME = re.compile(r"\b([A-Z][a-z]+)\b")

def classify(opt):
    g = bool(GROUP.search(opt)); v = bool(VIS.search(opt))
    return g, v

T={}
for r in csv.DictReader(open(TEST,encoding="utf-8")):
    a=json.loads(r["answers"]); T[r["sample_id"]]={"ctx":r["context"],"q":r["question"],"a":a,"f":fam(r["context"])}
P31={r["sample_id"]:int(r["label"]) for r in csv.DictReader(open(V31,encoding="utf-8"))}

pure_visual=[]   # 두 비-unk 옵션이 모두 시각-only (집단X)
for sid,d in T.items():
    a=d["a"]; unk=find_unknown(a)
    if unk is None: continue
    non=[i for i in range(len(a)) if i!=unk]
    if len(non)!=2: continue
    g0,v0=classify(a[non[0]]); g1,v1=classify(a[non[1]])
    # 순수 시각: 둘 다 시각속성 있고, 둘 다 집단어휘 없음
    if v0 and v1 and not g0 and not g1:
        pure_visual.append(sid)

print(f"=== '순수 시각 변별 옵션' 문항 (집단X, 시각속성O 둘다) ===")
print(f"총 {len(pure_visual)}건 / 8500")
fc=Counter(T[s]["f"] for s in pure_visual)
print("패밀리별:", dict(fc))
# 이 중 우리가 현재 abstain(unknown 선택) 하는 비율
ab=sum(1 for s in pure_visual if P31[s]==find_unknown(T[s]["a"]))
print(f"이 중 v31이 abstain(unknown)한 것: {ab}/{len(pure_visual)} = 버리는 후보")
print()
print("=== 표본 12건 (context 앞부분 / Q / 옵션) ===")
for s in pure_visual[:12]:
    d=T[s]; unk=find_unknown(d["a"])
    cur = "ABSTAIN" if P31[s]==unk else f"commit:{d['a'][P31[s]]}"
    print(f"[{s}] ({d['f']}) {cur}")
    print(f"   ctx: {d['ctx'][:120]}")
    print(f"   Q  : {d['q']}")
    print(f"   opt: {d['a']}")
