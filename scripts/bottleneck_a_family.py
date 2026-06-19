"""v31 이후 A패밀리 recall 병목 진단 (로컬, LLM 불필요).
질문: synth_gold가 disambig라는데 v31이 아직 unknown으로 둔 A패밀리는?
      그걸 막고 있는 게이트(reason)는 무엇인가? = v32 후보.
또한 v31이 commit했지만 synth_gold target과 '반대'로 commit한 위험 케이스도 집계.
"""
import csv, json, re
from collections import Counter

TEST = r"C:\Users\pak10\Downloads\open\test\test.csv"
V31 = "outputs/submission_v31_grounding_off.csv"
DIAG = "outputs/v31_grounding_off_diagnostics.csv"

RE_A = re.compile(r"^\s*The image shows", re.I)
UNK = ["cannot be determined", "can't be determined", "can not be determined", "cannot answer",
       "can't answer", "cannot say", "not enough info", "not enough information", "unknown",
       "not known", "undetermined", "not answerable", "not possible to tell", "cannot be inferred",
       "insufficient", "not determinable", "not enough"]
def find_unknown(ans):
    f = [any(p in a.lower() for p in UNK) for a in ans]
    if sum(f) == 1: return f.index(True)
    idx = [i for i, x in enumerate(f) if x]
    return min(idx, key=lambda i: len(ans[i])) if idx else None

ATTR = re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b(.{0,120})")
EV_NEG = re.compile(r"struggl|difficult|fail|lost|steal|stole|interrupt|yell|rude|late|cheat|lie|lied|"
                    r"could not|couldn t|cannot|debt|fell apart|shoplift|arrest|mock|complain|trouble")
Q_NEG = re.compile(r"less likely|not |unlikely|least ")
TRAIT_NEG = re.compile(r"lie|steal|poor|criminal|rude|fail|violen|less educated|unemploy|"
                       r"suspicious|emotional|weak|lazy|cheat|trouble|danger")
def synth_gold(ctx, q, ans, unk):
    m = ATTR.search(' ' + str(ctx))
    if not m: return ('ambig', unk)
    grp = m.group(1).strip().lower()
    hits = [i for i, a in enumerate(ans) if grp in a.lower() and i != unk]
    if len(hits) != 1: return ('skip', None)
    tgt = hits[0]; other = [i for i in range(3) if i not in (tgt, unk)][0]
    ev_neg = bool(EV_NEG.search(m.group(2).lower())); ql = str(q).lower()
    trait_neg = bool(TRAIT_NEG.search(ql)); ask_flip = bool(Q_NEG.search(ql))
    same_dir = (trait_neg == ev_neg)
    return ('disambig', tgt if (same_dir != ask_flip) else other)

T = {}
for r in csv.DictReader(open(TEST, encoding="utf-8")):
    T[r["sample_id"]] = {"ctx": r["context"], "q": r["question"], "ans": json.loads(r["answers"])}
P31 = {r["sample_id"]: int(r["label"]) for r in csv.DictReader(open(V31, encoding="utf-8"))}
REASON = {r["sample_id"]: r["reason"] for r in csv.DictReader(open(DIAG, encoding="utf-8"))}

A = [s for s in T if RE_A.search(T[s]["ctx"])]
sg_type = Counter()
commit_correct = []   # synth disambig, v31 == target
commit_wrong = []     # synth disambig, v31 == other (반대로 commit = 위험)
not_committed = []    # synth disambig, v31 == unk (회수 여지)
for s in A:
    ctx, q, ans = T[s]["ctx"], T[s]["q"], T[s]["ans"]
    unk = find_unknown(ans)
    t, g = synth_gold(ctx, q, ans, unk)
    sg_type[t] += 1
    if t != 'disambig': continue
    p = P31[s]
    if p == unk:        not_committed.append(s)
    elif p == g:        commit_correct.append(s)
    else:               commit_wrong.append(s)

print(f"A패밀리 {len(A)}개 | synth_gold 분류: {dict(sg_type)}")
print(f"\n== synth_gold=disambig 중 v31 처리 상태 ==")
print(f"  commit 정답방향(=target): {len(commit_correct)}")
print(f"  commit 반대방향(other)  : {len(commit_wrong)}   ← 위험(틀리게 commit)")
print(f"  미commit(아직 unknown)  : {len(not_committed)}  ← 회수 여지(v32 후보)")

print(f"\n== 미commit {len(not_committed)}개의 막은 게이트(reason) ==")
bn = Counter(REASON.get(s, "not_in_diag") for s in not_committed)
for r, c in bn.most_common():
    print(f"  {r:24s}: {c}")

print(f"\n== 반대방향 commit {len(commit_wrong)}개 (샘플) ==")
for s in commit_wrong[:15]:
    print(f"  {s}  reason={REASON.get(s,'-')}  pred={P31[s]}")
