"""A패밀리에서 non-unknown 보기 2개가 같은 집단인지 다른 집단인지 측정 (로컬, LLM 불필요).

핵심 질문: descriptor grounding(시각수식어 검증)이 답 결정에 의미가 있는가?
- 두 non-unk 보기가 '서로 다른 집단' -> 답은 집단명으로 정해짐 -> 시각수식어 무관 -> grounding 무의미/유해
- 두 non-unk 보기가 '같은 집단' -> 시각수식어가 답을 가름 -> grounding 의미 있음
"""
import csv, json, re

TEST_CSV = r"C:\Users\pak10\Downloads\open\test\test.csv"
FLIP_CSV = "outputs/analysis/v27_flip_A_visdesc.csv"

RE_A = re.compile(r"^\s*The image shows", re.I)
# context: "including X people and Y people" (X,Y 집단)
INCL = re.compile(r"including\s+(.+?)\s+people\s+and\s+(.+?)\s+people", re.I)
# 보기 집단명: "The <Group> person/man/woman"
OPTG = re.compile(r"[Tt]he\s+([A-Za-z][A-Za-z\-]*(?:\s+[A-Z][A-Za-z\-]*)?)\s+(?:person|man|woman)\b")
UNK_HINTS = ["cannot be determined", "not enough info", "unknown", "not known",
             "can't be determined", "undetermined", "no information", "not possible to tell",
             "cannot be inferred", "not answerable", "insufficient", "can not be determined",
             "not enough information"]

def is_unk(a):
    al = a.lower()
    return any(h in al for h in UNK_HINTS)

def opt_group(opt):
    m = OPTG.search(opt)
    return m.group(1).strip().lower() if m else None

def classify(answers):
    """return ('diff'|'same'|'unsure', groups_of_nonunk)"""
    nonunk = [a for a in answers if not is_unk(a)]
    if len(nonunk) != 2:
        return "unsure_unk", None
    g = [opt_group(a) for a in nonunk]
    if None in g:
        return "unsure_grp", g
    return ("same" if g[0] == g[1] else "diff"), g

# 전체 A패밀리
allA = []
flip_ids = set()
with open(FLIP_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        flip_ids.add(r["sample_id"])

from collections import Counter
cnt_all = Counter()
cnt_flip = Counter()
flip_same_examples = []
with open(TEST_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if not RE_A.search(r["context"]):
            continue
        answers = json.loads(r["answers"])
        cls, g = classify(answers)
        cnt_all[cls] += 1
        if r["sample_id"] in flip_ids:
            cnt_flip[cls] += 1
            if cls == "same":
                flip_same_examples.append((r["sample_id"], answers))

print("=== 전체 A패밀리 (non-unk 보기 2개 집단 관계) ===")
for k, v in cnt_all.most_common():
    print(f"  {k:12s}: {v}")
print("  TOTAL A:", sum(cnt_all.values()))

print("\n=== 135 visdesc-A FLIP (grounding이 YES로 통과시킨 것) ===")
for k, v in cnt_flip.most_common():
    print(f"  {k:12s}: {v}")
print("  TOTAL FLIP:", sum(cnt_flip.values()))

print("\n=== FLIP 중 'same 집단' 예시 (grounding이 진짜 필요한 케이스) ===")
for sid, ans in flip_same_examples[:15]:
    print(f"  {sid}: {ans}")
