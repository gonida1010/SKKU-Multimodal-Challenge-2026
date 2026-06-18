"""grounding이 NO로 막은 descriptor_ungrounded 79개 추출 + 이미지 검증 표본 (로컬).

각 케이스: context 귀속 집단(ATTR), commit 시도 옵션(시각수식어 포함), 이미지 경로.
이미지에서 확인할 것: 귀속 집단이 존재하는가 + 시각수식어 불일치가 답과 무관한가.
"""
import csv, json, re, random, sys

TEST_CSV = r"C:\Users\pak10\Downloads\open\test\test.csv"
DIAG = "outputs/v27_descriptor_grounding_diagnostics.csv"
IMG_ROOT = r"C:\Users\pak10\Downloads\open\test\images"

ATTR = re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b")

T = {}
with open(TEST_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        T[r["sample_id"]] = {"ctx": r["context"], "q": r["question"],
                             "answers": json.loads(r["answers"]), "img": r["image_path"]}

ung = []
with open(DIAG, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["reason"] == "descriptor_ungrounded":
            ung.append(r["sample_id"])
print("descriptor_ungrounded 총:", len(ung))

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
random.seed(seed)
samp = random.sample(ung, min(n, len(ung)))

for sid in samp:
    t = T[sid]; ctx = t["ctx"]
    m = ATTR.search(" " + ctx)
    grp = m.group(1).strip() if m else "?"
    opt = [a for a in t["answers"] if grp and grp.lower() in a.lower()]
    num = sid.split("_")[1]
    print(f"=== {sid}  ->  {IMG_ROOT}\\test_img_{num}.jpg")
    print(f"  귀속집단: {grp}  | commit시도옵션: {opt}")
    print(f"  Q: {t['q']}")
    print(f"  보기: {t['answers']}")
    print(f"  CTX: {ctx[:240]}")
