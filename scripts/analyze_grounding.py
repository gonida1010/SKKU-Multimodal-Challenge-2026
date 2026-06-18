"""v27 grounding 게이트 정밀도 분석 (로컬, LLM 불필요).

v28은 grounding이 '기각한' 쪽(descriptor_ungrounded, NO)만 검증했다.
이 스크립트는 grounding이 '통과시킨' 쪽(FLIP된 시각수식어 commit, YES)을 추출한다.
-> 이게 precision 누수(환각인데 통과)를 찾기 위한 육안검증 대상.

출력: outputs/analysis/v27_flip_A_visdesc.csv  (육안검증 대상 목록)
"""
import csv, json, re, os
from collections import Counter

TEST_CSV = r"C:\Users\pak10\Downloads\open\test\test.csv"
SUB = "outputs/submission_v27_descriptor_grounding.csv"
DIAG = "outputs/v27_descriptor_grounding_diagnostics.csv"

# 셀8 원본 VIS_DESC 그대로
VIS_DESC = re.compile(r"wearing|in the |on the |with glasses|with the|holding|near the|"
                      r"left|right|center|jacket|shirt|hoodie|scarf|suit|cap|beanie|tie|"
                      r"top|pants|dress|hat|glasses|bag|sign|protective", re.I)
RE_A = re.compile(r"^\s*The image shows", re.I)            # 핸드오프 확정 올바른 A (=1750)
RE_A_BUG = re.compile(r"^\s*The image shows multiple people", re.I)  # 셀11 버그 버전
RE_IMGREF = re.compile(r"\b(image|photo|picture)\b", re.I)

T = {}
with open(TEST_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        T[r["sample_id"]] = {"ctx": r["context"], "q": r["question"],
                             "answers": json.loads(r["answers"]), "img": r["image_path"]}

SUBL = {}
with open(SUB, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        SUBL[r["sample_id"]] = int(r["label"])

REASON = {}
with open(DIAG, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        REASON[r["sample_id"]] = r["reason"]

print("recovery reason 분포:", dict(Counter(REASON.values())))

flips = [sid for sid, rs in REASON.items() if rs == "FLIP"]
print("FLIP 총:", len(flips))

rows_out = []
for sid in flips:
    t = T[sid]; lab = SUBL[sid]
    opt = t["answers"][lab]
    famA = bool(RE_A.search(t["ctx"]))
    famA_bug = bool(RE_A_BUG.search(t["ctx"]))
    vis = bool(VIS_DESC.search(opt))
    rows_out.append((sid, famA, famA_bug, vis, lab, opt, t["img"], t["ctx"], t["q"]))

nA = sum(1 for r in rows_out if r[1])
nVis = sum(1 for r in rows_out if r[3])
nAvis = sum(1 for r in rows_out if r[1] and r[3])
print(f"FLIP 중 A패밀리(올바른 정의): {nA}")
print(f"FLIP 중 시각수식어 옵션(grounding 통과한 것): {nVis}")
print(f"FLIP 중 A & 시각수식어 (= 핵심 검증대상): {nAvis}")
print(f"FLIP 중 A & 비시각수식어 (순수 집단명 commit): {nA - nAvis}")

os.makedirs("outputs/analysis", exist_ok=True)
OUT = "outputs/analysis/v27_flip_A_visdesc.csv"
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "label", "option", "image_path", "context", "question"])
    for sid, famA, famA_bug, vis, lab, opt, img, ctx, q in rows_out:
        if famA and vis:
            w.writerow([sid, lab, opt, img, ctx, q])
print("저장:", OUT)
