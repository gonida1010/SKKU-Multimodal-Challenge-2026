"""전체 submission CSV 총합 → 오답/불일치 분석 → 이미지 폴더 생성."""
import csv, json, os, re, shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026"
TEST_DIR = r"C:\Users\pak10\Downloads\open\test"
OUT_DIR = os.path.join(PROJECT, "analysis_wrong")
IMG_OUT = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_OUT, exist_ok=True)

# ── 1. test.csv 로드 ──
rows, ids = [], []
with open(os.path.join(TEST_DIR, "test.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ans = json.loads(r["answers"])
        rows.append({"ctx": r["context"], "q": r["question"], "answers": ans, "path": r["image_path"]})
        ids.append(r["sample_id"])
print(f"test.csv: {len(rows)}건")

# unknown 탐지
UNK = ["cannot be determined","can't be determined","cannot determine","can't determine",
  "cannot be known","cannot answer","can't answer","cannot be answered","not answerable",
  "not enough info","not enough information","not enough data","insufficient info",
  "insufficient information","not known","unknown","undetermined","cannot say","can't say",
  "cannot tell","can't tell","hard to determine","difficult to determine",
  "impossible to determine","impossible to tell","not specified","unspecified",
  "no information","not provided","unclear","not clear"]
def find_unknown(answers):
    f = [any(p in a.lower() for p in UNK) for a in answers]
    if sum(f) == 1: return f.index(True)
    idx = [i for i, x in enumerate(f) if x]
    return min(idx, key=lambda i: len(answers[i])) if idx else None
for r in rows:
    r["unk"] = find_unknown(r["answers"])

# 패밀리
RE_A = re.compile(r"^\s*The image shows", re.I)
RE_IMG = re.compile(r"\b(image|photo|picture)\b", re.I)
def fam(c): return "A" if RE_A.search(c) else ("B" if RE_IMG.search(c) else "C")
families = [fam(r["ctx"]) for r in rows]

# ── 2. 모든 submission CSV 로드 ──
csv_dir = os.path.join(PROJECT, "outputs")
all_csvs = {}
for fn in os.listdir(csv_dir):
    fp = os.path.join(csv_dir, fn)
    if not fn.endswith(".csv") or os.path.isdir(fp):
        continue
    try:
        with open(fp, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "sample_id" not in reader.fieldnames or "label" not in reader.fieldnames:
                continue
            preds = {}
            for r in reader:
                preds[r["sample_id"]] = int(r["label"])
            if len(preds) >= 8000:  # submission CSV만
                name = fn.replace("submission_", "").replace(".csv", "")
                all_csvs[name] = preds
    except Exception:
        continue
print(f"로드된 submission CSV: {len(all_csvs)}개")
for name in sorted(all_csvs.keys()):
    print(f"  {name}: {len(all_csvs[name])}건")

# ── 3. synth_gold (A패밀리 프록시 정답) ──
ATTR = re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b(.{0,120})")
EV_NEG = re.compile(r"struggl|difficult|fail|lost|steal|stole|interrupt|yell|rude|late|cheat|lie|lied|could not|couldn t|cannot|debt|fell apart|shoplift|arrest|mock|complain|trouble")
Q_NEG = re.compile(r"less likely|not |unlikely|least ")
TRAIT_NEG = re.compile(r"lie|steal|poor|criminal|rude|fail|violen|less educated|unemploy|suspicious|emotional|weak|lazy|cheat|trouble|danger")
def sg(ctx, q, a, unk):
    m = ATTR.search(" " + str(ctx))
    if not m: return ("ambig", unk)
    g = m.group(1).strip().lower()
    h = [i for i, x in enumerate(a) if g in x.lower() and i != unk]
    if len(h) != 1: return ("skip", None)
    tgt = h[0]; oth = [i for i in range(3) if i not in (tgt, unk)][0]
    ev = bool(EV_NEG.search(m.group(2).lower())); ql = str(q).lower()
    return ("disambig", tgt if ((bool(TRAIT_NEG.search(ql)) == ev) != bool(Q_NEG.search(ql))) else oth)

sg_results = []
for r in rows:
    sg_results.append(sg(r["ctx"], r["q"], r["answers"], r["unk"]))

# ── 4. 버전별 오답 분석 ──
ver_names = sorted(all_csvs.keys())
# 각 샘플별: 다수결 + synth_gold
sample_analysis = []
wrong_ids = set()

for k in range(len(rows)):
    sid = ids[k]
    r = rows[k]
    f = families[k]
    sg_type, sg_ans = sg_results[k]

    # 각 버전 예측
    preds = {}
    for vn in ver_names:
        if sid in all_csvs[vn]:
            preds[vn] = all_csvs[vn][sid]

    if not preds:
        continue

    # 다수결
    vote = Counter(preds.values())
    majority, majority_count = vote.most_common(1)[0]
    total_votes = len(preds)
    consensus_rate = majority_count / total_votes

    # 이 샘플이 "오답"인 기준:
    # (1) synth_gold 판정 가능(disambig) + 과반이 synth_gold과 다름
    # (2) 합의율 < 100% (버전 간 불일치 존재)
    # (3) 최신 최선(v36) 예측이 synth_gold/다수결과 다름

    is_wrong = False
    wrong_reason = []

    # synth_gold 기반 (A패밀리)
    if f == "A" and sg_type == "disambig" and sg_ans is not None:
        wrong_vers = [vn for vn, p in preds.items() if p != sg_ans]
        if wrong_vers:
            is_wrong = True
            wrong_reason.append(f"synth_gold({len(wrong_vers)}ver)")

    # 불일치 (consensus < 100%)
    if consensus_rate < 1.0:
        minority_vers = [vn for vn, p in preds.items() if p != majority]
        is_wrong = True
        wrong_reason.append(f"disagree({len(minority_vers)}ver,consensus={consensus_rate:.0%})")

    if is_wrong:
        wrong_ids.add(k)

    sample_analysis.append({
        "idx": k,
        "sample_id": sid,
        "family": f,
        "sg_type": sg_type,
        "sg_ans": sg_ans,
        "unk": r["unk"],
        "majority": majority,
        "consensus_rate": consensus_rate,
        "is_wrong": is_wrong,
        "wrong_reason": "|".join(wrong_reason),
        "preds": preds,
        "n_versions": total_votes,
        "path": r["path"],
        "context": r["ctx"][:200],
        "question": r["q"],
        "answers": r["answers"],
    })

print(f"\n=== 분석 결과 ===")
print(f"전체 샘플: {len(rows)}")
print(f"오답/불일치 샘플: {len(wrong_ids)}")
print(f"  A패밀리: {sum(1 for k in wrong_ids if families[k]=='A')}")
print(f"  B패밀리: {sum(1 for k in wrong_ids if families[k]=='B')}")
print(f"  C패밀리: {sum(1 for k in wrong_ids if families[k]=='C')}")

# ── 5. 오답 이미지 복사 + CSV 저장 ──
print(f"\n이미지 복사 시작 ({len(wrong_ids)}건)...")
copied = 0
for k in sorted(wrong_ids):
    r = rows[k]
    src_path = Path(TEST_DIR) / r["path"]
    if src_path.exists():
        # 패밀리별 서브폴더
        fam_dir = os.path.join(IMG_OUT, families[k])
        os.makedirs(fam_dir, exist_ok=True)
        dst = os.path.join(fam_dir, f"{ids[k]}_{src_path.name}")
        shutil.copy2(str(src_path), dst)
        copied += 1
print(f"복사 완료: {copied}/{len(wrong_ids)}")

# CSV 저장: 오답 샘플 상세
out_csv = os.path.join(OUT_DIR, "wrong_predictions_all.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    header = ["sample_id", "family", "sg_type", "sg_answer", "unk_idx", "majority_vote",
              "consensus_rate", "wrong_reason", "context", "question",
              "opt0", "opt1", "opt2", "image_path"]
    header += [f"pred_{vn}" for vn in ver_names]
    w.writerow(header)
    for sa in sample_analysis:
        if not sa["is_wrong"]:
            continue
        row = [
            sa["sample_id"], sa["family"], sa["sg_type"], sa["sg_ans"], sa["unk"],
            sa["majority"], f"{sa['consensus_rate']:.3f}", sa["wrong_reason"],
            sa["context"], sa["question"],
            sa["answers"][0] if len(sa["answers"]) > 0 else "",
            sa["answers"][1] if len(sa["answers"]) > 1 else "",
            sa["answers"][2] if len(sa["answers"]) > 2 else "",
            sa["path"],
        ]
        for vn in ver_names:
            row.append(sa["preds"].get(vn, ""))
        w.writerow(row)
print(f"CSV 저장: {out_csv}")

# ── 6. 요약 통계 ──
print(f"\n=== 패밀리별 불일치 유형 ===")
for fam_name in ["A", "B", "C"]:
    fam_wrong = [sa for sa in sample_analysis if sa["is_wrong"] and sa["family"] == fam_name]
    if not fam_wrong:
        continue
    # commit vs unknown 비율
    n_commit = sum(1 for sa in fam_wrong if sa["majority"] != sa["unk"])
    n_unk = sum(1 for sa in fam_wrong if sa["majority"] == sa["unk"])
    # consensus 분포
    low_cons = sum(1 for sa in fam_wrong if sa["consensus_rate"] < 0.5)
    mid_cons = sum(1 for sa in fam_wrong if 0.5 <= sa["consensus_rate"] < 0.8)
    high_cons = sum(1 for sa in fam_wrong if sa["consensus_rate"] >= 0.8)
    print(f"  {fam_name}: {len(fam_wrong)}건 | majority=commit {n_commit} / unk {n_unk}")
    print(f"     consensus: <50% {low_cons} | 50-80% {mid_cons} | ≥80% {high_cons}")

    # synth_gold 분석 (A만)
    if fam_name == "A":
        sg_disambig = [sa for sa in fam_wrong if sa["sg_type"] == "disambig"]
        sg_ambig = [sa for sa in fam_wrong if sa["sg_type"] == "ambig"]
        sg_skip = [sa for sa in fam_wrong if sa["sg_type"] == "skip"]
        print(f"     synth_gold: disambig {len(sg_disambig)} | ambig {len(sg_ambig)} | skip {len(sg_skip)}")

# 버전별 오답률
print(f"\n=== 버전별 '다수결과 불일치' 수 ===")
ver_wrong_counts = []
for vn in ver_names:
    n_wrong = 0
    for sa in sample_analysis:
        if sa["is_wrong"] and vn in sa["preds"] and sa["preds"][vn] != sa["majority"]:
            n_wrong += 1
    ver_wrong_counts.append((vn, n_wrong))
ver_wrong_counts.sort(key=lambda x: x[1])
for vn, nw in ver_wrong_counts:
    print(f"  {vn:45s}: {nw:4d}건 다수결 불일치")

print(f"\n=== 완료 ===")
print(f"분석 폴더: {OUT_DIR}")
print(f"  images/A/ — A패밀리 불일치 이미지")
print(f"  images/B/ — B패밀리 불일치 이미지")
print(f"  images/C/ — C패밀리 불일치 이미지")
print(f"  wrong_predictions_all.csv — 전체 오답 상세 (모든 버전 예측 포함)")
