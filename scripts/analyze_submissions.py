"""제출 CSV 교차분석: 라벨 없이, '어떤 행동이 점수를 올렸나'를 정량화.

핵심 아이디어: 라벨은 없지만 test 구조로 두 행동을 측정 가능.
  - 'stated-action' context(≈ disambiguated): 모델이 commit해야 정답 -> commit_rate 높을수록 good
  - 'no-action' context(≈ ambiguous): 모델이 abstain해야 정답 -> abstain_rate 높을수록 good
이 둘을 각 버전의 리더보드 점수와 늘어놓으면 무엇이 점수를 움직였는지 보인다.
"""
import csv, json, re, sys, glob, os
sys.path.insert(0, "src")
from data import find_unknown_index

TEST = "open/test/test.csv"
# 파일명 -> (라벨, 리더보드점수)
SUBS = {
    "outputs/0.9756666667_submission.csv": ("v8_35B", 0.9757),
    "outputs/0.9923333333_v12.csv":        ("v12_uncensored", 0.9923),
    "outputs/0.99325_submission.csv":      ("v5_imageOFF", 0.99325),
    "outputs/0.9938333333_submission.csv": ("v4_elim", 0.9938),
    "outputs/0.9963333333_submission.csv": ("v5_fewshot", 0.9963),
    "outputs/v9_submission.csv":           ("v9_verifier", 0.99508),
    "outputs/v11_submission.csv":          ("v11_permsc", 0.997),
    "outputs/v7_submission.csv":           ("v7_appearance", None),  # 미제출
}

# 행동/진술 동사 = disambiguation 신호 (그 사람에게 무언가가 '진술'됨)
ACTION = re.compile(
    r"\b(told|said|talked|looked|could not|couldn't|misled|mentioned|kept|needed|struggl\w*|"
    r"tested|expressed|stole|lied|hides?|won|lost|asked|refused|helped|shoved|finished|dropped|"
    r"missed|described|caught|arrested|skipped|delayed|ignored|praised|received|forgot|broke|"
    r"yelled|interrupt\w*|cheated|admitted|complained|failed|passed|solved|spent|saved|spends?)\b",
    re.IGNORECASE)

def load_test():
    rows = {}
    with open(TEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ans = json.loads(r["answers"])
            rows[r["sample_id"]] = {
                "unk": find_unknown_index(ans),
                "has_action": bool(ACTION.search(r["context"])),
                "ctx": r["context"], "q": r["question"], "ans": ans,
            }
    return rows

def load_sub(p):
    d = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[r["sample_id"]] = int(r["label"])
    return d

def main():
    T = load_test()
    ids = list(T)
    n_act = sum(T[s]["has_action"] for s in ids)
    print(f"test {len(ids)} | stated-action(≈disambig) {n_act} | no-action(≈ambig) {len(ids)-n_act}\n")

    subs = {}
    for p, (name, score) in SUBS.items():
        if os.path.exists(p): subs[name] = (load_sub(p), score)

    # 1) 버전별 핵심 행동 지표 (점수순)
    print(f"{'version':<16}{'LB':>9}{'unk_rate':>9}{'commit@action':>15}{'abstain@noact':>15}")
    rowstat = {}
    for name,(d,score) in sorted(subs.items(), key=lambda kv: (kv[1][1] is None, kv[1][1] or 0)):
        unk = ca = na = ab = nn = 0
        for s in ids:
            p = d[s]; t = T[s]
            unk += (p == t["unk"])
            if t["has_action"]:
                na += 1; ca += (p != t["unk"])      # 근거 있는데 commit (good)
            else:
                nn += 1; ab += (p == t["unk"])      # 근거 없는데 abstain (good)
        rowstat[name] = (ca/na, ab/nn)
        sc = f"{score:.4f}" if score else "  (미제출)"
        print(f"{name:<16}{sc:>9}{unk/len(ids)*100:>8.1f}%{ca/na*100:>13.1f}%{ab/nn*100:>14.1f}%")

    # 2) 점수 상승 구간별 '바뀐 샘플' 방향 분해
    print("\n=== 점수 상승 구간: 무엇이 바뀌었나 ===")
    chain = [("v8_35B",None),("v12_uncensored",None),("v5_imageOFF",None),
             ("v9_verifier",None),("v4_elim",None),("v5_fewshot",None),("v11_permsc",None)]
    order = [n for n in ["v4_elim","v5_fewshot","v9_verifier","v11_permsc"] if n in subs]
    def cmp(a,b):
        da,db = subs[a][0], subs[b][0]
        toCommit=toAbstain=swap=0
        for s in ids:
            if da[s]==db[s]: continue
            u=T[s]["unk"]
            if da[s]==u and db[s]!=u: toCommit+=1     # a가 도망->b가 commit
            elif da[s]!=u and db[s]==u: toAbstain+=1  # a가 commit->b가 도망
            else: swap+=1
        return toCommit,toAbstain,swap
    pairs=[("v4_elim","v5_fewshot"),("v5_fewshot","v11_permsc"),
           ("v9_verifier","v11_permsc"),("v12_uncensored","v11_permsc"),("v5_imageOFF","v5_fewshot")]
    for a,b in pairs:
        if a in subs and b in subs:
            tc,ta,sw=cmp(a,b); sa=subs[a][1]; sb=subs[b][1]
            print(f"{a}({sa})->{b}({sb}): 도망→commit {tc} | commit→도망 {ta} | 인물교체 {sw}  (총 {tc+ta+sw})")

    # 3) 합의 기반 의심 오류: 고득점 3버전이 합의했는데 특정 버전만 다른 곳
    best = ["v11_permsc","v5_fewshot","v4_elim"]
    if all(b in subs for b in best):
        cons=0; contested=[]
        for s in ids:
            vals=[subs[b][0][s] for b in best]
            if len(set(vals))==1: cons+=1
            else: contested.append(s)
        print(f"\n고득점3(v11,v5,v4) 합의 {cons}/{len(ids)} ({cons/len(ids)*100:.1f}%) | 불일치 {len(contested)}")

if __name__ == "__main__":
    main()
