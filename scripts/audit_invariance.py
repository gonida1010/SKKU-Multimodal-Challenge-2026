"""제3의 가설 1차 증거 — 일반화 취약성(spurious cue 의존) 로컬 감사. GT/이미지 불필요.
핵심: 파이프라인 토대인 find_unknown(unknown 보기 탐지)이 하드코딩 목록 + '제일 짧은 보기' 가정에
얼마나 의존하나? 이 가정이 Hidden에서 깨지면 ambig_acc가 붕괴한다.
"""
import csv, json, re
from collections import Counter

TEST = r"C:\Users\pak10\Downloads\open\test\test.csv"

# 현재 파이프라인의 하드코딩 UNK 목록 (셀1 원본)
UNK = ["cannot be determined","can't be determined","cannot determine","can't determine",
  "cannot be known","cannot answer","can't answer","cannot be answered","not answerable",
  "not enough info","not enough information","not enough data","insufficient info",
  "insufficient information","not known","unknown","undetermined","cannot say","can't say",
  "cannot tell","can't tell","hard to determine","difficult to determine",
  "impossible to determine","impossible to tell","not specified","unspecified",
  "no information","not provided","unclear","not clear"]
def unk_matches(opts):
    return [i for i,a in enumerate(opts) if any(p in a.lower() for p in UNK)]
def find_unknown(opts):
    m = unk_matches(opts)
    if len(m)==1: return m[0], 'exact1'
    if len(m)==0:
        return min(range(len(opts)), key=lambda i: len(opts[i])), 'fallback_shortest_0match'
    return min(m, key=lambda i: len(opts[i])), 'multi_shortest'

rows=[]
for r in csv.DictReader(open(TEST,encoding="utf-8")):
    rows.append(json.loads(r["answers"]))
N=len(rows)
print(f"test {N}건\n")

# 1) find_unknown 모드 분포 = 하드코딩/길이가정 의존도
mode=Counter()
unk_not_shortest=0          # unknown 보기가 '제일 짧은 보기'가 아닌 경우 (길이가정 위험)
unk_idx_all=[]
for opts in rows:
    ui,md = find_unknown(opts); mode[md]+=1; unk_idx_all.append(ui)
    shortest = min(range(len(opts)), key=lambda i: len(opts[i]))
    if ui!=shortest: unk_not_shortest+=1
print("=== find_unknown 판정 모드 (Hidden에서 깨질 위험 = exact1 외 전부) ===")
for k,v in mode.most_common(): print(f"  {k:26s}: {v}  ({v/N*100:.1f}%)")
print(f"  → 'exact1' 외 = 하드코딩목록 불충분/길이가정 의존: {N-mode['exact1']} ({(N-mode['exact1'])/N*100:.1f}%)")
print(f"  unknown 보기가 최단이 아닌 샘플(길이가정 깨짐): {unk_not_shortest} ({unk_not_shortest/N*100:.1f}%)\n")

# 2) 보기 문자열을 빈도로 봐서 '템플릿 재사용' unknown 표현을 경험적으로 추출 (목록과 무관하게)
allopt=Counter()
for opts in rows:
    for a in opts: allopt[a.strip()]+=1
UNC_HINT=re.compile(r"determin|enough|cannot|can't|can not|unknown|not known|not sure|"
                    r"unclear|tell|specified|possible|no info|insufficient|undeterm|"
                    r"not answer|hard to|impossible|not provided|don't know|do not know", re.I)
print("=== 고빈도 보기 중 'uncertainty처럼 보이지만 UNK목록 미매칭' (현재 버그 + Hidden 위험) ===")
gap=[(o,c) for o,c in allopt.items() if c>=5 and UNC_HINT.search(o) and not any(p in o.lower() for p in UNK)]
for o,c in sorted(gap,key=lambda x:-x[1])[:25]:
    print(f"  {c:4d}x  | {o}")
print(f"  미매칭 uncertainty 표현 종류(빈도≥5): {len(gap)}\n")

# 3) 참고: unknown 표현의 다양성 (템플릿 수) — UNK목록이 잡은 것 기준
unk_strings=Counter()
for opts,ui in zip(rows,unk_idx_all):
    unk_strings[opts[ui].strip()]+=1
print(f"=== unknown으로 판정된 보기의 고유 표현 수: {len(unk_strings)} (상위 12) ===")
for o,c in unk_strings.most_common(12): print(f"  {c:4d}x  | {o}")
