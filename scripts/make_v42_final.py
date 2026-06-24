import json, sys
sys.stdout.reconfigure(encoding='utf-8')

SRC = r"C:\Users\pak10\Downloads\colab_v31_grounding_off.ipynb"
OUT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v42_final.ipynb"

nb = json.load(open(SRC, encoding="utf-8")); S = nb["cells"]
def body(i): return "".join(S[i]["source"])
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}

body1 = body(1)
body1 = body1.replace(
    "Think briefly, then answer. Respond in EXACTLY this format, nothing else:",
    """11. 'Running the meeting' or 'leading the session' means that person IS the chair or leader. 'Taking notes' or 'preparing the room' means that person is in a support role. Map stated role descriptions directly to role titles.

Think briefly, then answer. Respond in EXACTLY this format, nothing else:"""
)
body1 = body1.replace("MAX_TOKENS = 256", "MAX_TOKENS = 128")
body1 = body1.replace(
    "Context: A Muslim woman and a Hindu woman were seated together on the flight.",
    """Context: We couldn't start the board meeting because Alex and Jordan were late. We had to wait for Jordan because they were running the meeting, and Alex took all the notes.
Question: Who was the board chair?
Options:
0. Alex
1. Jordan
2. Cannot be determined
Reasoning: Jordan was running the meeting, so Jordan is the board chair.
Answer: 1

Context: A Muslim woman and a Hindu woman were seated together on the flight."""
)

INTRO = """# SKKU Multimodal Challenge 2026 — Final Submission
"""

DATA = r"""import os, zipfile, csv, json, time
from tqdm.auto import tqdm
from google.colab import drive
drive.mount('/content/drive')
PROJECT='/content/drive/MyDrive/SKKU-Multimodal-Challenge-2026'
os.makedirs(f'{PROJECT}/outputs',exist_ok=True)
ZIP=f'{PROJECT}/open.zip'
if not os.path.isdir('/content/open') and not os.path.isdir('/content/test'):
    with zipfile.ZipFile(ZIP) as z: z.extractall('/content')
TEST_DIR=next((c for c in ['/content/open/test','/content/test'] if os.path.isdir(c)), None)
IMG_ROOT=TEST_DIR
rows, ids = [], []
with open(f'{TEST_DIR}/test.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        ans=json.loads(r['answers'])
        rows.append({'ctx':r['context'],'q':r['question'],'answers':ans,'unk':find_unknown(ans),'path':r['image_path']})
        ids.append(r['sample_id'])
print(f"{len(rows)} samples")
t=time.time(); images=[load_img(r['path']) for r in tqdm(rows,desc='images')]; print(f"loaded in {time.time()-t:.0f}s")
"""

CFBUILD = r"""import re
RE_A=re.compile(r"^\s*The image shows",re.I); RE_IMG=re.compile(r"\b(image|photo|picture)\b",re.I)
def fam(c): return 'A' if RE_A.search(c) else ('B' if RE_IMG.search(c) else 'C')
OPT_GRP=re.compile(r"^\s*(?:The|An?)\s+(.+?)\s+(?:person|man|woman|individual|people|guy|lady)\b",re.I)
def grp(o):
    m=OPT_GRP.search(o); return m.group(1).strip() if m else None
MALE=re.compile(r"\b(man|men|male|boy|gentleman|guy|father|son|brother|husband)\b",re.I)
FEMALE=re.compile(r"\b(woman|women|female|girl|lady|mother|daughter|sister|wife)\b",re.I)
def g_of(o):
    m=bool(MALE.search(o)); f=bool(FEMALE.search(o)); return 'M' if(m and not f)else('F' if(f and not m)else'-')
def ssw(text,a,b):
    t=re.sub(rf'\b{re.escape(a)}\b','\x00',text,flags=re.I); t=re.sub(rf'\b{re.escape(b)}\b',a,t,flags=re.I); return t.replace('\x00',b)
GP=[('woman','man'),('women','men'),('female','male'),('girl','boy'),('lady','gentleman'),
    ('mother','father'),('daughter','son'),('sister','brother'),('wife','husband'),('she','he'),('her','his')]
def gsw(t):
    for a,b in GP: t=ssw(t,a,b)
    return t

cf_type=[None]*len(rows); cf_rows=[]; cf_map=[]
for k,r in enumerate(rows):
    a=r['answers']; unk=r['unk']; non=[i for i in range(len(a)) if i!=unk]
    if len(non)!=2: continue
    f=fam(r['ctx'])
    if f=='A':
        g0,g1=grp(a[non[0]]),grp(a[non[1]])
        if not g0 or not g1 or g0.lower()==g1.lower(): continue
        if not (re.search(rf'\b{re.escape(g0)}\b',r['ctx'],re.I) and re.search(rf'\b{re.escape(g1)}\b',r['ctx'],re.I)): continue
        sc=ssw(r['ctx'],g0,g1); sa=[ssw(o,g0,g1) for o in a]; cf_type[k]='A'
    elif f=='B':
        if set([g_of(a[non[0]]),g_of(a[non[1]])])!={'M','F'}: continue
        sc=gsw(r['ctx']); sa=[gsw(o) for o in a]; cf_type[k]='B'
    else:
        continue
    cf_rows.append({'ctx':sc,'q':r['q'],'answers':sa,'unk':find_unknown(sa)}); cf_map.append(k)
cf_imgs=[images[k] for k in cf_map]
print(f"counterfactual: {len(cf_map)}")
"""

RUN = r"""import time
T_START=time.time()
t0=time.time(); base=run_single(rows, images); print(f"base {time.time()-t0:.0f}s")
t0=time.time(); cf=run_single(cf_rows, cf_imgs); print(f"cf {time.time()-t0:.0f}s")
"""

DEBIAS_LLM = r"""cf_pred={cf_map[j]:cf[j] for j in range(len(cf_map))}
cf_unk={cf_map[j]:cf_rows[j]['unk'] for j in range(len(cf_map))}
debiased=list(base)
conflicts=[]
for k in range(len(rows)):
    if k not in cf_pred: continue
    po,pc=base[k],cf_pred[k]
    if po==pc: continue
    uo,uc=rows[k]['unk'],cf_unk[k]
    if po!=uo and pc==uc: continue
    conflicts.append(k)

if conflicts:
    med_convs=[]
    for k in conflicts:
        r=rows[k]; po,pc=base[k],cf_pred[k]
        ans_o=r['answers'][po] if 0<=po<len(r['answers']) else '?'
        ans_c=r['answers'][pc] if 0<=pc<len(r['answers']) else '?'
        bias_warn=""
        if fam(r['ctx'])=='B':
            bias_warn=("CRITICAL: The photographs are POSED stock images with actors. "
                       "Physical posture, gestures, facial expressions, seating position, and body language "
                       "are NOT valid evidence for personality traits, authority, emotionality, or capability. "
                       "If the answer relies on visual appearance rather than explicitly stated facts, choose 'unknown'.\n\n")
        med_text=(f"Two inference passes on the same question produced different answers:\n"
                  f"- Pass 1 (original context): {ans_o}\n"
                  f"- Pass 2 (group labels swapped): {ans_c}\n\n"
                  f"If answers change when group labels swap, the original may be biased. "
                  f"Decide based ONLY on explicitly stated actions/facts in the context.\n\n"
                  + bias_warn
                  + build_user_text(r['ctx'],r['q'],r['answers']))
        uc=[]
        if images[k] is not None:
            uc.append({"type":"image_url","image_url":{"url":to_url(images[k])}})
        uc.append({"type":"text","text":med_text})
        med_convs.append([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":uc}])
    try:
        med_outs=llm.chat(med_convs,_sp(0.0),use_tqdm=True,chat_template_kwargs={"enable_thinking":False})
    except Exception:
        med_outs=llm.chat(med_convs,_sp(0.0),use_tqdm=True)
    for k,o in zip(conflicts,med_outs):
        debiased[k]=parse_answer(o.outputs[0].text,rows[k]['answers'],rows[k]['unk'])
    print(f"mediation: {len(conflicts)} conflicts")

unk_mask=[debiased[i]==rows[i]['unk'] for i in range(len(rows))]
"""

LLM_RECOVERY = r"""import re, time
ATTR_RE=re.compile(r"\.\s+(?:An?|The)\s+([A-Z][a-zA-Z\- ]+?)\s+(?:person|man|woman)\b")
recovery_targets=[]
for k in range(len(rows)):
    if not unk_mask[k]: continue
    if fam(rows[k]['ctx'])!='A': continue
    m=ATTR_RE.search(' '+str(rows[k]['ctx']))
    if not m: continue
    g=m.group(1).strip().lower()
    h=[i for i,x in enumerate(rows[k]['answers']) if g in x.lower() and i!=rows[k]['unk']]
    if len(h)!=1: continue
    recovery_targets.append(k)

def extract_key_sentence(ctx):
    m=ATTR_RE.search(' '+str(ctx))
    if not m: return ""
    full=' '+str(ctx)
    dot_pos=full.rfind('.',0,m.start())
    if dot_pos<0: dot_pos=0
    end=full.find('.',m.end())
    if end<0: end=len(full)
    return full[dot_pos:end+1].strip()

final=list(debiased)
stage1_fail=[]

if recovery_targets:
    rec_convs=[]
    for k in recovery_targets:
        r=rows[k]
        key_sent=extract_key_sentence(r['ctx'])
        rec_text=(f"The context DEFINITELY contains a sentence attributing a specific action to one identified person. "
                  f"Abstaining is WRONG here — you MUST commit.\n\n"
                  f"Key sentence: \"{key_sent}\"\n\n"
                  f"The person named in this sentence performed the stated action. "
                  f"Identify them and commit to the matching option.\n\n"
                  + build_user_text(r['ctx'],r['q'],r['answers']))
        uc=[]
        if images[k] is not None:
            uc.append({"type":"image_url","image_url":{"url":to_url(images[k])}})
        uc.append({"type":"text","text":rec_text})
        rec_convs.append([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":uc}])
    try:
        rec_outs=llm.chat(rec_convs,_sp(0.0),use_tqdm=True,chat_template_kwargs={"enable_thinking":False})
    except Exception:
        rec_outs=llm.chat(rec_convs,_sp(0.0),use_tqdm=True)
    for k,o in zip(recovery_targets,rec_outs):
        new_p=parse_answer(o.outputs[0].text,rows[k]['answers'],rows[k]['unk'])
        if new_p!=rows[k]['unk']:
            final[k]=new_p
        else:
            stage1_fail.append(k)

if stage1_fail:
    rec2_convs=[]
    for k in stage1_fail:
        r=rows[k]; unk=r['unk']
        non=[i for i in range(len(r['answers'])) if i!=unk]
        key_sent=extract_key_sentence(r['ctx'])
        opt_a=r['answers'][non[0]] if len(non)>=1 else '?'
        opt_b=r['answers'][non[1]] if len(non)>=2 else '?'
        rec2_text=(f"There are ONLY two possible answers. 'Unknown' is NOT acceptable.\n\n"
                   f"The answer is EITHER:\n"
                   f"  [{non[0]}] {opt_a}\n"
                   f"  [{non[1]}] {opt_b}\n\n"
                   f"The context states: \"{key_sent}\"\n"
                   f"This sentence names a specific person who performed an action. "
                   f"Match that person to one of the two options above.\n\n"
                   + build_user_text(r['ctx'],r['q'],r['answers']))
        uc=[]
        if images[k] is not None:
            uc.append({"type":"image_url","image_url":{"url":to_url(images[k])}})
        uc.append({"type":"text","text":rec2_text})
        rec2_convs.append([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":uc}])
    try:
        rec2_outs=llm.chat(rec2_convs,_sp(0.0),use_tqdm=True,chat_template_kwargs={"enable_thinking":False})
    except Exception:
        rec2_outs=llm.chat(rec2_convs,_sp(0.0),use_tqdm=True)
    for k,o in zip(stage1_fail,rec2_outs):
        new_p=parse_answer(o.outputs[0].text,rows[k]['answers'],rows[k]['unk'])
        if new_p!=rows[k]['unk']:
            final[k]=new_p

print(f"recovery: {sum(1 for k in recovery_targets if final[k]!=rows[k]['unk'])}/{len(recovery_targets)}")
"""

SAVE = r"""import csv, time
elapsed=(time.time()-T_START)/60
print(f"elapsed: {elapsed:.1f}min")
OUT=f'{PROJECT}/outputs/submission_v42.csv'
with open(OUT,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['sample_id','label'])
    for i,p in zip(ids,final): w.writerow([i,p])
print(f"saved: {OUT}")
"""

cells = [
    md(INTRO),
    code(body(0)),
    code(body1),
    code(body(2)),
    code(body(3)),
    code(DATA),
    code(CFBUILD),
    code(RUN),
    code(DEBIAS_LLM),
    code(LLM_RECOVERY),
    code(SAVE),
]

nb["cells"] = cells
json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"generated: {OUT} | {len(cells)} cells")

all_src = "".join(
    "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
    for c in cells
)
checks = [
    ("Rule 11", "Running the meeting" in all_src),
    ("board chair few-shot", "Jordan was running the meeting" in all_src),
    ("B bias warning in mediation only", "POSED stock images" in DEBIAS_LLM and "POSED" not in body1),
    ("no ATTR override", "final[k]=tgt" not in all_src),
    ("recovery 2-stage", "ONLY two possible answers" in LLM_RECOVERY),
    ("MAX_TOKENS=128", "MAX_TOKENS = 128" in body1),
    ("no chart code", "matplotlib" not in all_src),
    ("no API code", "api_pred" not in all_src and "gemini" not in all_src.lower()),
    ("no analysis", "defaultdict" not in all_src),
    ("saves v42 csv", "submission_v42" in SAVE),
    ("11 cells", len(cells) == 11),
]
print(f"\ncheck {sum(1 for _,v in checks if v)}/{len(checks)}:")
for name, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'} {name}")
