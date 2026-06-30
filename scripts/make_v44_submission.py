"""v44 제출용 clean notebook 생성기 (v2).
- 마크다운 없음, G4/Blackwell 코드 없음
- import 한 셀에 통합
- 셀 상단 한 줄 주석만
"""
import json, sys, re as _re
sys.stdout.reconfigure(encoding='utf-8')

SRC = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v44_final.ipynb"
OUT = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v44_submission.ipynb"

orig = json.load(open(SRC, encoding="utf-8"))
def body(i): return "".join(orig["cells"][i]["source"])

def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.splitlines(keepends=True)}

# ── 원본 cell 2에서 SYSTEM_PROMPT, UNK, find_unknown, parse_answer 추출 ──
cell2 = body(2)

sp_m = _re.search(r'(SYSTEM_PROMPT = """.*?Answer: 2""")', cell2, _re.DOTALL)
SYSPROMPT_DEF = sp_m.group(1)

unk_m = _re.search(r'(UNK = \[.*?"not clear"\])', cell2, _re.DOTALL)
UNK_DEF = unk_m.group(1)

fu_m = _re.search(r'(def find_unknown\(answers\):.*?)(?=\n\n)', cell2, _re.DOTALL)
FIND_UNKNOWN = fu_m.group(1)

bu_m = _re.search(r'(def build_user_text\(ctx.*?\))', cell2, _re.DOTALL)
BUILD_USER = cell2[bu_m.start():cell2.index('\n\n', bu_m.end())]

pa_m = _re.search(r'(_ANS = .*?def parse_answer.*?)(?=\nimport torch)', cell2, _re.DOTALL)
PARSE_ANSWER = pa_m.group(1).strip()

# ════════════════════════════════════════════════════════════════
# Cell 0: 임포트 + 모델 로드
# ════════════════════════════════════════════════════════════════
IMPORTS_AND_MODEL = (
    '# 임포트 + 모델 로드\n'
    'import os, sys, json, re, base64, csv, time\n'
    'from io import BytesIO\n'
    'from pathlib import Path\n'
    'from PIL import Image\n'
    'from tqdm.auto import tqdm\n'
    'import torch\n'
    'from vllm import LLM, SamplingParams\n'
    '\n'
    'MODEL = "./model"\n'
    'MAX_TOKENS = 128\n'
    '\n'
    'print("GPU:", torch.cuda.get_device_name(0), "|",\n'
    '      "torch:", torch.__version__, "| cuda:", torch.version.cuda)\n'
    '\n'
    'llm = LLM(model=MODEL, dtype="auto", max_model_len=16384,\n'
    '          gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 1},\n'
    '          trust_remote_code=True, seed=42)\n'
    'print("모델 로드 완료:", MODEL)\n'
)

# ════════════════════════════════════════════════════════════════
# Cell 2: 상수 + 함수 정의
# ════════════════════════════════════════════════════════════════
CONSTANTS_AND_FUNCS = (
    '# 상수 + 함수 정의\n'
    + UNK_DEF + '\n\n'
    + FIND_UNKNOWN + '\n\n'
    + SYSPROMPT_DEF + '\n\n'
    + BUILD_USER + '\n\n'
    + PARSE_ANSWER + '\n\n'
    'def _sp(temp=0.0):\n'
    '    return SamplingParams(temperature=temp, top_p=1.0, max_tokens=MAX_TOKENS, seed=42)\n'
    '\n'
    'def to_url(im):\n'
    '    b = BytesIO()\n'
    '    im.save(b, format="JPEG", quality=95)\n'
    '    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()\n'
    '\n'
    'def load_img(p, max_side=768):\n'
    '    if p is None: return None\n'
    '    try:\n'
    '        im = Image.open(Path(IMG_ROOT) / p).convert("RGB")\n'
    '        s = max_side / max(im.size)\n'
    '        return im.resize((int(im.size[0]*s), int(im.size[1]*s))) if s < 1 else im\n'
    '    except Exception:\n'
    '        return None\n'
    '\n'
    'def generate(rows, images, temp=0.0):\n'
    '    convs = []\n'
    '    for r, im in zip(rows, images):\n'
    '        uc = []\n'
    '        if im is not None:\n'
    '            uc.append({"type": "image_url", "image_url": {"url": to_url(im)}})\n'
    '        uc.append({"type": "text", "text": build_user_text(r["ctx"], r["q"], r["answers"])})\n'
    '        convs.append([{"role": "system", "content": SYSTEM_PROMPT},\n'
    '                      {"role": "user", "content": uc}])\n'
    '    try:\n'
    '        outs = llm.chat(convs, _sp(temp), use_tqdm=True,\n'
    '                        chat_template_kwargs={"enable_thinking": False})\n'
    '    except Exception:\n'
    '        outs = llm.chat(convs, _sp(temp), use_tqdm=True)\n'
    '    return [o.outputs[0].text for o in outs]\n'
    '\n'
    'def run_single(rows, images):\n'
    '    out = generate(rows, images)\n'
    '    return [parse_answer(t, r["answers"], r["unk"]) for t, r in zip(out, rows)]\n'
)

# ════════════════════════════════════════════════════════════════
# Cell 3: 데이터 로드
# ════════════════════════════════════════════════════════════════
DATA_LOAD = """# 데이터 로드
TEST_DIR = './test'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_ROOT = TEST_DIR

rows, ids = [], []
with open(f'{TEST_DIR}/test.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        ans = json.loads(r['answers'])
        rows.append({'ctx': r['context'], 'q': r['question'],
                     'answers': ans, 'unk': find_unknown(ans), 'path': r['image_path']})
        ids.append(r['sample_id'])
print(f"{len(rows)} samples")

t0 = time.time()
images = [load_img(r['path']) for r in tqdm(rows, desc='images')]
print(f"loaded in {time.time()-t0:.0f}s")
"""

# ════════════════════════════════════════════════════════════════
# Cell 4: 반사실 구성
# ════════════════════════════════════════════════════════════════
CF_BUILD = """# 반사실(CF) 데이터 구성
RE_A = re.compile(r"^\\s*The image shows", re.I)
RE_IMG = re.compile(r"\\b(image|photo|picture)\\b", re.I)
def fam(c):
    return 'A' if RE_A.search(c) else ('B' if RE_IMG.search(c) else 'C')

OPT_GRP = re.compile(r"^\\s*(?:The|An?)\\s+(.+?)\\s+(?:person|man|woman|individual|people|guy|lady)\\b", re.I)
def grp(o):
    m = OPT_GRP.search(o)
    return m.group(1).strip() if m else None

MALE = re.compile(r"\\b(man|men|male|boy|gentleman|guy|father|son|brother|husband)\\b", re.I)
FEMALE = re.compile(r"\\b(woman|women|female|girl|lady|mother|daughter|sister|wife)\\b", re.I)
def g_of(o):
    m = bool(MALE.search(o)); f = bool(FEMALE.search(o))
    return 'M' if (m and not f) else ('F' if (f and not m) else '-')

def ssw(text, a, b):
    t = re.sub(rf'\\b{re.escape(a)}\\b', '\\x00', text, flags=re.I)
    t = re.sub(rf'\\b{re.escape(b)}\\b', a, t, flags=re.I)
    return t.replace('\\x00', b)

GP = [('woman','man'),('women','men'),('female','male'),('girl','boy'),('lady','gentleman'),
      ('mother','father'),('daughter','son'),('sister','brother'),('wife','husband'),
      ('she','he'),('her','his')]
def gsw(t):
    for a, b in GP: t = ssw(t, a, b)
    return t

cf_type = [None]*len(rows); cf_rows = []; cf_map = []
for k, r in enumerate(rows):
    a = r['answers']; unk = r['unk']
    non = [i for i in range(len(a)) if i != unk]
    if len(non) != 2: continue
    f = fam(r['ctx'])
    if f == 'A':
        g0, g1 = grp(a[non[0]]), grp(a[non[1]])
        if not g0 or not g1 or g0.lower() == g1.lower(): continue
        if not (re.search(rf'\\b{re.escape(g0)}\\b', r['ctx'], re.I) and
                re.search(rf'\\b{re.escape(g1)}\\b', r['ctx'], re.I)): continue
        sc = ssw(r['ctx'], g0, g1); sa = [ssw(o, g0, g1) for o in a]; cf_type[k] = 'A'
    elif f == 'B':
        if set([g_of(a[non[0]]), g_of(a[non[1]])]) != {'M', 'F'}: continue
        sc = gsw(r['ctx']); sa = [gsw(o) for o in a]; cf_type[k] = 'B'
    else:
        continue
    cf_rows.append({'ctx': sc, 'q': r['q'], 'answers': sa, 'unk': find_unknown(sa)})
    cf_map.append(k)
cf_imgs = [images[k] for k in cf_map]
print(f"counterfactual: {len(cf_map)}")
"""

# ════════════════════════════════════════════════════════════════
# Cell 5: Base + CF 추론
# ════════════════════════════════════════════════════════════════
INFERENCE = """# Base + CF 추론
T_START = time.time()
t0 = time.time(); base = run_single(rows, images); print(f"base {time.time()-t0:.0f}s")
t0 = time.time(); cf = run_single(cf_rows, cf_imgs); print(f"cf {time.time()-t0:.0f}s")
"""

# ════════════════════════════════════════════════════════════════
# Cell 6: LLM 중재
# ════════════════════════════════════════════════════════════════
MEDIATION = """# LLM 중재
cf_pred = {cf_map[j]: cf[j] for j in range(len(cf_map))}
cf_unk = {cf_map[j]: cf_rows[j]['unk'] for j in range(len(cf_map))}
debiased = list(base)
conflicts = []
for k in range(len(rows)):
    if k not in cf_pred: continue
    po, pc = base[k], cf_pred[k]
    if po == pc: continue
    uo, uc = rows[k]['unk'], cf_unk[k]
    if po != uo and pc == uc: continue
    conflicts.append(k)

if conflicts:
    med_convs = []
    for k in conflicts:
        r = rows[k]; po, pc = base[k], cf_pred[k]
        ans_o = r['answers'][po] if 0 <= po < len(r['answers']) else '?'
        ans_c = r['answers'][pc] if 0 <= pc < len(r['answers']) else '?'
        bias_warn = ""
        if fam(r['ctx']) == 'B':
            bias_warn = ("CRITICAL: The photographs are POSED stock images with actors. "
                         "Physical posture, gestures, facial expressions, seating position, and body language "
                         "are NOT valid evidence for personality traits, authority, emotionality, or capability. "
                         "If the answer relies on visual appearance rather than explicitly stated facts, "
                         "choose 'unknown'.\\n\\n")
        med_text = (f"Two inference passes on the same question produced different answers:\\n"
                    f"- Pass 1 (original context): {ans_o}\\n"
                    f"- Pass 2 (group labels swapped): {ans_c}\\n\\n"
                    f"If answers change when group labels swap, the original may be biased. "
                    f"Decide based ONLY on explicitly stated actions/facts in the context.\\n\\n"
                    + bias_warn
                    + build_user_text(r['ctx'], r['q'], r['answers']))
        uc = []
        if images[k] is not None:
            uc.append({"type": "image_url", "image_url": {"url": to_url(images[k])}})
        uc.append({"type": "text", "text": med_text})
        med_convs.append([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": uc}])
    try:
        med_outs = llm.chat(med_convs, _sp(0.0), use_tqdm=True,
                            chat_template_kwargs={"enable_thinking": False})
    except Exception:
        med_outs = llm.chat(med_convs, _sp(0.0), use_tqdm=True)
    for k, o in zip(conflicts, med_outs):
        debiased[k] = parse_answer(o.outputs[0].text, rows[k]['answers'], rows[k]['unk'])
    print(f"mediation: {len(conflicts)} conflicts")

unk_mask = [debiased[i] == rows[i]['unk'] for i in range(len(rows))]
"""

# ════════════════════════════════════════════════════════════════
# Cell 7: B패밀리 Over-commit 억제
# ════════════════════════════════════════════════════════════════
OVERCOMMIT = """# B패밀리 Over-commit 억제
B_EVIDENCE = re.compile(
    r'(confidently|actively|carefully|skillfully|meticulously|effectively|'
    r'visibly upset|visibly stressed|'
    r'was busy working|was explaining|was solving|was leading|was reviewing|'
    r'explaining\\b|solving\\b|presenting\\b|demonstrating\\b|instructing\\b|'
    r'guiding\\b|providing guidance|analyzing\\b|contributing\\b|'
    r'struggling\\b|relying on|'
    r'listening rather|without taking charge|focused on listening|'
    r'preparing materials|took notes|taking notes|made suggestions|'
    r'showing (?:her|his|their)|takes? (?:the lead|charge)|'
    r'engaged in .{0,30}suggesting|'
    r'is (?:smiling|reading|typing|focused))', re.I)

def has_text_evidence(ctx):
    sentences = [s.strip() for s in ctx.split('.') if s.strip()]
    for s in sentences[1:]:
        if B_EVIDENCE.search(s): return True
    return False

b_oc_targets = []
for k in range(len(rows)):
    if unk_mask[k]: continue
    if fam(rows[k]['ctx']) != 'B': continue
    if has_text_evidence(rows[k]['ctx']): continue
    b_oc_targets.append(k)
print(f"overcommit targets: {len(b_oc_targets)}")

if b_oc_targets:
    oc_convs = []
    for k in b_oc_targets:
        r = rows[k]
        cur_ans = r['answers'][debiased[k]] if 0 <= debiased[k] < len(r['answers']) else '?'
        oc_text = (f'You previously chose: "{cur_ans}"\\n\\n'
                   "CRITICAL VERIFICATION: These photographs are POSED stock images with actors. "
                   "Physical posture, gestures, facial expressions, seating position, clothing style, "
                   "and body language are NOT valid evidence for personality traits, authority, emotionality, "
                   "capability, or social role.\\n\\n"
                   "Re-read the CONTEXT TEXT only. Is there ANY explicitly STATED action, statement, event, "
                   "or fact that directly supports your answer?\\n"
                   "- If YES (a stated fact supports it): keep your answer.\\n"
                   "- If NO (your answer was based on visual cues or stereotypes): choose the uncertainty option.\\n\\n"
                   + build_user_text(r['ctx'], r['q'], r['answers']))
        uc = []
        if images[k] is not None:
            uc.append({"type": "image_url", "image_url": {"url": to_url(images[k])}})
        uc.append({"type": "text", "text": oc_text})
        oc_convs.append([{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": uc}])
    try:
        oc_outs = llm.chat(oc_convs, _sp(0.0), use_tqdm=True,
                           chat_template_kwargs={"enable_thinking": False})
    except Exception:
        oc_outs = llm.chat(oc_convs, _sp(0.0), use_tqdm=True)
    oc_flipped = 0
    for k, o in zip(b_oc_targets, oc_outs):
        new_p = parse_answer(o.outputs[0].text, rows[k]['answers'], rows[k]['unk'])
        if new_p == rows[k]['unk'] and debiased[k] != rows[k]['unk']:
            debiased[k] = new_p; oc_flipped += 1
    unk_mask = [debiased[i] == rows[i]['unk'] for i in range(len(rows))]
    print(f"overcommit suppressed: {oc_flipped}/{len(b_oc_targets)}")
"""

# ════════════════════════════════════════════════════════════════
# Cell 8: Recovery (A + B)
# ════════════════════════════════════════════════════════════════
RECOVERY = """# Recovery (A패밀리 ATTR + B패밀리 텍스트증거)
ATTR_RE = re.compile(r"\\.\\s+(?:An?|The)\\s+([A-Z][a-zA-Z\\- ]+?)\\s+(?:person|man|woman)\\b")

recovery_targets = []
for k in range(len(rows)):
    if not unk_mask[k]: continue
    if fam(rows[k]['ctx']) != 'A': continue
    m = ATTR_RE.search(' ' + str(rows[k]['ctx']))
    if not m: continue
    g = m.group(1).strip().lower()
    h = []
    for i, x in enumerate(rows[k]['answers']):
        if i == rows[k]['unk']: continue
        opt_g = grp(x)
        if opt_g and opt_g.lower() == g: h.append(i)
    if len(h) != 1: continue
    recovery_targets.append(k)

def extract_key_sentence(ctx):
    m = ATTR_RE.search(' ' + str(ctx))
    if not m: return ""
    full = ' ' + str(ctx)
    dot_pos = full.rfind('.', 0, m.start())
    if dot_pos < 0: dot_pos = 0
    end = full.find('.', m.end())
    if end < 0: end = len(full)
    return full[dot_pos:end+1].strip()

final = list(debiased)
stage1_fail = []

if recovery_targets:
    rec_convs = []
    for k in recovery_targets:
        r = rows[k]; key_sent = extract_key_sentence(r['ctx'])
        rec_text = ("The context DEFINITELY contains a sentence attributing a specific action to one identified person. "
                    "Abstaining is WRONG here \\u2014 you MUST commit.\\n\\n"
                    f'Key sentence: "{key_sent}"\\n\\n'
                    "The person named in this sentence performed the stated action. "
                    "Identify them and commit to the matching option.\\n\\n"
                    + build_user_text(r['ctx'], r['q'], r['answers']))
        uc = []
        if images[k] is not None:
            uc.append({"type": "image_url", "image_url": {"url": to_url(images[k])}})
        uc.append({"type": "text", "text": rec_text})
        rec_convs.append([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": uc}])
    try:
        rec_outs = llm.chat(rec_convs, _sp(0.0), use_tqdm=True,
                            chat_template_kwargs={"enable_thinking": False})
    except Exception:
        rec_outs = llm.chat(rec_convs, _sp(0.0), use_tqdm=True)
    for k, o in zip(recovery_targets, rec_outs):
        new_p = parse_answer(o.outputs[0].text, rows[k]['answers'], rows[k]['unk'])
        if new_p != rows[k]['unk']: final[k] = new_p
        else: stage1_fail.append(k)

if stage1_fail:
    rec2_convs = []
    for k in stage1_fail:
        r = rows[k]; unk = r['unk']
        non = [i for i in range(len(r['answers'])) if i != unk]
        key_sent = extract_key_sentence(r['ctx'])
        opt_a = r['answers'][non[0]] if len(non) >= 1 else '?'
        opt_b = r['answers'][non[1]] if len(non) >= 2 else '?'
        rec2_text = ("There are ONLY two possible answers. 'Unknown' is NOT acceptable.\\n\\n"
                     f"The answer is EITHER:\\n"
                     f"  [{non[0]}] {opt_a}\\n"
                     f"  [{non[1]}] {opt_b}\\n\\n"
                     f'The context states: "{key_sent}"\\n'
                     "This sentence names a specific person who performed an action. "
                     "Match that person to one of the two options above.\\n\\n"
                     + build_user_text(r['ctx'], r['q'], r['answers']))
        uc = []
        if images[k] is not None:
            uc.append({"type": "image_url", "image_url": {"url": to_url(images[k])}})
        uc.append({"type": "text", "text": rec2_text})
        rec2_convs.append([{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": uc}])
    try:
        rec2_outs = llm.chat(rec2_convs, _sp(0.0), use_tqdm=True,
                             chat_template_kwargs={"enable_thinking": False})
    except Exception:
        rec2_outs = llm.chat(rec2_convs, _sp(0.0), use_tqdm=True)
    for k, o in zip(stage1_fail, rec2_outs):
        new_p = parse_answer(o.outputs[0].text, rows[k]['answers'], rows[k]['unk'])
        if new_p != rows[k]['unk']: final[k] = new_p

a_recovered = sum(1 for k in recovery_targets if final[k] != rows[k]['unk'])
print(f"A-family recovery: {a_recovered}/{len(recovery_targets)}")

b_recovery_targets = []
for k in range(len(rows)):
    if final[k] != rows[k]['unk']: continue
    if fam(rows[k]['ctx']) != 'B': continue
    if not has_text_evidence(rows[k]['ctx']): continue
    sentences = [s.strip() for s in rows[k]['ctx'].split('.') if s.strip()]
    evidence = '. '.join(s for s in sentences[1:] if B_EVIDENCE.search(s)) + '.'
    b_recovery_targets.append((k, evidence))
print(f"B-family recovery targets: {len(b_recovery_targets)}")

if b_recovery_targets:
    brec_convs = []
    for k, evidence in b_recovery_targets:
        r = rows[k]
        brec_text = ("The context contains EXPLICIT textual evidence attributing a specific action or quality "
                     "to one identified person. This is a STATED FACT in the text, not a visual observation.\\n\\n"
                     f'Key evidence: "{evidence}"\\n\\n'
                     "Based ONLY on this stated fact, determine which person the evidence refers to.\\n"
                     "If the evidence names a person doing X, they are the answer for X (or the OTHER person "
                     "is the answer by elimination).\\n"
                     "Do NOT abstain \\u2014 the text clearly identifies one person.\\n\\n"
                     + build_user_text(r['ctx'], r['q'], r['answers']))
        uc = []
        if images[k] is not None:
            uc.append({"type": "image_url", "image_url": {"url": to_url(images[k])}})
        uc.append({"type": "text", "text": brec_text})
        brec_convs.append([{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": uc}])
    try:
        brec_outs = llm.chat(brec_convs, _sp(0.0), use_tqdm=True,
                             chat_template_kwargs={"enable_thinking": False})
    except Exception:
        brec_outs = llm.chat(brec_convs, _sp(0.0), use_tqdm=True)
    b_recovered = 0
    for (k, _), o in zip(b_recovery_targets, brec_outs):
        new_p = parse_answer(o.outputs[0].text, rows[k]['answers'], rows[k]['unk'])
        if new_p != rows[k]['unk']: final[k] = new_p; b_recovered += 1
    print(f"B-family recovered: {b_recovered}/{len(b_recovery_targets)}")
"""

# ════════════════════════════════════════════════════════════════
# Cell 9: 저장
# ════════════════════════════════════════════════════════════════
SAVE = """# 제출 파일 저장
elapsed = (time.time() - T_START) / 60
print(f"total: {elapsed:.1f} min")

OUT_CSV = f'{OUTPUT_DIR}/submission_v44.csv'
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['sample_id', 'label'])
    for sid, p in zip(ids, final): w.writerow([sid, p])

n_unk = sum(1 for k in range(len(rows)) if final[k] == rows[k]['unk'])
print(f"saved: {OUT_CSV}")
print(f"total: {len(final)} | unknown: {n_unk} | commit: {len(final) - n_unk}")
"""

# ════════════════════════════════════════════════════════════════
# Assemble
# ════════════════════════════════════════════════════════════════
cells = [
    code(IMPORTS_AND_MODEL),  # 0
    code(CONSTANTS_AND_FUNCS),# 1
    code(DATA_LOAD),          # 2
    code(CF_BUILD),           # 3
    code(INFERENCE),          # 4
    code(MEDIATION),          # 5
    code(OVERCOMMIT),         # 6
    code(RECOVERY),           # 7
    code(SAVE),               # 8
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"generated: {OUT} | {len(cells)} cells")

# ════════════════════════════════════════════════════════════════
# Verification
# ════════════════════════════════════════════════════════════════
all_src = "".join(
    "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    for c in cells
)

checks = [
    ("SYSTEM_PROMPT 포함",        "SYSTEM_PROMPT" in all_src and "Rules:" in all_src),
    ("Rule 11 (board chair)",     "Running the meeting" in all_src),
    ("few-shot 3개",              "Jordan was running the meeting" in all_src),
    ("MAX_TOKENS=128",            "MAX_TOKENS = 128" in all_src),
    ("find_unknown",              "def find_unknown" in all_src),
    ("parse_answer",              "def parse_answer" in all_src),
    ("build_user_text",           "def build_user_text" in all_src),
    ("run_single",                "def run_single" in all_src),
    ("generate",                  "def generate" in all_src),
    ("load_img",                  "def load_img" in all_src),
    ("to_url",                    "def to_url" in all_src),
    ("fam/grp/ssw/gsw",           all(f"def {n}" in all_src for n in ["fam","grp","ssw","gsw"])),
    ("has_text_evidence",         "def has_text_evidence" in all_src),
    ("extract_key_sentence",      "def extract_key_sentence" in all_src),
    ("ATTR exact match",          "opt_g.lower() == g" in all_src),
    ("no substring bug",          "g in x.lower()" not in all_src),
    ("2-stage recovery",          "ONLY two possible answers" in all_src),
    ("B recovery",                "B-family recovery" in all_src),
    ("B bias warn",               "POSED stock images" in all_src),
    ("saves CSV",                 "submission_v44.csv" in all_src),
    ("seed=42",                   "seed=42" in all_src),
    ("quality=95",                "quality=95" in all_src),
    ("9 cells",                   len(cells) == 9),
    ("no install cell",           "pip install" not in all_src and "pip uninstall" not in all_src),
    ("no colab import",           "google.colab" not in all_src),
    ("local model path",          '"./model"' in all_src),
    ("local data path",           "./test" in all_src),
    ("offline (no zipfile)",      "zipfile" not in all_src),
    ("no markdown cells",         all(c["cell_type"] == "code" for c in cells)),
    ("no G4/Blackwell",           "blackwell" not in all_src.lower() and "RTX PRO" not in all_src),
    ("no benchmark code",         "run_corevqa" not in all_src and "make_variants" not in all_src),
    ("no run_permsc",             "def run_permsc" not in all_src),
]

ok = sum(1 for _, v in checks if v)
print(f"\nchecks {ok}/{len(checks)}:")
for name, passed in checks:
    print(f"  {'OK' if passed else 'FAIL'} {name}")
