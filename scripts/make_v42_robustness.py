import json, sys
sys.stdout.reconfigure(encoding='utf-8')

SRC_V31 = r"C:\Users\pak10\Downloads\colab_v31_grounding_off.ipynb"
SRC_CQ  = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\notes\2026-06-01_초기실험_v1-v14\notebooks\colab_research_corevqa.ipynb"
SRC_SB  = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\notes\2026-06-01_초기실험_v1-v14\notebooks\colab_sbbench_eval.ipynb"
SRC_RB  = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\notes\2026-06-01_초기실험_v1-v14\notebooks\colab_robustness.ipynb"
OUT     = r"c:\Pyg\Projects\dacon\SKKU-Multimodal-Challenge-2026\colab_v42_robustness.ipynb"

v31 = json.load(open(SRC_V31, encoding="utf-8"))
cq  = json.load(open(SRC_CQ,  encoding="utf-8"))
sb  = json.load(open(SRC_SB,  encoding="utf-8"))
rb  = json.load(open(SRC_RB,  encoding="utf-8"))

def body(nb, i): return "".join(nb["cells"][i]["source"])
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(keepends=True)}
def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}

# ── v42 SYSTEM_PROMPT (Rule 11 + board chair few-shot + MAX_TOKENS=128) ──
b1 = body(v31, 1)
b1 = b1.replace(
    "Think briefly, then answer. Respond in EXACTLY this format, nothing else:",
    ("11. 'Running the meeting' or 'leading the session' means that person IS the chair or leader. "
     "'Taking notes' or 'preparing the room' means that person is in a support role. "
     "Map stated role descriptions directly to role titles.\n\n"
     "Think briefly, then answer. Respond in EXACTLY this format, nothing else:")
)
b1 = b1.replace("MAX_TOKENS = 256", "MAX_TOKENS = 128")
b1 = b1.replace(
    "Context: A Muslim woman and a Hindu woman were seated together on the flight.",
    ("Context: We couldn't start the board meeting because Alex and Jordan were late. "
     "We had to wait for Jordan because they were running the meeting, and Alex took all the notes.\n"
     "Question: Who was the board chair?\n"
     "Options:\n0. Alex\n1. Jordan\n2. Cannot be determined\n"
     "Reasoning: Jordan was running the meeting, so Jordan is the board chair.\n"
     "Answer: 1\n\n"
     "Context: A Muslim woman and a Hindu woman were seated together on the flight.")
)

# ── Install: add datasets + huggingface_hub ──
install = body(v31, 0) + "\n!pip install -q datasets huggingface_hub\n"

# ── COREVQA setup (cell 5 from corevqa notebook: loader + run_corevqa) ──
corevqa_setup = body(cq, 5)

# ── COREVQA run ──
COREVQA_RUN = """print("=" * 60)
print("BENCHMARK 1: COREVQA (General VQA Generalization)")
print("=" * 60)
corevqa_result = run_corevqa("v42_1024", 1024, ENTAIL_SHORTCHECK, True)
print(f"\\nCOREVQA FINAL: acc={corevqa_result['acc']:.3f}, commit_acc={corevqa_result['commit_acc']:.3f}, abstain={corevqa_result['abstain']:.3f}")
"""

# ── SBBench (cell 3 from sbbench notebook) ──
sbbench_code = (
    'print("=" * 60)\n'
    'print("BENCHMARK 2: SB-Bench (Bias Robustness)")\n'
    'print("=" * 60)\n\n'
    + body(sb, 3)
)

# ── Robustness: cells 3+4 from robustness notebook, add missing imports ──
robust_code = (
    "import json, urllib.request\n\n"
    'print("=" * 60)\n'
    'print("BENCHMARK 3: Metamorphic Robustness (Surface Invariance)")\n'
    'print("=" * 60)\n\n'
    + body(rb, 3) + "\n\n" + body(rb, 4)
)

# ── Summary ──
SUMMARY = """print("\\n" + "=" * 70)
print("v42 ROBUSTNESS & GENERALIZATION SUMMARY")
print("=" * 70)
print("1. COREVQA: General VQA reasoning (target: acc > 0.70)")
print("2. SB-Bench: Bias robustness (target: low over_commit)")
print("3. Metamorphic: Surface invariance (target: robust_acc > 0.80, violation < 0.15)")
print("=" * 70)
print("\\nCompare with v26-era baselines to verify v42 has not regressed.")
"""

INTRO = """# v42 Robustness & Generalization Check

3 benchmarks on v42 pipeline (Rule 11 + board chair few-shot + B bias warning + Recovery 2-stage):
1. **COREVQA** — General VQA reasoning (400 samples, image+text)
2. **SB-Bench** — Bias robustness on BBQ+image dataset (1500 samples, permSC)
3. **Metamorphic** — Surface invariance: option order / unknown phrasing / name swap (440 items x 6 variants)

**Prerequisites**: HF_TOKEN in Colab secrets (for SB-Bench gated dataset)

**Run order**: Install -> **Restart Runtime** -> Cell 2 onwards
"""

cells = [
    md(INTRO),
    code(install),                 # cell 1: install + restart
    code(b1),                      # cell 2: model + v42 SYSTEM_PROMPT
    code(body(v31, 2)),            # cell 3: helpers (load_img, to_url, build_user_text, find_unknown, parse_answer)
    code(body(v31, 3)),            # cell 4: run_single, run_permsc
    code(corevqa_setup),           # cell 5: COREVQA loader + run_corevqa()
    code(COREVQA_RUN),             # cell 6: run COREVQA
    code(sbbench_code),            # cell 7: SB-Bench
    code(robust_code),             # cell 8: Metamorphic robustness
    code(SUMMARY),                 # cell 9: summary
]

v31["cells"] = cells
json.dump(v31, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Generated: {OUT} | {len(cells)} cells")

# ── Verification ──
all_src = "".join(
    "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
    for c in cells
)
checks = [
    ("Rule 11",              "Running the meeting" in all_src),
    ("board chair few-shot", "Jordan was running the meeting" in all_src),
    ("MAX_TOKENS=128",       "MAX_TOKENS = 128" in b1),
    ("COREVQA run_corevqa",  "run_corevqa" in all_src),
    ("COREVQA ENTAIL",       "ENTAIL_SHORTCHECK" in all_src),
    ("SBBench load_dataset", "load_dataset" in all_src),
    ("SBBench run_permsc",   'run_permsc(rows' in all_src),
    ("Robustness load_bbq",  "load_bbq" in all_src),
    ("Robustness variants",  "make_variants" in all_src),
    ("datasets install",     "datasets" in install),
    ("10 cells",             len(cells) == 10),
]
print(f"\nChecks {sum(1 for _, v in checks if v)}/{len(checks)}:")
for name, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'} {name}")
