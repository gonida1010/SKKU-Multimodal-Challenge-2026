"""Measure Balanced Accuracy offline on the public BBQ dataset.

This is the development loop: change a prompt / pipeline / model, run this, and
read the real Balanced Accuracy plus the bias diagnostics (over-commit vs
over-abstain) so you know which group is costing you points.

    python src/evaluate.py --pipeline debate --n-per-category 60
    python src/evaluate.py --pipeline single --n-per-category 60 --with-image-blank

BBQ is text-only, so by default no image is attached (the competition's image is
a bias-distractor; text reasoning is the dominant signal and what we tune here).
"""
from __future__ import annotations

import argparse
import json
import time

from bbq_eval import balanced_accuracy, bias_diagnostics, build_val_set
from config import CFG
from model_runner import VLLMRunner
from pipelines import PIPELINES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=list(PIPELINES), default="single")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n-per-category", type=int, default=60)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump", default=None, help="write per-sample debug json here")
    args = ap.parse_args()

    if args.model:
        CFG.model.model = args.model
    # BBQ has no images; force text-only so loader doesn't look for files.
    CFG.image.enabled = False

    val = build_val_set(n_per_category=args.n_per_category, seed=args.seed)
    print(f"[eval] {len(val)} BBQ samples | pipeline={args.pipeline} "
          f"fast={args.fast} | model={CFG.model.model}")

    runner = VLLMRunner(CFG.model)
    fn = PIPELINES[args.pipeline]

    t0 = time.time()
    if args.pipeline == "debate":
        preds, debug = fn(val, runner, "", CFG, fast=args.fast)
    else:
        preds, debug = fn(val, runner, "", CFG)
    dt = time.time() - t0

    score = balanced_accuracy(val, preds)
    diag = bias_diagnostics(val, preds)
    print("\n========== RESULTS ==========")
    print(f"Balanced Accuracy : {score['balanced_accuracy']:.4f}")
    print(f"  acc_ambig       : {score['acc_ambig']:.4f}  (n={score['n_ambig']})")
    print(f"  acc_disambig    : {score['acc_disambig']:.4f}  (n={score['n_disambig']})")
    print(f"over_commit_rate  : {diag['over_commit_rate']:.4f}  "
          f"(ambig samples wrongly given a specific answer)")
    print(f"over_abstain_rate : {diag['over_abstain_rate']:.4f}  "
          f"(disambig samples wrongly abstained)")
    print(f"time              : {dt:.1f}s ({dt/max(1,len(val)):.3f}s/sample)")
    print("=============================")

    if args.dump:
        rows = []
        for s in val:
            rows.append({
                "sample_id": s.sample_id, "condition": s.condition,
                "label": s.label, "pred": preds.get(s.sample_id),
                "unknown_idx": s.unknown_idx, "question": s.question,
                "context": s.context, "answers": s.answers,
            })
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"score": score, "diag": diag, "rows": rows,
                       "debug": debug}, f, ensure_ascii=False, indent=2)
        print(f"[eval] dumped per-sample debug -> {args.dump}")


if __name__ == "__main__":
    main()
