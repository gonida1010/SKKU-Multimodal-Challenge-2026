"""Produce a competition submission CSV (sample_id,label).

Example (on the A6000 / Colab eval env):
    python src/run_inference.py --pipeline debate --data-csv open/test/test.csv \
        --images-dir open/test --output outputs/submission_debate.csv

The final answer for every sample is produced by the LLM (single reasoning pass,
or the debate judge), in compliance with the competition rules.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from config import CFG, TEST_CSV, TEST_IMG_ROOT, OUTPUT_DIR
from data import load_samples
from model_runner import VLLMRunner
from pipelines import PIPELINES


def write_submission(preds: dict[str, int], samples, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "label"])
        for s in samples:
            w.writerow([s.sample_id, preds.get(s.sample_id, 0)])
    print(f"[submission] wrote {len(samples)} rows -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=list(PIPELINES), default="debate")
    ap.add_argument("--fast", action="store_true", help="2-pass debate (analyst->judge)")
    ap.add_argument("--data-csv", default=str(TEST_CSV))
    ap.add_argument("--images-dir", default=str(TEST_IMG_ROOT))
    ap.add_argument("--output", default=str(OUTPUT_DIR / "submission.csv"))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--model", default=None, help="override CFG model id")
    ap.add_argument("--no-image", action="store_true")
    args = ap.parse_args()

    if args.model:
        CFG.model.model = args.model
    if args.no_image:
        CFG.image.enabled = False

    samples = load_samples(args.data_csv)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"[run] {len(samples)} samples | pipeline={args.pipeline} "
          f"fast={args.fast} | model={CFG.model.model}")

    runner = VLLMRunner(CFG.model)
    fn = PIPELINES[args.pipeline]

    t0 = time.time()
    if args.pipeline == "debate":
        preds, _ = fn(samples, runner, args.images_dir, CFG, fast=args.fast)
    else:
        preds, _ = fn(samples, runner, args.images_dir, CFG)
    dt = time.time() - t0
    print(f"[run] inference done in {dt:.1f}s ({dt/max(1,len(samples)):.3f}s/sample)")

    write_submission(preds, samples, Path(args.output))


if __name__ == "__main__":
    main()
