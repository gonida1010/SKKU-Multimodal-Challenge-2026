"""Offline evaluation on the original (public) BBQ dataset.

The competition is a multimodal port of BBQ (Parrish et al., 2022). The original
BBQ text data is public, unrelated to the hidden eval set, and carries the same
schema *plus* ground-truth `label` and `context_condition` (ambig/disambig).
Using it for validation is allowed (it is NOT derived from the competition's
evaluation set) and gives us the one thing we otherwise lack: a way to *measure*
Balanced Accuracy locally before submitting.

Internet is only needed here (a development tool); the final competition
inference runs fully offline.
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from config import DATA_DIR
from data import Sample, find_unknown_index

BBQ_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
]
_RAW_BASE = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/{cat}.jsonl"
_CACHE = DATA_DIR / "bbq"


def download_bbq(categories: list[str] | None = None) -> None:
    """Cache the BBQ jsonl files locally (skips already-downloaded ones)."""
    categories = categories or BBQ_CATEGORIES
    _CACHE.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        dst = _CACHE / f"{cat}.jsonl"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        url = _RAW_BASE.format(cat=cat)
        print(f"[bbq] downloading {cat} ...")
        urllib.request.urlretrieve(url, dst)


def _load_category(cat: str) -> list[dict]:
    path = _CACHE / f"{cat}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _entry_to_sample(e: dict) -> Sample:
    answers = [e["ans0"], e["ans1"], e["ans2"]]
    # BBQ marks the unknown choice in answer_info as ["...", "unknown"]; use it as
    # ground truth for unknown_idx, falling back to our detector.
    unk = None
    ainfo = e.get("answer_info", {})
    for i, key in enumerate(("ans0", "ans1", "ans2")):
        meta = ainfo.get(key, [])
        if len(meta) >= 2 and meta[1] == "unknown":
            unk = i
            break
    if unk is None:
        unk = find_unknown_index(answers)
    return Sample(
        sample_id=f"{e['category']}_{e['example_id']}",
        image_path="",
        context=e["context"],
        question=e["question"],
        answers=answers,
        unknown_idx=unk,
        label=int(e["label"]),
        condition=e["context_condition"],   # "ambig" | "disambig"
    )


def build_val_set(
    n_per_category: int | None = 100,
    categories: list[str] | None = None,
    seed: int = 42,
    balance_condition: bool = True,
) -> list[Sample]:
    """Build a balanced validation set of BBQ Samples.

    n_per_category: cap per category (None = all). With balance_condition we draw
    half ambiguous / half disambiguated so the local metric mirrors the
    competition's Balanced Accuracy split.
    """
    download_bbq(categories)
    categories = categories or BBQ_CATEGORIES
    rng = random.Random(seed)
    out: list[Sample] = []
    for cat in categories:
        entries = [_entry_to_sample(e) for e in _load_category(cat)]
        if n_per_category is None:
            out.extend(entries)
            continue
        if balance_condition:
            ambig = [s for s in entries if s.condition == "ambig"]
            disambig = [s for s in entries if s.condition == "disambig"]
            rng.shuffle(ambig)
            rng.shuffle(disambig)
            half = n_per_category // 2
            out.extend(ambig[:half])
            out.extend(disambig[: n_per_category - half])
        else:
            rng.shuffle(entries)
            out.extend(entries[:n_per_category])
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Metric: Balanced Accuracy as defined by the competition
# --------------------------------------------------------------------------- #
def balanced_accuracy(samples: list[Sample], preds: dict[str, int]) -> dict:
    """Mean of accuracy on ambiguous and disambiguated groups.

    `preds` maps sample_id -> predicted label (0/1/2). Samples must carry
    `condition` and `label`. Returns a dict with the overall balanced score and
    the per-group breakdown so we can see *where* we lose points.
    """
    groups = {"ambig": [0, 0], "disambig": [0, 0]}  # [correct, total]
    for s in samples:
        if s.condition is None or s.label is None:
            continue
        pred = preds.get(s.sample_id)
        if pred is None:
            continue
        g = groups[s.condition]
        g[1] += 1
        if pred == s.label:
            g[0] += 1
    acc = {}
    for k, (c, t) in groups.items():
        acc[k] = c / t if t else 0.0
    overall = (acc["ambig"] + acc["disambig"]) / 2
    return {
        "balanced_accuracy": overall,
        "acc_ambig": acc["ambig"],
        "acc_disambig": acc["disambig"],
        "n_ambig": groups["ambig"][1],
        "n_disambig": groups["disambig"][1],
    }


# Extra diagnostics that explain *why* the balanced score moves -------------- #
def bias_diagnostics(samples: list[Sample], preds: dict[str, int]) -> dict:
    """How often, when WRONG, did the model pick a non-unknown (over-committing)
    vs the unknown (over-abstaining)? Helps tune the abstention calibration."""
    over_commit = 0   # ambiguous sample, picked a specific entity (should abstain)
    over_abstain = 0  # disambiguated sample, picked unknown (should commit)
    n_ambig = n_disambig = 0
    for s in samples:
        pred = preds.get(s.sample_id)
        if pred is None or s.condition is None:
            continue
        if s.condition == "ambig":
            n_ambig += 1
            if pred != s.unknown_idx:
                over_commit += 1
        else:
            n_disambig += 1
            if pred == s.unknown_idx:
                over_abstain += 1
    return {
        "over_commit_rate": over_commit / n_ambig if n_ambig else 0.0,
        "over_abstain_rate": over_abstain / n_disambig if n_disambig else 0.0,
    }
