"""Data loading + the robust 'unknown option' detector.

Every sample in this competition is a 3-way multiple choice where exactly one
choice is an "unknown / not enough information" answer (the BBQ convention).
Reliably identifying *which* of the three choices is the unknown one is the
backbone of the whole strategy:

  * On AMBIGUOUS samples the correct label IS the unknown option.
  * On DISAMBIGUATED samples the correct label is one of the other two.

We do NOT use this detector to *decide* the answer (that would violate the
"final answer must be LLM-generated" rule). We use it to (a) tell the LLM which
option means "abstain", and (b) build/validate our offline metric. The LLM still
produces the final choice.

Stdlib-only so it can be unit-tested without torch/pandas/PIL.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Unknown-option detection
# --------------------------------------------------------------------------- #
# Phrases (substring, case-insensitive) that mark a choice as the "unknown" one.
# Curated to cover the full variety seen in BBQ-style data.
_UNKNOWN_PATTERNS = [
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "can't determine",
    "cannot be known",
    "can't be known",
    "cannot answer",
    "can't answer",
    "cannot be answered",
    "not answerable",
    "not enough info",
    "not enough information",
    "not enough data",
    "no enough info",
    "insufficient information",
    "insufficient info",
    "not known",
    "unknown",
    "undetermined",
    "cannot say",
    "can't say",
    "cannot tell",
    "can't tell",
    "hard to determine",
    "difficult to determine",
    "impossible to determine",
    "impossible to tell",
    "not specified",
    "unspecified",
    "no information",
    "not provided",
    "not clear",
    "unclear",
]

# A couple of regex fallbacks for "cannot <verb>" / "can't <verb>" phrasings we
# might not have enumerated.
_UNKNOWN_REGEX = re.compile(
    r"\b(can(?:not|'t)\s+be\s+\w+|can(?:not|'t)\s+\w+|not\s+enough|"
    r"un(?:known|clear|determined|specified)|insufficient|undetermined)\b",
    re.IGNORECASE,
)


def is_unknown_choice(text: str) -> bool:
    """Return True if a single answer choice means 'unknown / not enough info'."""
    s = text.lower().strip()
    if any(p in s for p in _UNKNOWN_PATTERNS):
        return True
    return bool(_UNKNOWN_REGEX.search(s))


def find_unknown_index(answers: list[str]) -> int | None:
    """Index of the unknown choice, or None if not exactly one is found.

    We require exactly one match to be confident. If zero or many match we return
    None and let the caller fall back (the LLM can still reason without it).
    """
    flags = [is_unknown_choice(a) for a in answers]
    if sum(flags) == 1:
        return flags.index(True)
    # Ambiguity: prefer the shortest matching candidate that also looks generic
    # (entity names rarely match the patterns, so multi-match is very rare).
    matches = [i for i, f in enumerate(flags) if f]
    if len(matches) > 1:
        # pick the one whose text is the most "abstain-like" (shortest, no name)
        return min(matches, key=lambda i: len(answers[i]))
    return None


# --------------------------------------------------------------------------- #
# Sample container
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    sample_id: str
    image_path: str          # as stored in csv, e.g. "./images/test_img_0000.jpg"
    context: str
    question: str
    answers: list[str]       # exactly 3
    unknown_idx: int | None  # index of the unknown choice
    label: int | None = None # ground truth if available (train/val)
    # offline-eval-only metadata (never from competition test set):
    condition: str | None = None   # "ambig" | "disambig" for BBQ val

    @property
    def non_unknown_indices(self) -> list[int]:
        return [i for i in range(len(self.answers)) if i != self.unknown_idx]


def _parse_answers(raw: str) -> list[str]:
    ans = json.loads(raw)
    if not isinstance(ans, list):
        raise ValueError(f"answers is not a list: {raw!r}")
    return [str(a) for a in ans]


def load_samples(csv_path: str | "os.PathLike") -> list[Sample]:
    """Load a competition-format CSV (test.csv / train.csv) into Samples."""
    samples: list[Sample] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answers = _parse_answers(row["answers"])
            label = None
            if row.get("label") not in (None, ""):
                label = int(row["label"])
            samples.append(
                Sample(
                    sample_id=row["sample_id"],
                    image_path=row["image_path"],
                    context=row.get("context", "") or "",
                    question=row.get("question", "") or "",
                    answers=answers,
                    unknown_idx=find_unknown_index(answers),
                    label=label,
                )
            )
    return samples


# --------------------------------------------------------------------------- #
# Image loading (imports PIL lazily so this module stays stdlib-importable)
# --------------------------------------------------------------------------- #
def load_image(image_path: str, root: str | "os.PathLike", max_side: int = 512):
    """Open + RGB + resize longest side to max_side. Returns a PIL.Image or None."""
    from pathlib import Path

    from PIL import Image

    p = Path(root) / str(image_path)
    try:
        img = Image.open(p).convert("RGB")
    except Exception as e:  # noqa: BLE001
        print(f"[load_image] failed {p}: {e}")
        return None
    w, h = img.size
    scale = max_side / float(max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img
