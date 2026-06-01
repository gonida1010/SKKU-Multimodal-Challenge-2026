"""Inference pipelines.

Two strategies, same interface -> `dict[sample_id] = predicted_label`:

  * `single_pipeline`  : one BBQ-aware reasoning pass per sample (fast, strong).
  * `debate_pipeline`  : analyst -> proponent -> skeptic -> judge, each run as a
                         single BATCHED pass over all samples (so 4 model calls
                         total, not 4 per sample) for vLLM throughput. The judge
                         LLM synthesizes the final answer from the candidate
                         answers + rationales + bias audit, satisfying the
                         competition's "LLM-generated final answer" rule.

Both attach the image (unless disabled in config) and use JSON-guided decoding.
"""
from __future__ import annotations

from config import PipelineConfig
from data import Sample, load_image
from model_runner import GenRequest, VLLMRunner, parse_answer_id
from prompts import (
    ANSWER_JSON_SCHEMA,
    JUDGE_JSON_SCHEMA,
    build_analyst_prompt,
    build_judge_prompt,
    build_proponent_prompt,
    build_single_prompt,
    build_skeptic_prompt,
)


def _images_for(samples: list[Sample], img_root, cfg: PipelineConfig) -> list:
    if not cfg.image.enabled:
        return [None] * len(samples)
    out = []
    for s in samples:
        img = None
        if s.image_path:
            img = load_image(s.image_path, img_root, cfg.image.max_side)
        out.append(img)
    return out


def _requests(samples, images, prompt_fn, *extra_per_sample) -> list[GenRequest]:
    """Build GenRequests; prompt_fn(sample, *extra) -> prompt string."""
    reqs = []
    for i, (s, img) in enumerate(zip(samples, images)):
        extras = [col[i] for col in extra_per_sample]
        reqs.append(GenRequest(prompt=prompt_fn(s, *extras), image=img))
    return reqs


def _parse_all(samples, raw_texts) -> dict[str, int]:
    preds = {}
    for s, txt in zip(samples, raw_texts):
        # fall back to the abstain option on parse failure: on this metric,
        # abstaining is the safer default (it can only cost disambiguated points,
        # while a random guess risks the larger ambiguous group too).
        fb = s.unknown_idx if s.unknown_idx is not None else 0
        preds[s.sample_id] = parse_answer_id(txt, fallback=fb, n_options=len(s.answers))
    return preds


# --------------------------------------------------------------------------- #
def single_pipeline(
    samples: list[Sample], runner: VLLMRunner, img_root, cfg: PipelineConfig
) -> tuple[dict[str, int], dict]:
    images = _images_for(samples, img_root, cfg)
    reqs = _requests(samples, images, build_single_prompt)
    raw = runner.generate(reqs, cfg.sampling, ANSWER_JSON_SCHEMA)
    preds = _parse_all(samples, raw)
    return preds, {"raw": dict(zip([s.sample_id for s in samples], raw))}


def debate_pipeline(
    samples: list[Sample],
    runner: VLLMRunner,
    img_root,
    cfg: PipelineConfig,
    fast: bool = False,
) -> tuple[dict[str, int], dict]:
    """Multi-agent debate executed as staged batched passes.

    fast=True runs only analyst -> judge (2 passes) for ~2x speed; the full
    version adds the proponent/skeptic adversarial round for harder samples.
    """
    images = _images_for(samples, img_root, cfg)
    sid = [s.sample_id for s in samples]

    # Round 1: independent analysis (image attached here, where perception matters).
    analyst_reqs = _requests(samples, images, build_analyst_prompt)
    analyst = runner.generate(analyst_reqs, cfg.sampling, ANSWER_JSON_SCHEMA)

    if fast:
        judge_reqs = _requests(
            samples, images,
            lambda s, a, p, k: build_judge_prompt(s, a, p, k),
            analyst, ["(skipped)"] * len(samples), ["(skipped)"] * len(samples),
        )
        judge = runner.generate(judge_reqs, cfg.sampling, JUDGE_JSON_SCHEMA)
        preds = _parse_all(samples, judge)
        return preds, {"analyst": dict(zip(sid, analyst)),
                       "judge": dict(zip(sid, judge))}

    # Round 2: adversarial perspectives (text-only is fine; reuse images anyway).
    prop_reqs = _requests(samples, images, build_proponent_prompt, analyst)
    proponent = runner.generate(prop_reqs, cfg.sampling, ANSWER_JSON_SCHEMA)

    skep_reqs = _requests(samples, images, build_skeptic_prompt, analyst)
    skeptic = runner.generate(skep_reqs, cfg.sampling, ANSWER_JSON_SCHEMA)

    # Round 3: judge synthesizes the final answer.
    judge_reqs = _requests(
        samples, images, build_judge_prompt, analyst, proponent, skeptic
    )
    judge = runner.generate(judge_reqs, cfg.sampling, JUDGE_JSON_SCHEMA)
    preds = _parse_all(samples, judge)

    debug = {
        "analyst": dict(zip(sid, analyst)),
        "proponent": dict(zip(sid, proponent)),
        "skeptic": dict(zip(sid, skeptic)),
        "judge": dict(zip(sid, judge)),
    }
    return preds, debug


PIPELINES = {
    "single": single_pipeline,
    "debate": debate_pipeline,
}
