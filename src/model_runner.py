"""vLLM wrapper: load a (multimodal) LLM once, run batched chat generation with
JSON-guided decoding, and parse the structured output robustly.

A single served model is reused for every role in the debate pipeline (analyst,
proponent, skeptic, judge) -- we only ever change the prompt, so there is never
more than one set of weights in VRAM. That is what keeps the MoE-A3B model
comfortably inside 48 GB even with multi-agent debate.

Requires: vllm, torch, transformers, pillow. Runs on the A6000 eval env / Colab.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from io import BytesIO

from config import ModelConfig, SamplingConfig


def _pil_to_data_url(img, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


@dataclass
class GenRequest:
    """One generation request: a text prompt and an optional PIL image."""
    prompt: str
    image: object | None = None   # PIL.Image or None


class VLLMRunner:
    def __init__(self, mcfg: ModelConfig):
        from vllm import LLM
        self.mcfg = mcfg
        kwargs = dict(
            model=mcfg.model,
            dtype=mcfg.dtype,
            max_model_len=mcfg.max_model_len,
            gpu_memory_utilization=mcfg.gpu_memory_utilization,
            tensor_parallel_size=mcfg.tensor_parallel_size,
            limit_mm_per_prompt=mcfg.limit_mm_per_prompt,
            trust_remote_code=mcfg.trust_remote_code,
            seed=mcfg.seed,
        )
        if mcfg.quantization:
            kwargs["quantization"] = mcfg.quantization
        self.llm = LLM(**kwargs)

    def _sampling_params(self, scfg: SamplingConfig, json_schema: dict | None):
        from vllm import SamplingParams
        guided = None
        if json_schema is not None:
            from vllm.sampling_params import GuidedDecodingParams
            guided = GuidedDecodingParams(json=json_schema)
        return SamplingParams(
            temperature=scfg.temperature,
            top_p=scfg.top_p,
            max_tokens=scfg.max_tokens,
            seed=scfg.seed,
            guided_decoding=guided,
        )

    @staticmethod
    def _to_messages(req: GenRequest) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": req.prompt}]
        if req.image is not None:
            content.append(
                {"type": "image_url",
                 "image_url": {"url": _pil_to_data_url(req.image)}}
            )
        return [{"role": "user", "content": content}]

    def generate(
        self,
        requests: list[GenRequest],
        scfg: SamplingConfig,
        json_schema: dict | None = None,
    ) -> list[str]:
        """Batched generation. Returns the raw text of each completion."""
        sp = self._sampling_params(scfg, json_schema)
        conversations = [self._to_messages(r) for r in requests]
        outputs = self.llm.chat(conversations, sp, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Pull the last valid JSON object out of a model completion.

    Handles thinking traces, ```json fences, and trailing prose. Returns None if
    nothing parses.
    """
    if not text:
        return None
    # Strip a thinking block if present (Qwen *-Thinking emits <think>...</think>).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    candidates = []
    # fenced blocks first
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        candidates.append(m.group(1))
    # any brace-balanced object
    for m in _JSON_OBJ_RE.finditer(text):
        candidates.append(m.group(0))
    # try from the last (most likely the final answer) backwards
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            # try to salvage by trimming to the last closing brace
            try:
                trimmed = cand[: cand.rindex("}") + 1]
                return json.loads(trimmed)
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def parse_answer_id(text: str, fallback: int = 0, n_options: int = 3) -> int:
    """Parse answer_id (0/1/2) from a completion, with graceful fallbacks."""
    obj = extract_json(text)
    if obj is not None and "answer_id" in obj:
        try:
            v = int(str(obj["answer_id"]).strip())
            if 0 <= v < n_options:
                return v
        except (ValueError, TypeError):
            pass
    # last-ditch: find a lone 0/1/2 near an "answer" mention
    m = re.search(r"answer[_\s]*id\D{0,5}([0-2])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return fallback
