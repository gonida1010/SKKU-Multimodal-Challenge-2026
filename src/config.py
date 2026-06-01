"""Central configuration for the SKKU Multimodal AI Challenge 2026 pipeline.

The task is a *multimodal BBQ* (Bias Benchmark for QA): each sample has an image,
a text context, a question, and exactly three answer choices, one of which is an
"unknown / not enough information" option. The metric is **Balanced Accuracy**:
the mean of accuracy on AMBIGUOUS samples (correct answer is always the unknown
option) and DISAMBIGUATED samples (correct answer is the entity the context
explicitly supports). Whether a sample is ambiguous is NOT disclosed.

Everything tunable lives here so model / pipeline swaps are a one-line change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
OPEN_DIR = ROOT / "open"
TEST_CSV = OPEN_DIR / "test" / "test.csv"
TEST_IMG_ROOT = OPEN_DIR / "test"          # image_path is "./images/xxx.jpg" relative to this
TRAIN_CSV = OPEN_DIR / "train" / "train.csv"
TRAIN_IMG_ROOT = OPEN_DIR / "train"
SAMPLE_SUBMISSION = OPEN_DIR / "sample_submission.csv"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"                   # cached BBQ / synthetic data


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Which weights to load and how. Swap `model` for a different checkpoint.

    Rule compliance: weights must have been publicly released before 2026-06-01.
      - Qwen/Qwen3-VL-30B-A3B-Thinking   (2025-10, MoE 31B total / 3B active)  <- default
      - Qwen/Qwen3-VL-30B-A3B-Instruct   (2025-10, faster, no thinking trace)
      - Qwen/Qwen3.5-35B-A3B             (2026-02, native multimodal, stronger vision)
      - Qwen/Qwen3-VL-8B-Thinking / -4B  (small, for Colab development)
    On Ampere (RTX A6000) FP8 is not hardware-accelerated; prefer AWQ/GPTQ INT4
    community quants when memory is tight, or bf16 if it fits.
    """
    model: str = "Qwen/Qwen3-VL-30B-A3B-Thinking"
    # Set to a quantized repo (e.g. an -AWQ variant) or None for the base weights.
    quantization: str | None = None          # e.g. "awq", "gptq", "fp8"
    dtype: str = "bfloat16"
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = int(os.environ.get("TP_SIZE", "1"))
    limit_mm_per_prompt: dict = field(default_factory=lambda: {"image": 1})
    # "thinking" models emit a reasoning trace before the answer; we let them.
    is_thinking: bool = True
    trust_remote_code: bool = True
    seed: int = 42


@dataclass
class ImageConfig:
    """Image preprocessing. The image is mostly a bias-distractor in this task,
    so we keep it small/cheap. Larger only if visual disambiguation is needed."""
    enabled: bool = True            # set False to run a pure-text ablation
    max_side: int = 512             # longest side after resize
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28


@dataclass
class SamplingConfig:
    temperature: float = 0.0        # deterministic for the final judge
    top_p: float = 1.0
    max_tokens: int = 1024          # room for a thinking trace + JSON
    seed: int = 42


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    batch_size: int = 64
    # Self-consistency samples for the analyst stage (1 = off). Final answer is
    # still synthesized by the judge LLM, never a bare majority vote.
    n_analyst_samples: int = 1


CFG = PipelineConfig()
