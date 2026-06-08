#!/usr/bin/env python3
"""Run the COREVQA visual-hard lab locally with Transformers.

This is the local Mac/MPS counterpart to colab_visualhard_lab.ipynb. It is for
generalization experiments, not final CUDA/vLLM submission generation.

Examples:
  python scripts/run_corevqa_local_transformers.py --experiment format_768 --limit 50
  python scripts/run_corevqa_local_transformers.py --experiment v16_clause --limit 50
  python scripts/run_corevqa_local_transformers.py --experiment both --limit 400
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import random
import re
import time
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

hf_hub_download = None
Image = None
tqdm = None


REPO = "COREVQA2025/COREVQA"
OPTS = ["True", "False", "Cannot be determined from the image"]
UNK = 2

TAGRX = {
    "counting": r"at least|exactly|\b(one|two|three|four|five|\d+)\b|single|no one|none",
    "spatial": r"left|right|behind|front|next to|between|foreground|background|center|near|far",
    "negation": r"not|no\b|none|without|n't|neither|nobody",
    "small_object": r"glasses|hat|watch|ring|logo|phone|camera|tie|bag|umbrella|bottle",
    "action_pose": r"holding|pointing|looking|sitting|lying|standing|walking|laughing|crossed arms",
    "complex": r"although|while|but|rather than|suggesting|whereas",
    "compound": r"\band\b|\bor\b",
}


ENTAIL_BASIC = """You are a careful, literal visual reasoning expert. You see an image of a (often crowded) real scene and a STATEMENT. Decide, using ONLY what is visibly verifiable, whether it is true or false.
Rules: judge ONLY from what is visible; 0=True if all asserted is clearly supported; 1=False if any part contradicts the image; 2=Cannot be determined ONLY if the image genuinely lacks the info (occlusion/not shown), else prefer 0/1.
Respond EXACTLY:
Reasoning: <one short sentence>
Answer: <0, 1, or 2>"""


CLAUSE_SYSTEM = """You are a strict visual claim verifier. You see an image and one STATEMENT about it.
Break the statement into the smallest visual checks needed to decide it. For each check, mark exactly one status: SUPPORTED, CONTRADICTED, or NOT_VISIBLE.

Rules:
- Use ONLY visible evidence from the image.
- For no / none / neither / not a single claims, mark SUPPORTED only when the relevant people/area are visible enough to verify absence.
- For only claims, check BOTH parts: the named target has the property, and no other relevant target has it.
- For counts, spatial relations, left/right/front/behind, small objects, hands, gaze, clothing, and occlusion, be literal.
- Do NOT infer intent, future events, personality, social role, or what probably happened before/after the photo.
- Do NOT give the final True/False answer.

Respond EXACTLY:
Checks:
- <atomic visual check>: SUPPORTED|CONTRADICTED|NOT_VISIBLE -- <brief visible evidence>
"""


def ensure_runtime_deps() -> None:
    """Import heavy/optional deps after argparse so --help works before install."""
    global hf_hub_download, Image, tqdm
    if hf_hub_download is not None:
        return
    try:
        from huggingface_hub import hf_hub_download as _hf_hub_download
        from PIL import Image as _Image
        from tqdm.auto import tqdm as _tqdm
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing local runtime dependencies. Install them with:\n"
            "  python -m pip install -r requirements-local-mps.txt\n"
            f"Original error: {e}"
        ) from e
    hf_hub_download = _hf_hub_download
    Image = _Image
    tqdm = _tqdm


CLAUSE_JUDGE_SYSTEM = """You are a strict logical judge. You get a STATEMENT and visual checks produced from the image. Decide whether the original statement is True, False, or Cannot be determined.

Decision rules:
- Answer 1 (False) if ANY required check is CONTRADICTED.
- Answer 2 (Cannot be determined) if NO required check is contradicted but at least one required check is NOT_VISIBLE.
- Answer 0 (True) only if every required check is SUPPORTED.
- For statements with no/none/neither/only, all absence and uniqueness checks must be supported for True.
- Do not use priors or guesses. Use the visual checks only.

Respond EXACTLY:
Reasoning: <one short sentence>
Answer: <0, 1, or 2>
"""


ANS_RE = re.compile(r"answer\s*[:\-]?\s*\**\s*([012])", re.I)
DIG_RE = re.compile(r"\b([012])\b")


def tag_statement(text: str) -> list[str]:
    low = text.lower()
    return [tag for tag, rx in TAGRX.items() if re.search(rx, low)] or ["untagged"]


def parse_answer(text: str) -> int:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    matches = list(ANS_RE.finditer(text))
    if matches:
        return int(matches[-1].group(1))
    digits = list(DIG_RE.finditer(text))
    if digits:
        return int(digits[-1].group(1))
    return UNK


def reasoning(raw: str) -> str:
    return re.split(r"answer\s*[:\-]", raw or "", flags=re.I)[0].strip()[:1000]


def resize_image(path: Path, long_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    scale = long_side / max(img.size)
    if scale < 1:
        img = img.resize((max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))), Image.LANCZOS)
    return img


def ensure_corevqa(data_dir: Path) -> tuple[list[dict], dict[str, Path]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = hf_hub_download(REPO, "COREVQA.csv", repo_type="dataset", local_dir=str(data_dir))
    with open(csv_path, encoding="utf-8-sig") as f:
        meta = list(csv.DictReader(f))

    img_dir = data_dir / "corevqa_imgs"
    existing = list(img_dir.glob("**/*.jpg"))
    if len(existing) < 9000:
        print(f"COREVQA images found={len(existing)} -> downloading/extracting zips")
        img_dir.mkdir(parents=True, exist_ok=True)
        for zf in ["CrowdHuman_train01.zip", "CrowdHuman_train02.zip"]:
            zp = hf_hub_download(REPO, zf, repo_type="dataset", local_dir=str(data_dir))
            print("extract:", zf)
            with zipfile.ZipFile(zp) as z:
                z.extractall(img_dir)
    index = {p.name: p for p in img_dir.glob("**/*.jpg")}
    print("COREVQA images indexed:", len(index))
    return meta, index


def find_image(image_id: str, index: dict[str, Path]) -> Path | None:
    for key in (image_id.strip(), image_id.split(",")[-1].strip()):
        if key in index:
            return index[key]
    return None


def fixed_samples(meta: list[dict], index: dict[str, Path], limit: int, seed: int) -> list[dict]:
    rows = list(meta)
    random.Random(seed).shuffle(rows)
    out: list[dict] = []
    for row in rows:
        if len(out) >= limit:
            break
        path = find_image(row["image_id"], index)
        ans = row["answer"].strip().upper()
        if path is None or ans not in {"TRUE", "FALSE"}:
            continue
        try:
            size = Image.open(path).size
        except Exception:
            continue
        out.append(
            {
                "image_id": row["image_id"],
                "image_path": str(path),
                "statement": row["question"].strip(),
                "gold": 0 if ans == "TRUE" else 1,
                "image_size": f"{size[0]}x{size[1]}",
            }
        )
    if len(out) < min(50, limit):
        raise RuntimeError(f"Too few matched COREVQA samples: {len(out)}")
    return out


def build_corevqa_user(statement: str) -> str:
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(OPTS))
    return (
        f'Statement to verify: "{statement}"\n'
        "Task: Decide whether the statement is TRUE or FALSE based ONLY on what is visible in the image.\n"
        f"Options:\n{opts}"
    )


def clause_user(statement: str) -> str:
    return (
        f'STATEMENT: "{statement}"\n\n'
        "List the atomic visual checks needed to verify this statement. Do not answer True/False."
    )


def judge_user(statement: str, checks: str) -> str:
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(OPTS))
    return (
        f'STATEMENT: "{statement}"\n\nVISUAL CHECKS:\n{checks}\n\n'
        f"Options:\n{opts}\n\nBased only on the visual checks, which option is correct?"
    )


class LocalVLM:
    def __init__(self, model_id: str, device: str, dtype: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device == "auto":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if dtype == "auto":
            torch_dtype = torch.float16 if device == "mps" else torch.float32
        else:
            torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"loaded {model_id} on {self.device} dtype={torch_dtype}")

    def generate_one(self, system_prompt: str, user_text: str, image: Image.Image | None, max_new_tokens: int) -> str:
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            kwargs = {"text": [text], "return_tensors": "pt"}
            if image is not None:
                kwargs["images"] = [image]
            inputs = self.processor(**kwargs)

        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        input_len = inputs["input_ids"].shape[-1]
        gen = out[:, input_len:]
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        if self.device.type == "mps":
            self.torch.mps.empty_cache()
        return text


def metric_pack(preds: list[int], gold: list[int]) -> dict:
    n = len(gold)

    def lab_acc(label: int) -> float:
        idx = [i for i, g in enumerate(gold) if g == label]
        return sum(preds[i] == gold[i] for i in idx) / max(1, len(idx))

    true_acc = lab_acc(0)
    false_acc = lab_acc(1)
    commit = [i for i, p in enumerate(preds) if p != UNK]
    return {
        "acc": sum(p == g for p, g in zip(preds, gold)) / n,
        "ba": (true_acc + false_acc) / 2,
        "true_acc": true_acc,
        "false_acc": false_acc,
        "commit_acc": sum(preds[i] == gold[i] for i in commit) / max(1, len(commit)),
        "abstain": 1 - len(commit) / n,
        "pred_counts": Counter(preds),
    }


def write_wrong_html(rows: list[dict], preds: list[int], raw: list[str], out_path: Path, limit: int = 100) -> None:
    wrong = [i for i, r in enumerate(rows) if preds[i] != r["gold"]][:limit]
    html = ["<html><meta charset='utf-8'><body style='font-family:sans-serif'>", f"<h2>{out_path.stem}: wrong {len(wrong)}</h2>"]
    for i in wrong:
        img = resize_image(Path(rows[i]["image_path"]), 256)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        gl = "True" if rows[i]["gold"] == 0 else "False"
        html.append(
            "<div style='border-bottom:1px solid #ccc;padding:8px'>"
            f"<img src='data:image/jpeg;base64,{b64}' style='float:left;margin-right:10px'>"
            f"<b>gold:</b> {gl} | <b>pred:</b> {OPTS[preds[i]]} | <b>tags:</b> {'|'.join(tag_statement(rows[i]['statement']))}<br>"
            f"<b>stmt:</b> {rows[i]['statement']}<br><b>reasoning:</b><pre>{reasoning(raw[i])}</pre>"
            "<div style='clear:both'></div></div>"
        )
    html.append("</body></html>")
    out_path.write_text("\n".join(html), encoding="utf-8")


def run_format(vlm: LocalVLM, rows: list[dict], out_dir: Path, long_side: int, max_tokens: int) -> Path:
    exp = f"format_{long_side}"
    raw_img: list[str] = []
    raw_txt: list[str] = []
    preds_img: list[int] = []
    preds_txt: list[int] = []
    start = time.time()
    for r in tqdm(rows, desc=exp):
        img = resize_image(Path(r["image_path"]), long_side)
        user = build_corevqa_user(r["statement"])
        ri = vlm.generate_one(ENTAIL_BASIC, user, img, max_tokens)
        rt = vlm.generate_one(ENTAIL_BASIC, user, None, max_tokens)
        raw_img.append(ri)
        raw_txt.append(rt)
        preds_img.append(parse_answer(ri))
        preds_txt.append(parse_answer(rt))
    elapsed = time.time() - start
    m = metric_pack(preds_img, [r["gold"] for r in rows])
    print_metric(exp, m, elapsed / len(rows))

    out_path = out_dir / f"corevqa_{exp}.csv"
    write_core_csv(out_path, rows, preds_img, preds_txt, raw_img, raw_txt, long_side, exp)
    write_wrong_html(rows, preds_img, raw_img, out_dir / f"corevqa_{exp}_wrong.html")
    return out_path


def write_core_csv(
    out_path: Path,
    rows: list[dict],
    preds_img: list[int],
    preds_txt: list[int],
    raw_img: list[str],
    raw_txt: list[str],
    long_side: int,
    exp: str,
) -> None:
    fields = [
        "image_id",
        "image_path",
        "statement",
        "gold_label",
        "pred_img",
        "pred_txt",
        "raw_output_img",
        "raw_output_txt",
        "reasoning_img",
        "reasoning_txt",
        "correct_img",
        "correct_txt",
        "auto_tags",
        "image_size",
        "resize_long_side",
        "experiment_name",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, pi, pt, ri, rt in zip(rows, preds_img, preds_txt, raw_img, raw_txt):
            w.writerow(
                {
                    "image_id": r["image_id"],
                    "image_path": r["image_path"],
                    "statement": r["statement"],
                    "gold_label": r["gold"],
                    "pred_img": pi,
                    "pred_txt": pt,
                    "raw_output_img": ri,
                    "raw_output_txt": rt,
                    "reasoning_img": reasoning(ri),
                    "reasoning_txt": reasoning(rt),
                    "correct_img": int(pi == r["gold"]),
                    "correct_txt": int(pt == r["gold"]),
                    "auto_tags": "|".join(tag_statement(r["statement"])),
                    "image_size": r["image_size"],
                    "resize_long_side": long_side,
                    "experiment_name": exp,
                }
            )


def run_clause(vlm: LocalVLM, rows: list[dict], out_dir: Path, long_side: int) -> Path:
    exp = f"v16_clause_{long_side}"
    checks: list[str] = []
    raw_final: list[str] = []
    preds: list[int] = []
    start = time.time()
    for r in tqdm(rows, desc=exp):
        img = resize_image(Path(r["image_path"]), long_side)
        chk = vlm.generate_one(CLAUSE_SYSTEM, clause_user(r["statement"]), img, 512)
        final = vlm.generate_one(CLAUSE_JUDGE_SYSTEM, judge_user(r["statement"], chk), None, 256)
        checks.append(chk)
        raw_final.append(final)
        preds.append(parse_answer(final))
    elapsed = time.time() - start
    gold = [r["gold"] for r in rows]
    m = metric_pack(preds, gold)
    print_metric(exp, m, elapsed / len(rows))

    out_path = out_dir / f"corevqa_{exp}.csv"
    base_path = out_dir / f"corevqa_format_{long_side}.csv"
    base = {}
    if base_path.exists():
        with base_path.open(encoding="utf-8") as f:
            base = {r["image_id"]: r for r in csv.DictReader(f)}

    fields = [
        "image_id",
        "image_path",
        "statement",
        "gold_label",
        "pred_img",
        "baseline_pred_img",
        "raw_output_img",
        "visual_checks",
        "reasoning_img",
        "correct_img",
        "baseline_correct_img",
        "changed_from_baseline",
        "auto_tags",
        "image_size",
        "resize_long_side",
        "experiment_name",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, pred, chk, raw in zip(rows, preds, checks, raw_final):
            b = base.get(r["image_id"], {})
            bp = b.get("pred_img", "")
            w.writerow(
                {
                    "image_id": r["image_id"],
                    "image_path": r["image_path"],
                    "statement": r["statement"],
                    "gold_label": r["gold"],
                    "pred_img": pred,
                    "baseline_pred_img": bp,
                    "raw_output_img": raw,
                    "visual_checks": chk,
                    "reasoning_img": reasoning(raw),
                    "correct_img": int(pred == r["gold"]),
                    "baseline_correct_img": b.get("correct_img", ""),
                    "changed_from_baseline": int(str(pred) != str(bp)) if bp != "" else "",
                    "auto_tags": "|".join(tag_statement(r["statement"])),
                    "image_size": r["image_size"],
                    "resize_long_side": long_side,
                    "experiment_name": exp,
                }
            )
    write_wrong_html(rows, preds, raw_final, out_dir / f"corevqa_{exp}_wrong.html")
    if base:
        compare_to_baseline(rows, preds, out_dir, long_side, exp)
    return out_path


def compare_to_baseline(rows: list[dict], preds: list[int], out_dir: Path, long_side: int, exp: str) -> None:
    base_path = out_dir / f"corevqa_format_{long_side}.csv"
    with base_path.open(encoding="utf-8") as f:
        base_rows = {r["image_id"]: r for r in csv.DictReader(f)}
    base_preds = [int(base_rows[r["image_id"]]["pred_img"]) for r in rows]
    gold = [r["gold"] for r in rows]
    bm = metric_pack(base_preds, gold)
    vm = metric_pack(preds, gold)
    improved = [i for i in range(len(rows)) if base_preds[i] != gold[i] and preds[i] == gold[i]]
    degraded = [i for i in range(len(rows)) if base_preds[i] == gold[i] and preds[i] != gold[i]]
    verdict = "KEEP_CANDIDATE" if (vm["ba"] - bm["ba"] >= 0.02 and vm["true_acc"] >= bm["true_acc"] - 0.01) else "REJECT"
    print("\n=== vs baseline ===")
    print_metric("baseline", bm, None)
    print_metric(exp, vm, None)
    print(f"changed={sum(a != b for a, b in zip(base_preds, preds))} improved={len(improved)} degraded={len(degraded)} net={len(improved)-len(degraded)} verdict={verdict}")

    lines = [f"verdict={verdict}", f"improved={len(improved)} degraded={len(degraded)} net={len(improved)-len(degraded)}"]
    for title, idxs in [("IMPROVED", improved[:20]), ("DEGRADED", degraded[:20])]:
        lines.append(f"\n## {title}")
        for i in idxs:
            lines.extend(
                [
                    f"[{rows[i]['image_id']}] gold={gold[i]} base={base_preds[i]} v16={preds[i]}",
                    f"stmt: {rows[i]['statement']}",
                ]
            )
    (out_dir / f"corevqa_{exp}_diff_vs_format_{long_side}.txt").write_text("\n".join(lines), encoding="utf-8")


def print_metric(name: str, m: dict, sec: float | None) -> None:
    speed = "" if sec is None else f" sec/sample={sec:.3f}"
    print(
        f"{name}: acc={m['acc']*100:.1f}% BA={m['ba']*100:.1f}% "
        f"TRUE={m['true_acc']*100:.1f}% FALSE={m['false_acc']*100:.1f}% "
        f"commit_acc={m['commit_acc']*100:.1f}% abstain={m['abstain']*100:.1f}% pred={dict(m['pred_counts'])}{speed}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--experiment", choices=["format_768", "v16_clause", "both"], default="both")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--data-dir", default="data/corevqa_local")
    ap.add_argument("--out-dir", default="outputs/corevqa_local")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    ensure_runtime_deps()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, index = ensure_corevqa(data_dir)
    rows = fixed_samples(meta, index, args.limit, args.seed)
    print(f"fixed samples: {len(rows)}")

    vlm = LocalVLM(args.model, args.device, args.dtype)
    if args.experiment in {"format_768", "both"}:
        run_format(vlm, rows, out_dir, args.long_side, args.max_tokens)
    if args.experiment in {"v16_clause", "both"}:
        run_clause(vlm, rows, out_dir, args.long_side)


if __name__ == "__main__":
    main()
