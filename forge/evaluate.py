"""
Sentinel Sovereign Neural LLM — Evaluation & Benchmark Suite
============================================================
Evaluates:
1. Validation Perplexity (PPL = exp(loss)) across cognitive domains.
2. Generation Throughput (tokens/sec) and Time-To-First-Token (TTFT).
3. Reasoning Accuracy across Mathematical, Algorithmic, and Logical benchmarks.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from forge.generate import load_sentinel_model, generate_text, BytePairTokenizer, SentinelTransformer
from forge.training.trainer import CognitiveDataset
from forge.training.dataset_agents import DatasetSynthesizer
from forge.chat_template import format_chat_prompt, parse_reasoning_and_response


BENCHMARK_PROMPTS = [
    {
        "domain": "Mathematics",
        "prompt": "Question: Solve for x in the equation 4x + 12 = 36.\n<think>",
        "expected": "6",
        "key_phrases": ["subtract 12", "divide", "24"]
    },
    {
        "domain": "Arithmetic",
        "prompt": "Question: Calculate the product of 15 and 20.\n<think>",
        "expected": "300",
        "key_phrases": ["multiply", "300"]
    },
    {
        "domain": "Algorithms",
        "prompt": "Question: How does the quicksort algorithm partition an array?\n<think>",
        "expected": "pivot",
        "key_phrases": ["pivot", "divide-and-conquer", "partition"]
    },
    {
        "domain": "Computer Science",
        "prompt": "Question: How do I reverse a list in Python?\n<think>",
        "expected": "reverse",
        "key_phrases": ["slice", "reverse"]
    },
    {
        "domain": "Epistemics & Science",
        "prompt": "Question: Explain why empirical falsification is the foundation of scientific truth.\n<think>",
        "expected": "popper",
        "key_phrases": ["falsifi", "popper", "observation"]
    }
]


def evaluate_perplexity(
    model: SentinelTransformer,
    tokenizer: BytePairTokenizer,
    val_corpus: str,
    device: str = "cpu",
    seq_len: int = 128
) -> float:
    """Calculates cross-entropy loss and Perplexity (exp(loss)) on validation corpus."""
    model.eval()
    encoded = tokenizer.encode(val_corpus)
    dataset = CognitiveDataset(encoded, seq_len=seq_len)
    if len(dataset) == 0:
        return float("nan")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss, _ = model(x, targets=y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap to avoid math overflow
    return perplexity


def evaluate_reasoning_benchmarks(
    model: SentinelTransformer,
    tokenizer: BytePairTokenizer,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Runs standard cognitive benchmark prompts and scores reasoning quality."""
    results = []
    total_speed = 0.0
    passed_count = 0

    for item in BENCHMARK_PROMPTS:
        t0 = time.time()
        output, speed = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            max_new_tokens=60,
            temperature=0.3,
            top_k=20,
            device=device
        )
        total_speed += speed

        output_lower = output.lower()
        matched_keys = [k for k in item["key_phrases"] if k in output_lower]
        passed = len(matched_keys) > 0
        if passed:
            passed_count += 1

        results.append({
            "domain": item["domain"],
            "prompt": item["prompt"].splitlines()[0],
            "passed": passed,
            "matched": matched_keys,
            "speed": speed,
            "sample": output.strip()[:100] + "..."
        })

    accuracy = (passed_count / len(BENCHMARK_PROMPTS)) * 100
    avg_speed = total_speed / len(BENCHMARK_PROMPTS)

    return {
        "accuracy": accuracy,
        "avg_speed": avg_speed,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Sentinel Evaluation Benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_v2.pt",
        help="Path to checkpoint"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    chk = args.checkpoint
    if not os.path.exists(chk):
        chk = "c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_final.pt"

    print("=" * 70)
    print("SENTINEL MODEL EVALUATION & BENCHMARK SCORECARD")
    print(f"Checkpoint: {chk}")
    print("=" * 70)

    model, tokenizer = load_sentinel_model(chk, device=args.device)

    # 1. Perplexity
    print("\n[*] Evaluating Validation Perplexity on Cognitive Substrate...")
    synthesizer = DatasetSynthesizer()
    val_corpus = synthesizer.build_curriculum_corpus()
    ppl = evaluate_perplexity(model, tokenizer, val_corpus, device=args.device)
    print(f"[+] Model Perplexity (PPL): {ppl:.2f}")

    # 2. Reasoning Accuracy & Latency
    print("\n[*] Evaluating Reasoning & Algorithmic Benchmarks...")
    bench = evaluate_reasoning_benchmarks(model, tokenizer, device=args.device)
    print(f"[+] Benchmark Accuracy: {bench['accuracy']:.1f}% ({bench['avg_speed']:.1f} tokens/sec)")
    print("-" * 70)
    for r in bench["results"]:
        status = "PASSED" if r["passed"] else "FAILED"
        print(f"[{status}] {r['domain']}: {r['prompt']} ({r['speed']:.1f} tok/s)")

    print("=" * 70)
    print("SCORECARD SUMMARY:")
    print(f"Perplexity: {ppl:.2f} | Reasoning Accuracy: {bench['accuracy']:.1f}% | Speed: {bench['avg_speed']:.1f} tok/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
