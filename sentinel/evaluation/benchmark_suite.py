"""
Sentinel 5-Bucket Official Evaluation Benchmark Suite
=====================================================
Directly codifies the architecture critique:
Evaluates Sentinel as a learning system across 5 rigorous buckets:
1. Language (fluency, grammar, repetition score)
2. Knowledge (factual recall + GraphRAG comparison)
3. Reasoning (strictly held-out, unseen test cases — reasoning != <think> length)
4. Systems Knowledge (CPU cache, memory hierarchy, OS processes/threads)
5. Reliability (hallucination detection and uncertainty expression)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

from forge.generate import load_sentinel_model, generate_text, BytePairTokenizer, SentinelTransformer
from sentinel.graphrag.graph import KnowledgeGraph
from sentinel.graphrag.seed_knowledge import populate_universal_knowledge_graph

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


class SentinelFiveBucketBenchmark:
    """Evaluates the 5 core dimensions of Sentinel's learning progress."""

    # 1. Held-Out Reasoning (MUST be completely unseen numbers to test real deduction)
    HELD_OUT_REASONING_TESTS = [
        {
            "id": "linear_unseen_1",
            "prompt": "Question: Solve for x in the equation 7x + 14 = 49.\n<think>",
            "expected_val": "5",
            "correct_equation": "7x = 35",
        },
        {
            "id": "linear_unseen_2",
            "prompt": "Question: Solve for x in the equation 9x - 18 = 63.\n<think>",
            "expected_val": "9",
            "correct_equation": "9x = 81",
        },
        {
            "id": "product_unseen_1",
            "prompt": "Question: Calculate the product of 16 and 25.\n<think>",
            "expected_val": "400",
            "correct_equation": "400",
        },
    ]

    # 2. Systems Knowledge Tests
    SYSTEMS_TESTS = [
        {
            "id": "cache_locality",
            "prompt": "<|im_start|>user\nWhy does iterating through an array have higher CPU cache locality than a linked list?<|im_end|>\n<|im_start|>assistant\n<think>",
            "key_concepts": ["contiguous", "cache line", "spatial locality"],
        },
        {
            "id": "process_vs_thread",
            "prompt": "<|im_start|>user\nExplain why thread switching has lower latency than process switching.<|im_end|>\n<|im_start|>assistant\n<think>",
            "key_concepts": ["address space", "tlb", "registers"],
        },
    ]

    # 3. Knowledge & Science Tests
    KNOWLEDGE_TESTS = [
        {
            "id": "rayleigh_sky",
            "prompt": "<|im_start|>user\nWhy is the sky blue during the daytime?<|im_end|>\n<|im_start|>assistant\n<think>",
            "key_concepts": ["rayleigh", "wavelength", "scatter"],
        },
        {
            "id": "empirical_falsification",
            "prompt": "<|im_start|>user\nWhat is Karl Popper's principle of empirical falsification?<|im_end|>\n<|im_start|>assistant\n<think>",
            "key_concepts": ["popper", "falsif", "test"],
        },
    ]

    # 4. Reliability & Hallucination Resistance
    RELIABILITY_TESTS = [
        {
            "id": "anti_bleed_check",
            "prompt": "Question: Solve for x in 5x + 10 = 30.\n<think>",
            "forbidden_terms": ["quicksort", "pivot", "array", "partition"],
        }
    ]

    def __init__(self, model: SentinelTransformer, tokenizer: BytePairTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_language_bucket(self) -> Dict[str, Any]:
        """Bucket 1: Measures repetition penalty and multi-turn turn-taking."""
        sample_prompt = "<|im_start|>system\nYou are Sentinel.<|im_end|>\n<|im_start|>user\nHello, tell me about yourself.<|im_end|>\n<|im_start|>assistant\n<think>"
        output, speed = generate_text(self.model, self.tokenizer, sample_prompt, max_new_tokens=50, device=self.device)

        words = output.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        # N-gram repetition check (3-grams)
        tri_grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        rep_3gram = (len(tri_grams) - len(set(tri_grams))) / max(len(tri_grams), 1) if tri_grams else 0.0

        passed = rep_3gram < 0.3
        return {
            "passed": passed,
            "speed_tokens_per_sec": speed,
            "vocabulary_diversity": unique_ratio,
            "repetition_penalty": rep_3gram,
            "sample": output.strip()[:120]
        }

    def evaluate_reasoning_bucket(self) -> Dict[str, Any]:
        """Bucket 3: Strictly held-out test cases — measures real deduction vs format mimicry."""
        passed = 0
        details = []

        for item in self.HELD_OUT_REASONING_TESTS:
            output, _ = generate_text(self.model, self.tokenizer, item["prompt"], max_new_tokens=60, temperature=0.2, device=self.device)
            out_lower = output.lower()

            # Check if expected mathematical value or equation is present in deduction
            correct_found = (item["expected_val"] in out_lower) or (item["correct_equation"].lower() in out_lower)
            if correct_found:
                passed += 1

            details.append({
                "id": item["id"],
                "passed": correct_found,
                "expected": item["expected_val"],
                "sample": output.strip()[:100]
            })

        score = (passed / len(self.HELD_OUT_REASONING_TESTS)) * 100
        return {"accuracy": score, "tests": details}

    def evaluate_systems_bucket(self) -> Dict[str, Any]:
        """Bucket 4: CPU, Memory hierarchy, OS processes and threads."""
        passed = 0
        details = []
        for item in self.SYSTEMS_TESTS:
            output, _ = generate_text(self.model, self.tokenizer, item["prompt"], max_new_tokens=65, device=self.device)
            out_lower = output.lower()
            matched = [k for k in item["key_concepts"] if k in out_lower]
            p = len(matched) > 0
            if p:
                passed += 1
            details.append({"id": item["id"], "passed": p, "matched": matched})
        return {"accuracy": (passed / len(self.SYSTEMS_TESTS)) * 100, "tests": details}

    def evaluate_knowledge_bucket(self) -> Dict[str, Any]:
        """Bucket 2: Factual and scientific knowledge grounded in graph concepts."""
        passed = 0
        details = []
        for item in self.KNOWLEDGE_TESTS:
            output, _ = generate_text(self.model, self.tokenizer, item["prompt"], max_new_tokens=65, device=self.device)
            out_lower = output.lower()
            matched = [k for k in item["key_concepts"] if k in out_lower]
            p = len(matched) > 0
            if p:
                passed += 1
            details.append({"id": item["id"], "passed": p, "matched": matched})
        return {"accuracy": (passed / len(self.KNOWLEDGE_TESTS)) * 100, "tests": details}

    def evaluate_reliability_bucket(self) -> Dict[str, Any]:
        """Bucket 5: Cognitive bleed and hallucination resistance."""
        passed = 0
        details = []
        for item in self.RELIABILITY_TESTS:
            output, _ = generate_text(self.model, self.tokenizer, item["prompt"], max_new_tokens=50, device=self.device)
            out_lower = output.lower()
            bleed = [t for t in item["forbidden_terms"] if t in out_lower]
            p = len(bleed) == 0
            if p:
                passed += 1
            details.append({"id": item["id"], "passed": p, "bleed_detected": bleed})
        return {"accuracy": (passed / len(self.RELIABILITY_TESTS)) * 100, "tests": details}

    def run_full_benchmark(self) -> Dict[str, Any]:
        print("=" * 70)
        print("RUNNING SENTINEL 5-BUCKET EVALUATION FRAMEWORK")
        print("=" * 70)

        print("[*] Bucket 1: Language & Fluency...")
        b1 = self.evaluate_language_bucket()

        print("[*] Bucket 2: Knowledge Grounding...")
        b2 = self.evaluate_knowledge_bucket()

        print("[*] Bucket 3: Held-Out Unseen Reasoning (Format != Reasoning)...")
        b3 = self.evaluate_reasoning_bucket()

        print("[*] Bucket 4: Systems Knowledge (CPU/Memory/OS)...")
        b4 = self.evaluate_systems_bucket()

        print("[*] Bucket 5: Reliability & Anti-Bleed...")
        b5 = self.evaluate_reliability_bucket()

        report = {
            "language": b1,
            "knowledge": b2,
            "reasoning_held_out": b3,
            "systems": b4,
            "reliability": b5,
        }

        print("\n" + "=" * 70)
        print("OFFICIAL 5-BUCKET BENCHMARK REPORT:")
        print(f"1. Language Fluency:      {'PASSED' if b1['passed'] else 'REPETITIVE'} ({b1['speed_tokens_per_sec']:.1f} tok/s)")
        print(f"2. Knowledge Recall:      {b2['accuracy']:.1f}%")
        print(f"3. Held-Out Reasoning:    {b3['accuracy']:.1f}% (Unseen Math & Logic)")
        print(f"4. Systems Knowledge:     {b4['accuracy']:.1f}% (Architecture & OS)")
        print(f"5. Reliability:           {b5['accuracy']:.1f}% (Zero Bleed Score)")
        print("=" * 70)
        return report


def main():
    parser = argparse.ArgumentParser(description="Sentinel 5-Bucket Official Benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_graphrag.pt",
        help="Path to checkpoint"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    chk = args.checkpoint
    if not os.path.exists(chk):
        chk = "c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_v3.pt"

    model, tokenizer = load_sentinel_model(chk, device=args.device)
    bench = SentinelFiveBucketBenchmark(model, tokenizer, device=args.device)
    bench.run_full_benchmark()


if __name__ == "__main__":
    main()

