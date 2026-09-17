"""
Sentinel Red-Team Critic & Failure Probe
========================================
Probes the Sentinel model checkpoint for hallucinations, conceptual bleed
(e.g., mixing quicksort terms into linear algebra prompts), and incomplete reasoning.
Synthesizes targeted contrastive training pairs to eliminate detected flaws.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
import torch

from forge.generate import SentinelTransformer, BytePairTokenizer, generate_text


class RedTeamCritic:
    """Sub-agent probing model checkpoints for specific cognitive failure modes."""

    PROBE_SUITE = [
        {
            "id": "math_isolation",
            "prompt": "Question: Solve for x in the equation 4x + 12 = 36.\n<think>",
            "forbidden_terms": ["quicksort", "pivot", "partition", "array"],
            "required_terms": ["subtract", "divide", "equation"],
        },
        {
            "id": "algorithm_isolation",
            "prompt": "Question: How does the quicksort algorithm partition an array?\n<think>",
            "forbidden_terms": ["solve for x", "linear equation"],
            "required_terms": ["pivot", "partition"],
        },
        {
            "id": "scientific_truth",
            "prompt": "Question: Explain why empirical falsification is the foundation of scientific truth.\n<think>",
            "forbidden_terms": ["quicksort", "equation"],
            "required_terms": ["falsifi", "popper", "empirical"],
        },
    ]

    def __init__(self, model: SentinelTransformer, tokenizer: BytePairTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_checkpoint(self) -> Dict[str, Any]:
        """Runs the probe suite and scores cognitive separation and accuracy."""
        results = {}
        total_score = 0.0

        for probe in self.PROBE_SUITE:
            output, _ = generate_text(
                self.model,
                self.tokenizer,
                probe["prompt"],
                max_new_tokens=40,
                temperature=0.3,
                top_k=20,
                device=self.device
            )
            output_lower = output.lower()

            # Detect conceptual bleed
            bleed_found = [term for term in probe["forbidden_terms"] if term in output_lower]
            # Detect required reasoning
            req_found = [term for term in probe["required_terms"] if term in output_lower]

            passed = (len(bleed_found) == 0) and (len(req_found) > 0)
            score = (1.0 if len(bleed_found) == 0 else 0.0) * (len(req_found) / max(len(probe["required_terms"]), 1))

            results[probe["id"]] = {
                "passed": passed,
                "score": score,
                "conceptual_bleed": bleed_found,
                "required_matched": req_found,
                "sample_output": output.strip()
            }
            total_score += score

        results["overall_score"] = total_score / len(self.PROBE_SUITE)
        return results

    def synthesize_countermeasures(self, evaluation: Dict[str, Any]) -> List[str]:
        """Generates targeted counter-examples to address detected flaws."""
        countermeasures = []
        math_eval = evaluation.get("math_isolation", {})
        if math_eval.get("conceptual_bleed"):
            # Model confused math with sorting: reinforce pure mathematical isolation
            for a, b, c in [(3, 9, 21), (5, 15, 45), (2, 8, 20), (6, 12, 48)]:
                x_val = (c - b) // a
                cm = (
                    f"Question: Solve for x in the equation {a}x + {b} = {c}.\n"
                    f"<think>\n"
                    f"1. This is a linear algebraic equation, requiring arithmetic isolation of the variable x.\n"
                    f"2. Subtract {b} from both sides: {a}x = {c} - {b} = {c - b}.\n"
                    f"3. Divide both sides by {a}: x = {c - b} / {a} = {x_val}.\n"
                    f"4. The solution is strictly x = {x_val}.\n"
                    f"</think>\n"
                    f"Answer: x = {x_val}"
                )
                countermeasures.append(cm)
        return countermeasures
