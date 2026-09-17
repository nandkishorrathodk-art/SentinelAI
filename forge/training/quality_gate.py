"""
Sentinel Quality Gate & Code Verifier
====================================
Sub-agent that programmatically verifies synthetic training examples:
- Ensures valid <think> / </think> tag balance.
- Validates embedded Python code syntax via Python AST parser.
- Filters out empty or degenerate examples.
"""

from __future__ import annotations

import ast
import re
from typing import List, Tuple


class QualityGate:
    """Rigorous gatekeeper preventing corrupted or invalid data from reaching the model."""

    def __init__(self):
        self.stats = {
            "total_inspected": 0,
            "passed": 0,
            "rejected_tag_mismatch": 0,
            "rejected_syntax_error": 0,
            "rejected_too_short": 0,
        }

    def verify_example(self, example: str) -> bool:
        self.stats["total_inspected"] += 1

        # 1. Length check
        if len(example.strip()) < 30:
            self.stats["rejected_too_short"] += 1
            return False

        # 2. Tag balance check
        open_tags = len(re.findall(r"<think>", example))
        close_tags = len(re.findall(r"</think>", example))
        if open_tags != close_tags:
            self.stats["rejected_tag_mismatch"] += 1
            return False

        # 3. Code block AST syntax check
        code_blocks = re.findall(r"```python\s*(.*?)\s*```", example, flags=re.DOTALL)
        for code in code_blocks:
            try:
                ast.parse(code)
            except SyntaxError:
                self.stats["rejected_syntax_error"] += 1
                return False

        self.stats["passed"] += 1
        return True

    def filter_dataset(self, examples: List[str]) -> List[str]:
        """Filters a list of examples, keeping only verified high-quality items."""
        return [ex for ex in examples if self.verify_example(ex)]

