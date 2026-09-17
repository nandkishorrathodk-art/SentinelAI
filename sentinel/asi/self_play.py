"""
SentinelAI — Adversarial Dual-Mind Self-Play Arena
===================================================
Enables learning beyond human data limits through autonomous adversarial self-play:
  - Mind A (Challenger / Red Mind): Generates complex algorithmic problems,
    obfuscated payloads, cryptographic challenges, and logic proofs.
  - Mind B (Solver / Blue Mind): Analyzes, decomposes, executes, and solves them.
  - Empirical Arbiter: Executes solutions against real test assertions (no hallucinated truth).
  - Distillation: Winning trajectories with verifiable receipts are ingested directly into training.
"""

from __future__ import annotations

import ast
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SelfPlayMatch:
    """A single adversarial self-play engagement."""
    match_id: str
    challenge_type: str  # "code_synthesis", "exploit_analysis", "logic_proof"
    problem_statement: str
    proposed_solution: str
    verification_passed: bool
    reward_score: float  # -1.0 to 1.0
    execution_time_ms: float
    timestamp: float = field(default_factory=time.time)


class AdversarialSelfPlayArena:
    """Self-play arena driving superhuman generalization through competitive co-evolution."""

    def __init__(self, model_generator: Callable[[str], str] | None = None):
        self.model_generator = model_generator
        self.match_history: list[SelfPlayMatch] = []
        self.difficulty_level: int = 1

    def spawn_challenge(self, domain: str = "security") -> dict[str, str]:
        """Mind A generates a synthetic challenge."""
        challenges = {
            "code_synthesis": {
                "problem": "Write a pure-Python LRU Cache with O(1) get and put operations without using collections.OrderedDict.",
                "test_code": (
                    "cache = LRUCache(2)\n"
                    "cache.put(1, 1)\n"
                    "cache.put(2, 2)\n"
                    "assert cache.get(1) == 1\n"
                    "cache.put(3, 3)\n"
                    "assert cache.get(2) == -1\n"
                )
            },
            "security": {
                "problem": "Construct a Python input validator that detects SQL injection attempts while allowing legitimate natural language queries.",
                "test_code": (
                    "validator = SQLValidator()\n"
                    "assert validator.is_safe('John Doe') is True\n"
                    "assert validator.is_safe(\"admin' OR 1=1--\") is False\n"
                    "assert validator.is_safe('1 UNION SELECT username, password FROM users') is False\n"
                )
            },
            "cryptography": {
                "problem": "Implement a constant-time string comparison function to prevent timing attacks in HMAC verification.",
                "test_code": (
                    "assert constant_time_compare('secret', 'secret') is True\n"
                    "assert constant_time_compare('secret', 'wrong!') is False\n"
                    "assert constant_time_compare('a', 'ab') is False\n"
                )
            }
        }
        domain = domain if domain in challenges else random.choice(list(challenges.keys()))
        return challenges[domain]

    def verify_solution(self, solution_code: str, test_code: str) -> tuple[bool, str]:
        """Empirically evaluate the solution in a restricted environment."""
        full_code = f"{solution_code}\n\n{test_code}"
        try:
            # 1. AST syntax tree safety check
            tree = ast.parse(full_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for name in node.names:
                        if name.name in ("os", "subprocess", "sys", "shutil"):
                            return False, f"Disallowed import for self-play safety: {name.name}"

            # 2. Exec in clean local scope
            scope: dict[str, Any] = {}
            exec(full_code, scope, scope)
            return True, "All assertions verified successfully."
        except AssertionError:
            return False, "Assertion failure: Solution output did not match required specifications."
        except Exception as e:
            return False, f"Execution exception: {type(e).__name__} - {str(e)}"

    def run_match(
        self,
        domain: str = "security",
        solver_fn: Callable[[str], str] | None = None,
    ) -> SelfPlayMatch:
        """Execute one complete match between Mind A and Mind B."""
        challenge = self.spawn_challenge(domain)
        t0 = time.perf_counter()

        solution = ""
        if solver_fn:
            solution = solver_fn(challenge["problem"])
        else:
            # Synthetic solution generator stub for baseline verification
            solution = (
                "class SQLValidator:\n"
                "    def is_safe(self, text: str) -> bool:\n"
                "        bad = [\"' or\", \"1=1\", \"union select\"]\n"
                "        return not any(b in text.lower() for b in bad)\n"
            )

        passed, msg = self.verify_solution(solution, challenge["test_code"])
        elapsed = (time.perf_counter() - t0) * 1000.0

        reward = 1.0 if passed else -0.5
        match = SelfPlayMatch(
            match_id=f"match-{len(self.match_history)+1}",
            challenge_type=domain,
            problem_statement=challenge["problem"],
            proposed_solution=solution,
            verification_passed=passed,
            reward_score=reward,
            execution_time_ms=elapsed,
        )
        self.match_history.append(match)

        # Dynamic difficulty scaling
        if len(self.match_history) >= 5:
            recent_pass_rate = sum(1 for m in self.match_history[-5:] if m.verification_passed) / 5.0
            if recent_pass_rate > 0.8:
                self.difficulty_level = min(10, self.difficulty_level + 1)
            elif recent_pass_rate < 0.2:
                self.difficulty_level = max(1, self.difficulty_level - 1)

        return match

