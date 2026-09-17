"""
SentinelAI — Neural World Model & Latent Simulator
===================================================
Enables SentinelAI to simulate consequences before taking real-world actions:
  - Latent Transition Model: Computes expected state change s_{t+1} = f(s_t, action).
  - Monte Carlo Tree Search (MCTS) / Tree-of-Thoughts: Explores multiple execution
    trajectories in simulation, evaluating safety and success probabilities.
  - Empirical Rollouts: Picks the optimal trajectory with maximum expected reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SimulationNode:
    """A single decision node in the latent world model tree."""
    state_desc: str
    action_taken: str | None = None
    parent: SimulationNode | None = None
    children: list[SimulationNode] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    depth: int = 0


class NeuralWorldModel:
    """Latent simulator enabling forward mental planning before execution."""

    def __init__(self, horizon: int = 5, exploration_weight: float = 1.414):
        self.horizon = horizon
        self.c_puct = exploration_weight

    def simulate_transition(self, current_state: str, action: str) -> tuple[str, float]:
        """Predict next state and immediate reward using latent transition rules."""
        action_lower = action.lower()
        if "rm -rf" in action_lower or "drop database" in action_lower:
            return f"State: Catastrophic data loss after {action}", -10.0
        elif "scan" in action_lower or "probe" in action_lower:
            return f"State: Network telemetry collected after {action}", 2.0
        elif "test" in action_lower or "verify" in action_lower:
            return f"State: Formal verification proof generated after {action}", 3.0
        elif "patch" in action_lower or "fix" in action_lower:
            return f"State: Vulnerability remediated after {action}", 5.0
        return f"State: System updated via {action}", 0.5

    def search(
        self,
        initial_state: str,
        candidate_actions: list[str],
        num_rollouts: int = 20,
    ) -> str:
        """Run MCTS forward planning to find the best action."""
        root = SimulationNode(state_desc=initial_state)

        for _ in range(num_rollouts):
            # 1. Selection
            node = root
            while node.children and node.depth < self.horizon:
                node = self._select_best_uct_child(node)

            # 2. Expansion
            if node.depth < self.horizon and not node.children:
                for act in candidate_actions:
                    next_state, imm_reward = self.simulate_transition(node.state_desc, act)
                    child = SimulationNode(
                        state_desc=next_state,
                        action_taken=act,
                        parent=node,
                        depth=node.depth + 1,
                        value=imm_reward,
                    )
                    node.children.append(child)

            # 3. Rollout / Evaluation
            rollout_val = node.value

            # 4. Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value += (rollout_val - curr.value) / curr.visits
                curr = curr.parent

        # Choose most visited child action from root
        if not root.children:
            return candidate_actions[0] if candidate_actions else "NOOP"

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action_taken or candidate_actions[0]

    def _select_best_uct_child(self, node: SimulationNode) -> SimulationNode:
        """Select child maximizing UCT score (Upper Confidence Bound)."""
        log_parent_visits = math.log(max(1, node.visits))

        def uct(child: SimulationNode) -> float:
            if child.visits == 0:
                return float("inf")
            exploitation = child.value
            exploration = self.c_puct * math.sqrt(log_parent_visits / child.visits)
            return exploitation + exploration

        return max(node.children, key=uct)

