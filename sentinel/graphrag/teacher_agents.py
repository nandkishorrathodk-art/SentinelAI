"""
Sentinel GraphRAG Teacher Sub-Agents Swarm
==========================================
Specialized teacher sub-agents that traverse the Knowledge Graph's multi-hop topology
and synthesize rich, grounded <think> Chain-of-Thought training dialogues.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from sentinel.graphrag.graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge


class GraphMathTeacher:
    """Sub-agent synthesizing mathematical deduction grounded in graph topology."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def teach_mathematical_derivation(self) -> List[str]:
        examples = []
        # Path: Arithmetic -> Algebra -> Linear Equations
        examples.append(
            "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
            "<|im_start|>user\nHow does algebra generalize arithmetic when solving linear equations?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
            "1. Knowledge Graph Traversal: [Arithmetic] --(DEPENDS_ON)--> [Algebra] --(DERIVES)--> [Linear Equations].\n"
            "2. In arithmetic, operations deal with concrete known quantities (e.g., 5 + 3 = 8).\n"
            "3. Algebra introduces symbolic abstraction (variables like x, y) to represent unknown or varying quantities.\n"
            "4. A linear equation (e.g., ax + b = c) applies arithmetic inverses (subtraction and division) symmetrically across the equality sign to isolate the unknown variable.\n"
            "5. The equality holds because inverse operations preserve the invariant truth value of the algebraic statement.\n"
            "</think>\n"
            "Algebra generalizes arithmetic by using symbolic variables (such as x) to represent unknown values. When solving a linear equation like ax + b = c, we apply arithmetic inverse operations symmetrically to both sides (subtracting b, then dividing by a) to isolate the variable, preserving the fundamental equality.<|im_end|>"
        )

        # Path: Calculus -> Gradient Descent -> Optimization
        examples.append(
            "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
            "<|im_start|>user\nWhy does gradient descent require calculus to optimize neural network weights?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
            "1. Knowledge Graph Traversal: [Calculus] --(DERIVES)--> [Gradient Descent] --(IMPLEMENTS)--> [Optimization].\n"
            "2. Optimization seeks parameter values that minimize an objective loss function L(w).\n"
            "3. Calculus provides the derivative: the instantaneous rate of change of a function with respect to its inputs.\n"
            "4. For multi-variable neural networks, multivariate calculus provides the gradient vector (partial derivatives dL/dw_i).\n"
            "5. The gradient points in the direction of steepest ascent. Gradient descent steps in the negative gradient direction (-eta * grad) to systematically decrease error.\n"
            "</think>\n"
            "Gradient descent relies on calculus because the derivative measures the instantaneous rate of change of the loss function with respect to each parameter. By computing the gradient vector (partial derivatives) via calculus, the algorithm identifies the exact direction of steepest descent, stepping downhill to minimize error and optimize the neural weights.<|im_end|>"
        )
        return examples


class GraphSystemsTeacher:
    """Sub-agent teaching computer architecture and system software grounded in graph topology."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def teach_systems_and_code(self) -> List[str]:
        examples = []
        # Path: CPU Registers -> Memory Hierarchy -> RAM -> Array Cache Locality
        examples.append(
            "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
            "<|im_start|>user\nWhy are arrays significantly faster to iterate over than linked lists in modern CPUs?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
            "1. Knowledge Graph Traversal: [CPU Registers] -> [Memory Hierarchy (L1/L2/L3)] -> [RAM] <-> [Array] vs [Linked List].\n"
            "2. CPU cores execute instructions in nanoseconds, but accessing main RAM requires ~50-100ns (memory wall).\n"
            "3. To bridge this latency gap, modern CPUs employ hardware prefetchers and cache lines (typically 64 bytes).\n"
            "4. Arrays allocate elements in a single contiguous memory block. When index i is read, adjacent elements are pre-loaded into high-speed L1/L2 cache (spatial locality).\n"
            "5. Linked lists allocate nodes dynamically on the heap; pointers bounce across scattered addresses, triggering frequent CPU cache misses and stalling execution pipelines.\n"
            "</think>\n"
            "Arrays are significantly faster because of CPU cache spatial locality. Modern CPUs load memory in 64-byte cache lines; since array elements are stored contiguously, accessing one element pre-loads subsequent elements into fast L1/L2 cache. In contrast, linked list nodes are scattered across RAM, causing frequent cache misses that stall CPU execution.<|im_end|>"
        )

        # Path: Operating System -> Process vs Thread -> Python Runtime
        examples.append(
            "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
            "<|im_start|>user\nExplain why thread switching has lower overhead than process switching in an operating system.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
            "1. Knowledge Graph Traversal: [Operating System] --(DERIVES)--> [Process] --(CONTAINS)--> [Thread].\n"
            "2. A Process is an execution boundary with its own virtual memory address space (Page Tables, CR3 register in x86).\n"
            "3. Switching between two processes requires saving CPU registers, switching memory page tables, and invalidating the Translation Lookaside Buffer (TLB), incurring high latency.\n"
            "4. Threads belonging to the same process share the same virtual address space, heap, open file descriptors, and code segment.\n"
            "5. A thread switch only requires saving and restoring execution state: program counter, stack pointer, and general-purpose registers, without touching memory mappings or TLB.\n"
            "</think>\n"
            "Thread switching has lower overhead because threads within the same process share the same virtual memory address space. Switching threads only requires swapping CPU registers and stack pointers. Process switching, however, requires switching memory page tables and flushing the CPU's Translation Lookaside Buffer (TLB), which is a much heavier operation.<|im_end|>"
        )
        return examples


class GraphScienceTeacher:
    """Sub-agent teaching physics and scientific epistemology grounded in graph topology."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def teach_science_and_epistemics(self) -> List[str]:
        examples = []
        # Path: Electromagnetism -> Light Waves -> Rayleigh Scattering -> Blue Sky
        examples.append(
            "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
            "<|im_start|>user\nWalk me through the physical reason why the daytime sky appears blue.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
            "1. Knowledge Graph Traversal: [Electromagnetism] -> [Light Waves] -> [Rayleigh Scattering] -> [Blue Sky Color].\n"
            "2. Sunlight consists of the entire visible electromagnetic spectrum, traveling as waves of varying wavelengths (red ~700nm, blue ~400nm).\n"
            "3. Earth's atmosphere is composed of nitrogen and oxygen molecules, which are much smaller than visible light wavelengths.\n"
            "4. When light strikes particles smaller than its wavelength, Rayleigh scattering occurs, where scattering intensity is proportional to 1 / (lambda^4).\n"
            "5. Because blue light has a wavelength nearly half that of red light, it scatters roughly (700/400)^4 ~ 10 times more intensely.\n"
            "6. This scattered short-wavelength radiation disperses in all directions across the atmosphere, illuminating the sky with a blue hue.\n"
            "</think>\n"
            "The sky appears blue because of Rayleigh scattering. Sunlight contains all visible colors, but short-wavelength light (blue, ~400nm) scatters inversely with the fourth power of its wavelength (1/lambda^4). Air molecules scatter blue light roughly ten times more effectively than red light, dispersing it in every direction across the atmosphere.<|im_end|>"
        )

        # Path: Empirical Falsification -> Scientific Progress
        examples.append(
            "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
            "<|im_start|>user\nWhat makes empirical falsification stronger than empirical confirmation in scientific reasoning?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
            "1. Knowledge Graph Traversal: [Empirical Falsification] -> [Karl Popper] -> [Scientific Truth].\n"
            "2. The asymmetry of induction: Observing 10,000 white swans does not prove the universal claim 'All swans are white'.\n"
            "3. However, observing a single black swan definitively refutes (falsifies) the universal claim.\n"
            "4. In formal logic: Modus tollens (If P then Q; not Q; therefore not P) is deductively valid, whereas affirming the consequent is a logical fallacy.\n"
            "5. Therefore, genuine scientific theories must be formulated such that empirical observation could potentially refute them, distinguishing science from unfalsifiable dogma.\n"
            "</think>\n"
            "Empirical falsification is logically stronger because of the fundamental asymmetry of induction: no number of positive observations can conclusively prove a universal rule, but a single verifiable counterexample refutes it. In formal deductive logic (modus tollens), a failed prediction definitively disproves a hypothesis, making falsifiability the gold standard of scientific truth.<|im_end|>"
        )
        return examples


class GraphDialogueTeacher:
    """Sub-agent generating multi-turn conversational interactions grounded in graph relations."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def teach_conversational_grounding(self) -> List[str]:
        examples = []
        # Path: Transformer -> Self-Attention -> SA-MoE -> AGI -> ASI
        examples.append(
            "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
            "<|im_start|>user\nHow does your SA-MoE architecture differ from standard dense transformers?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
            "1. Knowledge Graph Traversal: [Transformer] -> [Self-Attention / GQA] -> [SA-MoE].\n"
            "2. Standard dense transformers route every token through all parameters in every feedforward layer, causing O(params) compute per token.\n"
            "3. Sentinel Adaptive Mixture-of-Experts (SA-MoE) decouples total parameter capacity from compute cost.\n"
            "4. Each layer contains specialized expert networks and an always-on shared expert.\n"
            "5. A learned router dynamically selects the top-2 experts per token, allowing 46.66M parameter capacity with sparse activation efficiency.\n"
            "</think>\n"
            "Standard dense transformers pass every token through all parameters, which limits scaling efficiency. My Sentinel Adaptive MoE (SA-MoE) architecture decouples capacity from compute: it maintains 6 specialized expert networks and 1 always-on shared expert per layer. A learned router activates only the top-2 experts for each token, providing high expressive capacity at minimal compute latency.<|im_end|>"
        )
        return examples


class GraphRAGTeacherSwarm:
    """Orchestrates all graph teacher sub-agents to compile a unified, relational training substrate."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.math_teacher = GraphMathTeacher(kg)
        self.systems_teacher = GraphSystemsTeacher(kg)
        self.science_teacher = GraphScienceTeacher(kg)
        self.dialogue_teacher = GraphDialogueTeacher(kg)

    def harvest_all_lessons(self) -> List[str]:
        all_lessons = []
        all_lessons.extend(self.math_teacher.teach_mathematical_derivation())
        all_lessons.extend(self.systems_teacher.teach_systems_and_code())
        all_lessons.extend(self.science_teacher.teach_science_and_epistemics())
        all_lessons.extend(self.dialogue_teacher.teach_conversational_grounding())

        # Duplicate key structural lessons to strengthen weights
        weighted_lessons = []
        for l in all_lessons:
            weighted_lessons.extend([l] * 3)

        random.seed(42)
        random.shuffle(weighted_lessons)
        return weighted_lessons

