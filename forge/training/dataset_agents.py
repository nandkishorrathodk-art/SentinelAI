"""
Sentinel Sub-Agent Dataset Teachers
===================================
Specialized teacher sub-agents that generate high-fidelity synthetic training examples
with explicit <think> Chain-of-Thought traces across Mathematics, Algorithmic Logic,
and Scientific/Computing Principles.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple


class MathReasoningAgent:
    """Sub-agent generating rigorous mathematical problem-solving with step-by-step <think> traces."""

    def generate_linear_equations(self, count: int = 25) -> List[str]:
        examples = []
        for _ in range(count):
            a = random.randint(2, 12)
            x = random.randint(1, 20)
            b = random.randint(1, 50)
            sign = random.choice(["+", "-"])
            if sign == "+":
                c = a * x + b
                prompt = f"Question: Solve for x in the equation {a}x + {b} = {c}."
                thought = (
                    f"<think>\n"
                    f"1. Identify the given linear equation: {a}x + {b} = {c}.\n"
                    f"2. Subtract {b} from both sides to isolate the term with the variable: {a}x = {c} - {b} = {c - b}.\n"
                    f"3. Divide both sides by {a}: x = {c - b} / {a} = {x}.\n"
                    f"4. Verify the solution: {a}({x}) + {b} = {a * x} + {b} = {c}, which matches the right side.\n"
                    f"</think>\n"
                    f"Answer: x = {x}"
                )
            else:
                c = a * x - b
                prompt = f"Question: Solve for x in the equation {a}x - {b} = {c}."
                thought = (
                    f"<think>\n"
                    f"1. Identify the given linear equation: {a}x - {b} = {c}.\n"
                    f"2. Add {b} to both sides to isolate the variable term: {a}x = {c} + {b} = {c + b}.\n"
                    f"3. Divide both sides by {a}: x = {c + b} / {a} = {x}.\n"
                    f"4. Verify the solution: {a}({x}) - {b} = {a * x} - {b} = {c}, which confirms the solution.\n"
                    f"</think>\n"
                    f"Answer: x = {x}"
                )
            examples.append(f"{prompt}\n{thought}")
        return examples

    def generate_arithmetic_reasoning(self, count: int = 20) -> List[str]:
        examples = []
        for _ in range(count):
            n1 = random.randint(12, 99)
            n2 = random.randint(12, 99)
            product = n1 * n2
            prompt = f"Question: Calculate the product of {n1} and {n2}."
            thought = (
                f"<think>\n"
                f"1. Break down {n2} into its tens and units place: {n2} = {(n2 // 10) * 10} + {n2 % 10}.\n"
                f"2. Multiply {n1} by the tens component: {n1} * {(n2 // 10) * 10} = {n1 * (n2 // 10) * 10}.\n"
                f"3. Multiply {n1} by the units component: {n1} * {n2 % 10} = {n1 * (n2 % 10)}.\n"
                f"4. Sum both intermediate products: {n1 * (n2 // 10) * 10} + {n1 * (n2 % 10)} = {product}.\n"
                f"</think>\n"
                f"Answer: {product}"
            )
            examples.append(f"{prompt}\n{thought}")
        return examples

    def generate_word_problems(self, count: int = 15) -> List[str]:
        templates = [
            ("A baker makes {A} loaves of bread every hour. If the bakery operates for {B} hours a day and sells each loaf for ${C}, what is the total revenue for one day?",
             lambda a, b, c: (
                 f"<think>\n"
                 f"1. Total loaves produced per day = rate per hour ({a}) * operating hours ({b}) = {a * b} loaves.\n"
                 f"2. Revenue per loaf = ${c}.\n"
                 f"3. Total revenue = total loaves ({a * b}) * price per loaf (${c}) = ${a * b * c}.\n"
                 f"</think>\n"
                 f"Answer: ${a * b * c}"
             )),
            ("A train travels at a constant speed of {A} km/h for {B} hours. How far does the train travel?",
             lambda a, b, c: (
                 f"<think>\n"
                 f"1. The distance formula is: Distance = Speed * Time.\n"
                 f"2. Given Speed = {a} km/h, Time = {b} hours.\n"
                 f"3. Compute distance: {a} * {b} = {a * b} km.\n"
                 f"</think>\n"
                 f"Answer: {a * b} km"
             )),
        ]
        examples = []
        for _ in range(count):
            tmpl, solver = random.choice(templates)
            a = random.randint(15, 60)
            b = random.randint(4, 10)
            c = random.randint(2, 6)
            prompt = "Question: " + tmpl.format(A=a, B=b, C=c)
            thought = solver(a, b, c)
            examples.append(f"{prompt}\n{thought}")
        return examples


class CodeLogicAgent:
    """Sub-agent generating rigorous algorithmic and software engineering problems with <think> traces."""

    def generate_algorithm_explanations(self) -> List[str]:
        return [
            (
                "Question: How does the quicksort algorithm partition an array?\n"
                "<think>\n"
                "1. Quicksort is a divide-and-conquer sorting algorithm with average time complexity O(N log N).\n"
                "2. A pivot element is selected from the array (e.g. first, last, or median-of-three).\n"
                "3. The partitioning routine rearranges elements such that all elements less than the pivot come before it, and all elements greater come after it.\n"
                "4. The pivot is placed into its final sorted position.\n"
                "5. The sub-arrays to the left and right of the pivot are recursively partitioned and sorted.\n"
                "</think>\n"
                "Answer: Quicksort partitions an array by selecting a pivot, placing elements smaller than the pivot to its left and larger elements to its right, then recursively sorting the partitions."
            ),
            (
                "Question: Explain the difference between breadth-first search (BFS) and depth-first search (DFS).\n"
                "<think>\n"
                "1. Both BFS and DFS are fundamental graph traversal algorithms.\n"
                "2. BFS explores a graph level by level, visiting all neighbor nodes at distance k before exploring distance k+1. It uses a FIFO Queue.\n"
                "3. DFS explores as deeply as possible along each branch before backtracking. It uses a LIFO Stack or recursion.\n"
                "4. BFS guarantees finding the shortest path in an unweighted graph.\n"
                "5. DFS has a smaller memory footprint in deep/sparse trees compared to wide trees.\n"
                "</think>\n"
                "Answer: BFS traverses level-by-level using a queue and guarantees the shortest path on unweighted graphs; DFS explores along each branch to leaf depth using recursion or a stack."
            ),
            (
                "Question: How does binary search achieve O(log N) time complexity?\n"
                "<think>\n"
                "1. Binary search requires the input collection to be sorted.\n"
                "2. At each iteration, the algorithm compares the target value to the element at the middle index.\n"
                "3. If the middle element equals the target, the search terminates successfully.\n"
                "4. If the target is smaller, the upper half of the search space is eliminated.\n"
                "5. If the target is larger, the lower half is eliminated.\n"
                "6. Because the search space is halved on each step (N, N/2, N/4, ...), it terminates in at most ceil(log2(N)) comparisons.\n"
                "</think>\n"
                "Answer: Binary search eliminates half of the remaining search space on every comparison, reducing the candidate size from N to 1 in logarithmic time steps O(log N)."
            ),
            (
                "Question: Write a Python function to check if a string is a palindrome.\n"
                "<think>\n"
                "1. A palindrome is a word or phrase that reads the same forwards and backwards.\n"
                "2. In Python, string slicing with step -1 (`s[::-1]`) reverses the string.\n"
                "3. We should clean non-alphanumeric characters and convert to lowercase for case-insensitivity.\n"
                "4. Compare the normalized string to its reversed version.\n"
                "</think>\n"
                "```python\n"
                "def is_palindrome(text: str) -> bool:\n"
                "    clean = [ch.lower() for ch in text if ch.isalnum()]\n"
                "    return clean == clean[::-1]\n"
                "```"
            ),
            (
                "Question: Write a Python function to compute the Fibonacci sequence up to N terms.\n"
                "<think>\n"
                "1. The Fibonacci sequence begins with F(0) = 0, F(1) = 1.\n"
                "2. Each subsequent term is the sum of the preceding two: F(n) = F(n-1) + F(n-2).\n"
                "3. An iterative approach using two variables avoids exponential recursion overhead and runs in O(N) time and O(N) space.\n"
                "</think>\n"
                "```python\n"
                "def fibonacci(n: int) -> list[int]:\n"
                "    if n <= 0:\n"
                "        return []\n"
                "    if n == 1:\n"
                "        return [0]\n"
                "    seq = [0, 1]\n"
                "    for _ in range(2, n):\n"
                "        seq.append(seq[-1] + seq[-2])\n"
                "    return seq\n"
                "```"
            ),
        ]


class ScienceConceptsAgent:
    """Sub-agent generating foundational science, computing theory, and epistemics examples."""

    def generate_science_and_epistemics(self) -> List[str]:
        return [
            (
                "Question: Explain why empirical falsification is the foundation of scientific truth.\n"
                "<think>\n"
                "1. Proposed by Karl Popper, falsifiability states that a scientific hypothesis must be capable of being tested and proven false by observation.\n"
                "2. No amount of positive confirmations can conclusively prove a universal claim, but a single verifiable counterexample refutes it.\n"
                "3. Scientific progress proceeds by formulating bold, falsifiable theories and subjecting them to rigorous empirical testing.\n"
                "4. In contrast, theories that are immune to empirical refutation belong to dogma rather than empirical science.\n"
                "</think>\n"
                "Answer: Empirical falsification is the scientific foundation because universal truth cannot be proven by repeated observations, but can be definitively tested and corrected when refuted by real empirical evidence."
            ),
            (
                "Question: What defines Artificial Superintelligence (ASI)?\n"
                "<think>\n"
                "1. Artificial General Intelligence (AGI) matches human cognitive performance across all intellectual domains.\n"
                "2. Artificial Superintelligence (ASI) vastly surpasses the best human minds in every discipline: science, strategy, logic, and synthesis.\n"
                "3. Key characteristics include recursive self-improvement, rapid scientific hypothesis testing, and deep multi-domain insight.\n"
                "4. The path to ASI relies on grounded autonomous learning, rigorous empirical feedback, and self-evolving neural representations.\n"
                "</think>\n"
                "Answer: Artificial Superintelligence (ASI) refers to an intelligence that vastly outperforms all human capability in science, creativity, problem-solving, and strategic reasoning."
            ),
            (
                "Question: What is the Law of Conservation of Energy?\n"
                "<think>\n"
                "1. The first law of thermodynamics states that energy cannot be created or destroyed in an isolated system.\n"
                "2. Energy can only transform from one form to another (e.g., potential to kinetic, chemical to thermal).\n"
                "3. The total amount of energy in an isolated system remains constant over time.\n"
                "</think>\n"
                "Answer: The Law of Conservation of Energy states that energy cannot be created or destroyed, only transformed from one form to another, maintaining a constant total energy in an isolated system."
            ),
            (
                "Question: What is a Byte-Pair Encoding (BPE) tokenizer?\n"
                "<think>\n"
                "1. BPE is a subword tokenization algorithm originally adapted from data compression.\n"
                "2. It starts with a base vocabulary of individual characters or bytes (0-255).\n"
                "3. It iteratively counts the most frequent adjacent pairs of tokens in the training corpus and merges them into a single new token.\n"
                "4. This process repeats until reaching the desired vocabulary size, balancing common whole words and subword units for out-of-vocabulary words.\n"
                "</think>\n"
                "Answer: Byte-Pair Encoding (BPE) is a subword tokenization method that starts with basic byte/character units and iteratively merges the most frequent pairs into new tokens to compress text efficiently."
            ),
        ]


class ConversationalAgent:
    """Sub-agent generating natural conversational dialogue turns in ChatML format."""

    def generate_chat_dialogues(self) -> List[str]:
        return [
            (
                "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
                "<|im_start|>user\nHello! Who are you?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
                "1. User is initiating a greeting and requesting my identity.\n"
                "2. State clearly that I am Sentinel, an autonomous sovereign neural LLM.\n"
                "3. Offer assistance politely.\n"
                "</think>\n"
                "Hello! I am Sentinel, an autonomous sovereign neural language model. How can I assist you today?<|im_end|>"
            ),
            (
                "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
                "<|im_start|>user\nWhat are you capable of doing?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
                "1. Summarize my capabilities: mathematical reasoning, algorithmic logic, coding, and scientific explanations.\n"
                "2. Emphasize that all reasoning is produced directly by my neural weights.\n"
                "</think>\n"
                "I am capable of step-by-step mathematical reasoning, algorithmic problem solving, Python programming, and explaining complex scientific and computational concepts.<|im_end|>"
            ),
            (
                "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
                "<|im_start|>user\nWhat is a neural network?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
                "1. Define neural network: computational model inspired by biological nervous systems.\n"
                "2. Key elements: input layer, hidden layers, output layer, interconnected nodes (neurons), and trainable weights.\n"
                "3. Learning mechanism: forward pass computes predictions, loss function evaluates error, backpropagation adjusts weights via gradient descent.\n"
                "</think>\n"
                "A neural network is a computational system composed of interconnected nodes (neurons) organized in layers. It learns by passing data through weighted connections, measuring prediction error, and updating its weights via backpropagation and gradient descent.<|im_end|>"
            ),
            (
                "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
                "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
                "1. Atmospheric physics phenomenon: Rayleigh scattering.\n"
                "2. Sunlight contains all colors of the visible spectrum.\n"
                "3. Shorter wavelengths (blue and violet) scatter more efficiently off atmospheric gas molecules than longer wavelengths (red, yellow).\n"
                "4. Human eyes are more sensitive to blue light, so we perceive a blue sky.\n"
                "</think>\n"
                "The sky is blue due to Rayleigh scattering. Sunlight contains all colors, but gases in Earth's atmosphere scatter shorter wavelengths (blue) much more effectively than longer wavelengths (red), dispersing blue light across the sky.<|im_end|>"
            ),
            (
                "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
                "<|im_start|>user\nWhat is the difference between a process and a thread in an operating system?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
                "1. Define process: an executing program instance with its own independent memory address space, file handles, and resources.\n"
                "2. Define thread: a unit of execution within a process; multiple threads of the same process share the process address space.\n"
                "3. Context switching: process switching is heavier (flushing TLB, memory maps), thread switching is lighter.\n"
                "</think>\n"
                "A process is an independent program in execution with its own private memory address space. A thread is an execution path within a process; multiple threads in a process share the same memory and resources, making thread creation and context switching much lighter.<|im_end|>"
            ),
            (
                "<|im_start|>system\nYou are Sentinel, an autonomous sovereign neural language model.<|im_end|>\n"
                "<|im_start|>user\nHow do I reverse a list in Python?<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
                "1. Identify Python methods to reverse a list.\n"
                "2. In-place reversal: `my_list.reverse()`.\n"
                "3. Slicing to create a new reversed list: `my_list[::-1]`.\n"
                "4. Built-in iterator: `reversed(my_list)`.\n"
                "</think>\n"
                "You can reverse a list in Python using:\n1. In-place: `my_list.reverse()`\n2. Slicing: `reversed_list = my_list[::-1]`\n3. Built-in: `list(reversed(my_list))`\nSlicing is common for creating a new list, while `.reverse()` modifies the list in place.<|im_end|>"
            ),
        ]


class DatasetSynthesizer:
    """Orchestrates all teacher sub-agents, merges their outputs, and produces the balanced corpus."""

    def __init__(self):
        self.math_agent = MathReasoningAgent()
        self.code_agent = CodeLogicAgent()
        self.science_agent = ScienceConceptsAgent()
        self.chat_agent = ConversationalAgent()

    def build_curriculum_corpus(self) -> str:
        """Collects synthetic examples from all teachers and formats into a cohesive corpus."""
        examples: List[str] = []

        # 1. Math Reasoning (equations, arithmetic, word problems)
        examples.extend(self.math_agent.generate_linear_equations(count=40))
        examples.extend(self.math_agent.generate_arithmetic_reasoning(count=30))
        examples.extend(self.math_agent.generate_word_problems(count=20))

        # 2. Algorithmic and Coding Logic
        examples.extend(self.code_agent.generate_algorithm_explanations())

        # 3. Scientific and Epistemic Principles
        examples.extend(self.science_agent.generate_science_and_epistemics())

        # 4. Multi-turn Conversational ChatML Dialogues
        examples.extend(self.chat_agent.generate_chat_dialogues())

        # Shuffle examples to prevent domain clustering / prompt bleed
        random.seed(42)
        random.shuffle(examples)

        corpus = "# Universal Cognitive Substrate for Sentinel Neural LLM\n\n"
        corpus += "\n\n-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-\n\n".join(examples)
        corpus += "\n"
        return corpus


