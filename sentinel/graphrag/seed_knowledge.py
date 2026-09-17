"""
Sentinel Knowledge Graph Seed Substrate
=======================================
Populates interconnected foundational knowledge across Mathematics,
Computer Architecture, Algorithms, Natural Sciences, and AI Epistemics.
"""

from __future__ import annotations

from sentinel.graphrag.graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge


def populate_universal_knowledge_graph(kg: KnowledgeGraph) -> KnowledgeGraph:
    """Populates the knowledge graph with multi-domain interconnected concepts and relations."""

    # --------------------------------------------------------------------------
    # 1. Mathematics & Logic
    # --------------------------------------------------------------------------
    kg.add_node(KnowledgeNode("arithmetic", "Arithmetic", "Mathematics", "Study of numbers and fundamental operations: addition, subtraction, multiplication, division."))
    kg.add_node(KnowledgeNode("algebra", "Algebra", "Mathematics", "Mathematical branch dealing with variables, equations, and abstract manipulation of symbols."))
    kg.add_node(KnowledgeNode("linear_equations", "Linear Equations", "Mathematics", "Equations of the first degree where terms are constants or products of a constant and single variable."))
    kg.add_node(KnowledgeNode("calculus", "Calculus", "Mathematics", "Mathematical study of continuous change, encompassing limits, derivatives, and integrals."))
    kg.add_node(KnowledgeNode("optimization", "Optimization", "Mathematics", "Selection of the best element from a set of available alternatives by minimizing or maximizing a function."))
    kg.add_node(KnowledgeNode("gradient_descent", "Gradient Descent", "Mathematics", "First-order iterative optimization algorithm for finding a local minimum of a differentiable function."))

    kg.add_edge(KnowledgeEdge("arithmetic", "algebra", "DEPENDS_ON", "Algebra generalizes arithmetic operations through symbolic variables."))
    kg.add_edge(KnowledgeEdge("algebra", "linear_equations", "DERIVES", "Linear equations are the fundamental first-degree structures in algebra."))
    kg.add_edge(KnowledgeEdge("algebra", "calculus", "DEPENDS_ON", "Calculus requires algebraic manipulation of functional limits and difference quotients."))
    kg.add_edge(KnowledgeEdge("calculus", "gradient_descent", "DERIVES", "Gradient descent uses partial derivatives (gradients) from multivariate calculus to determine step direction."))
    kg.add_edge(KnowledgeEdge("gradient_descent", "optimization", "IMPLEMENTS", "Gradient descent iteratively minimizes loss functions to achieve optimal parameter weights."))

    # --------------------------------------------------------------------------
    # 2. Computer Architecture & Operating Systems
    # --------------------------------------------------------------------------
    kg.add_node(KnowledgeNode("logic_gates", "Logic Gates", "Computer Systems", "Physical electronic devices implementing Boolean functions using transistors."))
    kg.add_node(KnowledgeNode("cpu", "Central Processing Unit (CPU)", "Computer Systems", "Primary electronic circuitry that executes machine instructions of a computer program."))
    kg.add_node(KnowledgeNode("cpu_registers", "CPU Registers", "Computer Systems", "Extremely fast, small memory storage cells located directly inside the CPU core."))
    kg.add_node(KnowledgeNode("memory_hierarchy", "Memory Hierarchy", "Computer Systems", "Layered computer storage structure balancing latency and capacity: Registers -> L1/L2/L3 Cache -> RAM -> SSD."))
    kg.add_node(KnowledgeNode("ram", "Random Access Memory (RAM)", "Computer Systems", "Volatile primary storage providing fast uniform-time byte read and write access to active processes."))
    kg.add_node(KnowledgeNode("operating_system", "Operating System (OS)", "Computer Systems", "System software managing computer hardware, resource allocation, and scheduling execution."))
    kg.add_node(KnowledgeNode("process", "Process", "Computer Systems", "An executing program instance with dedicated private address space, environment variables, and resources."))
    kg.add_node(KnowledgeNode("thread", "Thread", "Computer Systems", "Smallest sequence of programmed instructions managed independently by an OS scheduler, sharing process memory."))
    kg.add_node(KnowledgeNode("python_runtime", "Python Runtime", "Computer Systems", "Bytecode interpreter and virtual machine (CPython) managing execution, objects, and reference counting."))

    kg.add_edge(KnowledgeEdge("logic_gates", "cpu", "PART_OF", "Logic gates form arithmetic logic units (ALU) and control logic within the CPU."))
    kg.add_edge(KnowledgeEdge("cpu", "cpu_registers", "PART_OF", "Registers reside directly on CPU silicon with sub-nanosecond access latencies."))
    kg.add_edge(KnowledgeEdge("cpu_registers", "memory_hierarchy", "PART_OF", "Registers represent the fastest and most expensive tier of the computer memory hierarchy."))
    kg.add_edge(KnowledgeEdge("ram", "memory_hierarchy", "PART_OF", "RAM serves as the primary volatile storage tier below CPU caches."))
    kg.add_edge(KnowledgeEdge("operating_system", "process", "DERIVES", "The OS kernel creates processes to isolate program execution into virtual address spaces."))
    kg.add_edge(KnowledgeEdge("process", "thread", "PART_OF", "Processes contain one or more threads that execute concurrently within the same shared address space."))
    kg.add_edge(KnowledgeEdge("thread", "process", "CONTRASTS", "Threads share memory and have low context-switch overhead, while processes isolate memory with high context-switch cost."))
    kg.add_edge(KnowledgeEdge("python_runtime", "process", "IMPLEMENTS", "The CPython virtual machine executes as an OS process, managing an internal heap via reference counting."))

    # --------------------------------------------------------------------------
    # 3. Data Structures & Algorithms
    # --------------------------------------------------------------------------
    kg.add_node(KnowledgeNode("array", "Array", "Data Structures", "Contiguous block of computer memory storing elements of the same type with O(1) indexed access."))
    kg.add_node(KnowledgeNode("hash_table", "Hash Table", "Data Structures", "Associative data structure mapping keys to values using a hash function with average O(1) lookups."))
    kg.add_node(KnowledgeNode("linked_list", "Linked List", "Data Structures", "Linear collection of data elements called nodes, where each node points to the next via a pointer."))
    kg.add_node(KnowledgeNode("binary_tree", "Binary Tree", "Data Structures", "Hierarchical tree data structure in which each node has at most two children: left and right."))
    kg.add_node(KnowledgeNode("divide_and_conquer", "Divide and Conquer", "Algorithms", "Algorithmic paradigm that breaks a problem into smaller subproblems, solves them recursively, and combines results."))
    kg.add_node(KnowledgeNode("quicksort", "Quicksort", "Algorithms", "Divide-and-conquer sorting algorithm partitioning an array around a pivot element with average O(N log N) complexity."))
    kg.add_node(KnowledgeNode("binary_search", "Binary Search", "Algorithms", "Search algorithm that finds the position of a target value within a sorted array in O(log N) time."))
    kg.add_node(KnowledgeNode("bfs_dfs", "BFS and DFS", "Algorithms", "Graph traversal algorithms: BFS uses a FIFO queue for shortest paths, DFS uses recursion/stack for deep paths."))

    kg.add_edge(KnowledgeEdge("array", "memory_hierarchy", "DEPENDS_ON", "Contiguous array storage maximizes CPU cache locality, drastically reducing memory bus latency."))
    kg.add_edge(KnowledgeEdge("divide_and_conquer", "quicksort", "IMPLEMENTS", "Quicksort implements divide-and-conquer by partitioning the array around a pivot and sorting sub-arrays."))
    kg.add_edge(KnowledgeEdge("array", "quicksort", "APPLIES_TO", "Quicksort operates in-place on contiguous array partitions with cache-friendly access patterns."))
    kg.add_edge(KnowledgeEdge("array", "binary_search", "APPLIES_TO", "Binary search requires contiguous random indexed access to evaluate the middle element in O(1) time."))
    kg.add_edge(KnowledgeEdge("hash_table", "array", "DEPENDS_ON", "Hash tables use an underlying array of buckets indexed by the hashed key modulo table size."))
    kg.add_edge(KnowledgeEdge("binary_tree", "bfs_dfs", "APPLIES_TO", "Tree traversal algorithms (preorder, inorder, postorder, level-order) are special cases of DFS and BFS."))

    # --------------------------------------------------------------------------
    # 4. Physical Sciences & Epistemics
    # --------------------------------------------------------------------------
    kg.add_node(KnowledgeNode("energy_conservation", "Conservation of Energy", "Physics", "First law of thermodynamics: total energy in an isolated system remains constant; energy only transforms."))
    kg.add_node(KnowledgeNode("thermodynamics", "Thermodynamics", "Physics", "Physical branch dealing with heat, work, temperature, and statistical behavior of energy transformations."))
    kg.add_node(KnowledgeNode("electromagnetism", "Electromagnetism", "Physics", "Fundamental interaction between electrically charged particles, describing electromagnetic radiation."))
    kg.add_node(KnowledgeNode("light_waves", "Light Waves", "Physics", "Electromagnetic radiation visible to the human eye, traveling as oscillating electric and magnetic fields."))
    kg.add_node(KnowledgeNode("rayleigh_scattering", "Rayleigh Scattering", "Physics", "Elastic scattering of light by particles much smaller than the wavelength of the radiation, inversely proportional to lambda^4."))
    kg.add_node(KnowledgeNode("blue_sky", "Blue Sky Color", "Physics", "Phenomenon where Earth's atmosphere preferentially scatters short-wavelength blue sunlight across the sky."))
    kg.add_node(KnowledgeNode("science", "Empirical Science", "Epistemics", "Systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe."))
    kg.add_node(KnowledgeNode("empirical_falsification", "Empirical Falsification", "Epistemics", "Karl Popper's principle: scientific theories cannot be definitively verified, only falsified by observational testing."))

    kg.add_edge(KnowledgeEdge("energy_conservation", "thermodynamics", "PART_OF", "Conservation of energy forms the foundational First Law of Thermodynamics."))
    kg.add_edge(KnowledgeEdge("electromagnetism", "light_waves", "DERIVES", "Visible light is electromagnetic radiation propagating at speed c according to Maxwell's equations."))
    kg.add_edge(KnowledgeEdge("light_waves", "rayleigh_scattering", "APPLIES_TO", "Rayleigh scattering efficiency scales as 1/(wavelength^4), heavily favoring blue/violet light."))
    kg.add_edge(KnowledgeEdge("rayleigh_scattering", "blue_sky", "EXPLAINS", "Because blue light scatters 10x more intensely than red light, scattered blue wavelengths saturate the daytime sky."))
    kg.add_edge(KnowledgeEdge("empirical_falsification", "science", "EXPLAINS", "Scientific validity requires hypotheses to be empirically testable and refutable by real observation."))

    # --------------------------------------------------------------------------
    # 5. Neural Networks & Transformer AI Architecture
    # --------------------------------------------------------------------------
    kg.add_node(KnowledgeNode("neural_network", "Neural Network", "AI Architecture", "Computational graph of interconnected nodes with learned weights modeling non-linear representations."))
    kg.add_node(KnowledgeNode("backpropagation", "Backpropagation", "AI Architecture", "Algorithm calculating gradients of the loss function with respect to weights via the chain rule of calculus."))
    kg.add_node(KnowledgeNode("transformer", "Transformer", "AI Architecture", "Neural architecture replacing recurrence with self-attention mechanisms for parallel sequence modeling."))
    kg.add_node(KnowledgeNode("self_attention", "Self-Attention (GQA)", "AI Architecture", "Mechanism calculating dynamically weighted interactions between tokens in a sequence."))
    kg.add_node(KnowledgeNode("sa_moe", "Mixture of Experts (SA-MoE)", "AI Architecture", "Sparse architectural scaling activating top-k expert subnetworks per token plus shared experts."))
    kg.add_node(KnowledgeNode("agi", "Artificial General Intelligence (AGI)", "AI Architecture", "Hypothetical machine intelligence matching human cognitive versatility across intellectual tasks."))
    kg.add_node(KnowledgeNode("asi", "Artificial Superintelligence (ASI)", "AI Architecture", "Intelligence surpassing all human capability in science, strategy, logic, and recursive self-improvement."))

    kg.add_edge(KnowledgeEdge("calculus", "backpropagation", "DERIVES", "Backpropagation recursively applies the multi-variable calculus chain rule to calculate weight gradients."))
    kg.add_edge(KnowledgeEdge("gradient_descent", "neural_network", "APPLIES_TO", "Neural network parameters are updated by subtracting learning_rate * gradient during training."))
    kg.add_edge(KnowledgeEdge("neural_network", "transformer", "DERIVES", "Transformers are a deep neural network architecture designed for scalable sequence-to-sequence modeling."))
    kg.add_edge(KnowledgeEdge("self_attention", "transformer", "PART_OF", "Grouped-Query Attention (GQA) enables the Transformer to attend across different sequence positions efficiently."))
    kg.add_edge(KnowledgeEdge("sa_moe", "transformer", "IMPLEMENTS", "Mixture-of-Experts expands parameter capacity without proportional computational cost per token."))
    kg.add_edge(KnowledgeEdge("transformer", "agi", "DERIVES", "Large multimodal Transformer architectures serve as the dominant empirical foundation towards AGI."))
    kg.add_edge(KnowledgeEdge("agi", "asi", "DERIVES", "Once AGI achieves recursive algorithmic and architectural self-improvement, cognitive capability scales towards ASI."))

    return kg
