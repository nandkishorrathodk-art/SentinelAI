"""
Sentinel 5-Bucket Official Evaluation Framework
===============================================
1. Language (fluency, grammar, repetition penalty)
2. Knowledge (Parametric vs GraphRAG retrieval)
3. Reasoning (Held-out unseen numerical & logic test set)
4. Systems Knowledge (CPU, memory, OS, Python internals)
5. Reliability (Hallucination resistance and uncertainty bounds)
"""

from sentinel.evaluation.benchmark_suite import SentinelFiveBucketBenchmark

__all__ = ["SentinelFiveBucketBenchmark"]

