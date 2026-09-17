"""
Sentinel GraphRAG Training & Evolution Pipeline
===============================================
Orchestrates:
1. Knowledge Graph population with universal foundational domains.
2. GraphRAG Teacher Sub-Agent Swarm execution to synthesize multi-hop lessons.
3. Quality Gate verification (AST code check + <think> validation).
4. PyTorch Neural Training on Sentinel's weights.
5. Export of graph-grounded Sentinel checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from sentinel.graphrag.graph import KnowledgeGraph
from sentinel.graphrag.seed_knowledge import populate_universal_knowledge_graph
from sentinel.graphrag.teacher_agents import GraphRAGTeacherSwarm
from forge.training.quality_gate import QualityGate
from forge.training.dataset_agents import DatasetSynthesizer
from forge.training.trainer import SentinelTrainer
from forge.generate import load_sentinel_model, generate_text

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


def run_graphrag_training_pipeline(
    baseline_checkpoint: str,
    output_checkpoint: str,
    epochs: int = 3,
    lr: float = 7e-5,
    device: str = "cpu"
):
    print("=" * 70)
    print("SENTINEL GRAPHRAG TEACHER SWARM & EVOLUTION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Step 1: Initialize & Seed Knowledge Graph
    # --------------------------------------------------------------------------
    print("\n[*] Phase 1: Building Foundational Knowledge Graph...")
    kg = KnowledgeGraph()
    populate_universal_knowledge_graph(kg)
    stats = kg.stats()
    print(f"[+] Knowledge Graph Built:")
    print(f"    - Total Concept Nodes: {stats['total_nodes']}")
    print(f"    - Total Relational Edges: {stats['total_edges']}")
    print(f"    - Domains: {', '.join(stats['domains'])}")

    # --------------------------------------------------------------------------
    # Step 2: Teacher Sub-Agents Harvest Multi-Hop Lessons
    # --------------------------------------------------------------------------
    print("\n[*] Phase 2: Teacher Sub-Agents Swarm harvesting graph lessons...")
    swarm = GraphRAGTeacherSwarm(kg)
    graph_lessons = swarm.harvest_all_lessons()
    print(f"[+] Teacher Swarm synthesized {len(graph_lessons)} relational lessons.")

    # Complement with base math and algorithms from Synthesizer
    base_synth = DatasetSynthesizer()
    base_corpus = base_synth.build_curriculum_corpus()
    base_examples = base_corpus.split("\n\n-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-\n\n")

    combined_examples = graph_lessons + base_examples

    # --------------------------------------------------------------------------
    # Step 3: Quality Gate Verification
    # --------------------------------------------------------------------------
    print("\n[*] Phase 3: Quality Gate verifying code AST and <think> integrity...")
    gate = QualityGate()
    verified_examples = gate.filter_dataset(combined_examples)
    print(f"[+] Quality Gate Results: {gate.stats['passed']}/{gate.stats['total_inspected']} passed.")

    final_corpus = "\n\n-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-\n\n".join(verified_examples)

    # --------------------------------------------------------------------------
    # Step 4: Load Latest Model & Train on GraphRAG Curriculum
    # --------------------------------------------------------------------------
    print(f"\n[*] Phase 4: Loading Sentinel model from {baseline_checkpoint}...")
    model, tokenizer = load_sentinel_model(baseline_checkpoint, device=device)

    print(f"\n[*] Phase 5: Commencing PyTorch Neural Training ({epochs} epochs)...")
    trainer = SentinelTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=lr
    )
    loss_history = trainer.train(
        corpus=final_corpus,
        epochs=epochs,
        batch_size=8,
        seq_len=128,
        output_checkpoint=output_checkpoint
    )

    print("\n" + "=" * 70)
    print("GRAPHRAG TRAINING COMPLETE!")
    print(f"Saved to: {output_checkpoint}")
    print(f"Final Loss: {loss_history[-1]:.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Sentinel GraphRAG Training Pipeline")
    parser.add_argument(
        "--baseline",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_v3.pt",
        help="Base checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_graphrag.pt",
        help="Output checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=7e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    chk = args.baseline
    if not os.path.exists(chk):
        chk = "c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_v2.pt"

    run_graphrag_training_pipeline(
        baseline_checkpoint=chk,
        output_checkpoint=args.output,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()

