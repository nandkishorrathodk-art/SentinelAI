"""
Sentinel Sub-Agent Training & Self-Improvement Pipeline
======================================================
Executes the end-to-end self-improvement loop:
1. Sub-Agent Teacher Generation (Math, Code, Science).
2. Quality Gate & Code AST Verification.
3. Red-Team Probing on Baseline Checkpoint.
4. Targeted Countermeasure Synthesis.
5. Neural Weight Fine-Tuning & Training.
6. Post-Training Empirical Evaluation & Comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from forge.generate import load_sentinel_model, generate_text
from forge.training.dataset_agents import DatasetSynthesizer
from forge.training.quality_gate import QualityGate
from forge.training.red_team_critic import RedTeamCritic
from forge.training.trainer import SentinelTrainer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


def run_pipeline(
    baseline_checkpoint: str,
    output_checkpoint: str,
    epochs: int = 3,
    lr: float = 1e-4,
    device: str = "cpu"
):
    print("=" * 70)
    print("SENTINEL SUB-AGENT TRAINING & SELF-IMPROVEMENT PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Step 1: Sub-Agent Teacher Data Generation
    # --------------------------------------------------------------------------
    print("\n[*] Phase 1: Sub-Agent Teachers synthesizing training curriculum...")
    t0 = time.time()
    synthesizer = DatasetSynthesizer()
    raw_corpus = synthesizer.build_curriculum_corpus()
    print(f"[+] Sub-Agents generated raw corpus ({len(raw_corpus):,} characters).")

    # --------------------------------------------------------------------------
    # Step 2: Quality Gate & AST Code Verification
    # --------------------------------------------------------------------------
    print("\n[*] Phase 2: Quality Gate verifying <think> structure and code syntax...")
    gate = QualityGate()
    raw_examples = raw_corpus.split("\n\n-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-\n\n")
    verified_examples = gate.filter_dataset(raw_examples)
    print(f"[+] Quality Gate Results:")
    print(f"    - Inspected: {gate.stats['total_inspected']}")
    print(f"    - Passed:    {gate.stats['passed']}")
    print(f"    - Rejected:  {gate.stats['total_inspected'] - gate.stats['passed']}")

    # --------------------------------------------------------------------------
    # Step 3: Load Baseline Model & Red-Team Probe
    # --------------------------------------------------------------------------
    print(f"\n[*] Phase 3: Loading baseline model to probe weaknesses...")
    model, tokenizer = load_sentinel_model(baseline_checkpoint, device=device)
    critic = RedTeamCritic(model, tokenizer, device=device)

    print("[*] Running Red-Team baseline evaluation...")
    pre_eval = critic.evaluate_checkpoint()
    print(f"[+] Baseline Overall Accuracy Score: {pre_eval['overall_score'] * 100:.1f}%")
    for probe_id, res in pre_eval.items():
        if probe_id != "overall_score":
            status = "PASSED" if res["passed"] else "FAILED"
            print(f"    - {probe_id}: {status} (Score: {res['score']:.2f}, Bleed: {res['conceptual_bleed']})")

    # --------------------------------------------------------------------------
    # Step 4: Red-Team Countermeasure Synthesis
    # --------------------------------------------------------------------------
    print("\n[*] Phase 4: Synthesizing targeted counter-measures from Red-Team probe...")
    countermeasures = critic.synthesize_countermeasures(pre_eval)
    print(f"[+] Generated {len(countermeasures)} targeted countermeasure examples.")
    verified_examples.extend(countermeasures)

    # Reconstruct final training corpus
    final_corpus = "\n\n-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-\n\n".join(verified_examples)

    # --------------------------------------------------------------------------
    # Step 5: Neural Weight Training
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Step 6: Post-Training Evaluation on Upgraded Model
    # --------------------------------------------------------------------------
    print(f"\n[*] Phase 6: Empirical Post-Training Evaluation on {output_checkpoint}...")
    post_critic = RedTeamCritic(model, tokenizer, device=device)
    post_eval = post_critic.evaluate_checkpoint()
    print(f"[+] Upgraded Model Overall Accuracy Score: {post_eval['overall_score'] * 100:.1f}%")
    for probe_id, res in post_eval.items():
        if probe_id != "overall_score":
            status = "PASSED" if res["passed"] else "FAILED"
            print(f"    - {probe_id}: {status} (Score: {res['score']:.2f}, Bleed: {res['conceptual_bleed']})")

    print("\n" + "=" * 70)
    print(f"SELF-IMPROVEMENT CYCLE COMPLETE!")
    print(f"Baseline Score: {pre_eval['overall_score'] * 100:.1f}%  -->  Upgraded Score: {post_eval['overall_score'] * 100:.1f}%")
    print(f"Model saved to: {output_checkpoint}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Sentinel Sub-Agent Training Pipeline")
    parser.add_argument(
        "--baseline",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_final.pt",
        help="Path to baseline checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_v2.pt",
        help="Path to save upgraded checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    run_pipeline(
        baseline_checkpoint=args.baseline,
        output_checkpoint=args.output,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()

