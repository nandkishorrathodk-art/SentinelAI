"""
Sentinel Sovereign Neural LLM — Interactive Chat CLI
====================================================
Real-time interactive terminal chat with Sentinel.
Streams step-by-step <think> reasoning traces and final conversational answers.

Usage:
  python -m forge.chat
  python -m forge.chat --checkpoint /path/to/sentinel_v2.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import torch

from forge.generate import load_sentinel_model, generate_text
from forge.chat_template import (
    format_chat_prompt,
    parse_reasoning_and_response,
    DEFAULT_SYSTEM_PROMPT,
    IM_END,
)


def run_interactive_chat(
    checkpoint_path: str,
    device: str = "cpu",
    temperature: float = 0.6,
    top_k: int = 30,
    max_tokens: int = 120
):
    print("=" * 70)
    print("SENTINEL SOVEREIGN NEURAL LLM — INTERACTIVE CHAT TERMINAL")
    print("Zero External APIs | 100% Native Neural Weights (.pt)")
    print("=" * 70)

    model, tokenizer = load_sentinel_model(checkpoint_path, device=device)

    print("\n[+] Sentinel ready! Commands: 'exit' to quit, 'clear' to reset context.")
    print("-" * 70)

    messages: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("\n[User] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Sentinel chat session. Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Session ended.")
            break

        if user_input.lower() == "clear":
            messages.clear()
            print("[*] Conversation history cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        # Format ChatML prompt
        prompt = format_chat_prompt(messages)

        print("\n[Sentinel Thinking...]")
        t0 = time.time()
        raw_output, speed = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device
        )
        elapsed = time.time() - t0

        # Extract newly generated text (strip prompt prefix)
        if raw_output.startswith(prompt):
            gen_content = raw_output[len(prompt):]
        else:
            gen_content = raw_output

        # Clean trailing tokens
        if IM_END in gen_content:
            gen_content = gen_content.split(IM_END)[0]

        reasoning, answer = parse_reasoning_and_response(gen_content)

        # Display step-by-step thinking trace if present
        if reasoning:
            print("--------------------------------------------------")
            print("🧠 REASONING (<think>):")
            for line in reasoning.splitlines():
                if line.strip():
                    print(f"  {line.strip()}")
            print("--------------------------------------------------")

        # Display final response
        print(f"\n[Sentinel]:\n{answer if answer.strip() else '(Completed reasoning)'}")
        print(f"\n[Stats: {speed:.1f} tokens/sec | {elapsed:.2f}s]")

        # Append assistant turn to history
        assistant_turn = ""
        if reasoning:
            assistant_turn += f"<think>\n{reasoning}\n</think>\n"
        assistant_turn += answer
        messages.append({"role": "assistant", "content": assistant_turn})


def main():
    parser = argparse.ArgumentParser(description="Sentinel Interactive Chat")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_v2.pt",
        help="Path to trained Sentinel checkpoint"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K sampling")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max new tokens")

    args = parser.parse_args()

    # Fallback to sentinel_final.pt if sentinel_v2.pt doesn't exist
    chk = args.checkpoint
    if not os.path.exists(chk):
        chk = "c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_final.pt"

    run_interactive_chat(
        checkpoint_path=chk,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()

