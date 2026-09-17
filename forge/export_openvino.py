"""
Sentinel Intel NPU & OpenVINO Export Pipeline
=============================================
Converts trained Sentinel PyTorch checkpoints (.pt) to ONNX and OpenVINO IR (.xml / .bin)
for accelerated local neural execution on Intel Core Ultra NPU and Arc GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn

from forge.generate import load_sentinel_model, BytePairTokenizer, SentinelTransformer


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class SentinelOnnxWrapper(nn.Module):
    """Clean wrapper exposing single-tensor forward pass for ONNX and OpenVINO export."""

    def __init__(self, model: SentinelTransformer):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(input_ids)
        return logits


def export_to_openvino_direct(
    checkpoint_path: str,
    ir_output_dir: str = "c:/Users/nandk/_society/sentinel-ai/notebooks/output/openvino_ir",
    model_name: str = "sentinel",
    seq_len: int = 128,
    device: str = "cpu"
) -> str:
    """Converts PyTorch model directly to OpenVINO IR (.xml / .bin) for Intel NPU."""
    import openvino as ov

    print(f"[*] Loading PyTorch checkpoint: {checkpoint_path}")
    model, tokenizer = load_sentinel_model(checkpoint_path, device=device)
    wrapper = SentinelOnnxWrapper(model)
    wrapper.eval()

    core = ov.Core()
    print(f"[+] OpenVINO Available Accelerators: {core.available_devices}")

    dummy_input = torch.zeros(1, seq_len, dtype=torch.long, device=device)

    print(f"[*] Compiling Sentinel SA-MoE directly to OpenVINO IR...")
    ov_model = ov.convert_model(wrapper, example_input=dummy_input)

    # NPUs require bounded or static shapes for hardware SRAM tile layout
    print(f"[*] Binding static shape [1, {seq_len}] for Intel NPU hardware compatibility...")
    ov_model.reshape([1, seq_len])

    os.makedirs(ir_output_dir, exist_ok=True)
    xml_path = os.path.join(ir_output_dir, f"{model_name}.xml")
    bin_path = os.path.join(ir_output_dir, f"{model_name}.bin")

    ov.save_model(ov_model, xml_path)
    print(f"[+] OpenVINO IR Export Successful!")
    print(f"    - XML: {xml_path}")
    print(f"    - BIN: {bin_path}")

    # Try accelerators in priority order: NPU -> Intel Arc GPU -> CPU
    devices_to_try = ["NPU", "GPU", "CPU"]
    compiled_model = None
    active_dev = None
    for dev in devices_to_try:
        if dev in core.available_devices:
            try:
                print(f"[*] Attempting OpenVINO compilation on [{dev}] accelerator...")
                compiled_model = core.compile_model(ov_model, dev)
                active_dev = dev
                print(f"[+] SUCCESS! Model compiled and loaded on {dev} hardware accelerator!")
                break
            except Exception as e:
                print(f"[!] Compilation on [{dev}] passed: {e}")

    return xml_path


def main():
    parser = argparse.ArgumentParser(description="Sentinel Intel NPU OpenVINO Export")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_v2.pt",
        help="Path to trained .pt checkpoint"
    )
    parser.add_argument(
        "--openvino_dir",
        type=str,
        default="c:/Users/nandk/_society/sentinel-ai/notebooks/output/openvino_ir",
        help="Output directory for OpenVINO IR"
    )

    args = parser.parse_args()

    chk = args.checkpoint
    if not os.path.exists(chk):
        chk = "c:/Users/nandk/_society/sentinel-ai/notebooks/output/sentinel_final.pt"

    export_to_openvino_direct(chk, args.openvino_dir)


if __name__ == "__main__":
    main()

