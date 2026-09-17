import os
import time
import torch
import torch.nn as nn

try:
    import openvino as ov
except ImportError:
    ov = None

def export_to_openvino(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_dir: str,
    quantize_int8: bool = False
) -> str:
    """
    Exports a PyTorch model to ONNX and then compiles to OpenVINO IR.
    Supports INT8 Post-training quantization placeholder.
    """
    if ov is None:
        raise ImportError("OpenVINO is not installed. Please 'pip install openvino'.")
        
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "sentinel.onnx")
    xml_path = os.path.join(output_dir, "sentinel.xml")
    
    print(f"Exporting PyTorch model to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_length'},
            'logits': {0: 'batch_size', 1: 'seq_length'}
        }
    )
    
    print(f"Converting ONNX to OpenVINO IR: {xml_path}")
    core = ov.Core()
    ov_model = core.read_model(onnx_path)
    
    if quantize_int8:
        print("Note: INT8 Quantization requires nncf (Neural Network Compression Framework).")
        # Example: nncf.quantize(ov_model, calibration_dataset)
        
    ov.save_model(ov_model, xml_path)
    print("OpenVINO export complete.")
    return xml_path


class OpenVINOInferenceWrapper:
    """
    OpenVINO runtime inference wrapper targeting Intel Core Ultra NPU, GPU, or CPU.
    """
    def __init__(self, model_xml: str, device: str = "NPU"):
        if ov is None:
            raise ImportError("OpenVINO is not installed.")
        self.core = ov.Core()
        devices = self.core.available_devices
        if device not in devices:
            print(f"Warning: {device} not found in available devices {devices}. Falling back to CPU.")
            device = "CPU"
            
        print(f"Compiling model for {device}...")
        self.compiled_model = self.core.compile_model(model_xml, device)
        self.infer_request = self.compiled_model.create_infer_request()
        
    def infer(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Runs inference and returns the output tensor (logits).
        """
        np_input = input_ids.cpu().numpy()
        self.infer_request.infer({'input_ids': np_input})
        logits_np = self.infer_request.get_output_tensor(0).data
        return torch.from_numpy(logits_np)
        
    def benchmark(self, dummy_input: torch.Tensor, iterations: int = 100) -> float:
        """
        Throughput benchmarking (tokens/sec).
        """
        np_input = dummy_input.cpu().numpy()
        
        # Warmup
        for _ in range(10):
            self.infer_request.infer({'input_ids': np_input})
            
        start_time = time.time()
        for _ in range(iterations):
            self.infer_request.infer({'input_ids': np_input})
        end_time = time.time()
        
        duration = end_time - start_time
        total_tokens = dummy_input.numel() * iterations
        tps = total_tokens / duration
        print(f"Benchmark: {tps:.2f} tokens/sec over {iterations} iterations.")
        return tps
