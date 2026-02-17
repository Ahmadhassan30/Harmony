"""Export PyTorch models to ONNX format for production inference."""

from __future__ import annotations

from pathlib import Path

import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 1, 128, 1024),
    opset_version: int = 18,
    dynamic_axes: dict | None = None,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model.
        output_path: Path for the .onnx file.
        input_shape: Example input tensor shape.
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axis configuration.

    Returns:
        Path to the exported ONNX model.
    """
    model.eval()
    device = next(model.parameters()).device

    dummy_input = torch.randn(*input_shape, device=device)

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 3: "time"},
            "output": {0: "batch_size"},
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model to: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")

    return output_path


def quantize_onnx(input_path: str, output_path: str | None = None) -> str:
    """Quantize an ONNX model to INT8 for faster inference.

    Args:
        input_path: Path to the source .onnx file.
        output_path: Path for the quantized .onnx file.

    Returns:
        Path to the quantized model.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if output_path is None:
        output_path = input_path.replace(".onnx", "_int8.onnx")

    quantize_dynamic(input_path, output_path, weight_type=QuantType.QInt8)

    original_size = Path(input_path).stat().st_size
    quantized_size = Path(output_path).stat().st_size

    print(f"Quantized model: {output_path}")
    print(f"  Original: {original_size / 1024 / 1024:.1f} MB")
    print(f"  Quantized: {quantized_size / 1024 / 1024:.1f} MB")
    print(f"  Compression: {(1 - quantized_size / original_size) * 100:.1f}%")

    return output_path
