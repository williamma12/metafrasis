"""
Export PyTorch models to ONNX format.

ONNX enables deployment across different runtimes and hardware.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...],
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: Dict[str, Dict[int, str]] = None,
    opset_version: int = 14,
    simplify: bool = True,
) -> Path:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path for the output ONNX file
        input_shape: Shape of the input tensor (including batch dimension)
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes specification for variable-length inputs
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model

    Returns:
        Path to the exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Create dummy input
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    # Default names
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # Default dynamic axes for batch dimension
    if dynamic_axes is None:
        dynamic_axes = {
            name: {0: "batch_size"} for name in input_names + output_names
        }

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx.load(str(output_path))
            simplified_model, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified_model, str(output_path))
        except ImportError:
            pass  # onnxsim not installed

    return output_path


def export_crnn_to_onnx(
    model: nn.Module,
    output_path: Path,
    img_height: int = 32,
    img_width: int = 128,
) -> Path:
    """
    Export a CRNN model to ONNX.

    Args:
        model: CRNN model
        output_path: Output path
        img_height: Input image height
        img_width: Input image width

    Returns:
        Path to exported model
    """
    return export_to_onnx(
        model,
        output_path,
        input_shape=(1, 1, img_height, img_width),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size", 3: "width"},
            "logits": {0: "sequence_length", 1: "batch_size"},
        },
    )


def export_craft_to_onnx(
    model: nn.Module,
    output_path: Path,
    img_size: int = 768,
) -> Path:
    """
    Export a CRAFT model to ONNX.

    Args:
        model: CRAFT model
        output_path: Output path
        img_size: Input image size

    Returns:
        Path to exported model
    """
    return export_to_onnx(
        model,
        output_path,
        input_shape=(1, 3, img_size, img_size),
        input_names=["image"],
        output_names=["region_map", "affinity_map"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "region_map": {0: "batch_size"},
            "affinity_map": {0: "batch_size"},
        },
    )


def export_db_to_onnx(
    model: nn.Module,
    output_path: Path,
    img_size: int = 640,
) -> Path:
    """
    Export a DB model to ONNX.

    Args:
        model: DBNet model
        output_path: Output path
        img_size: Input image size

    Returns:
        Path to exported model
    """
    return export_to_onnx(
        model,
        output_path,
        input_shape=(1, 3, img_size, img_size),
        input_names=["image"],
        output_names=["prob_map", "thresh_map", "binary_map"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "prob_map": {0: "batch_size"},
            "thresh_map": {0: "batch_size"},
            "binary_map": {0: "batch_size"},
        },
    )


def verify_onnx_model(
    onnx_path: Path,
    pytorch_model: nn.Module,
    input_shape: Tuple[int, ...],
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Verify that ONNX model produces same outputs as PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        input_shape: Shape of test input
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if outputs match within tolerance
    """
    import onnxruntime as ort
    import numpy as np

    # Create test input
    device = next(pytorch_model.parameters()).device
    test_input = torch.randn(*input_shape, device=device)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        if isinstance(pytorch_output, tuple):
            pytorch_outputs = [o.cpu().numpy() for o in pytorch_output]
        else:
            pytorch_outputs = [pytorch_output.cpu().numpy()]

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: test_input.cpu().numpy()})

    # Compare outputs
    for pytorch_out, onnx_out in zip(pytorch_outputs, onnx_outputs):
        if not np.allclose(pytorch_out, onnx_out, rtol=rtol, atol=atol):
            return False

    return True


def get_onnx_model_info(onnx_path: Path) -> Dict[str, Any]:
    """
    Get information about an ONNX model.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        Dictionary with model information
    """
    import onnx

    model = onnx.load(str(onnx_path))

    # Get inputs
    inputs = []
    for inp in model.graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        inputs.append({
            "name": inp.name,
            "shape": shape,
            "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
        })

    # Get outputs
    outputs = []
    for out in model.graph.output:
        shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        outputs.append({
            "name": out.name,
            "shape": shape,
            "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
        })

    # Get model size
    import os
    file_size = os.path.getsize(onnx_path)

    return {
        "opset_version": model.opset_import[0].version,
        "inputs": inputs,
        "outputs": outputs,
        "file_size_mb": file_size / (1024 * 1024),
        "n_nodes": len(model.graph.node),
    }
