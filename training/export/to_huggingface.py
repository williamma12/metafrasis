"""
Upload models to HuggingFace Hub.

Provides functionality to:
- Upload PyTorch models
- Upload ONNX models
- Upload LoRA adapters
- Create model cards
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json


def upload_to_hub(
    model_path: Path,
    repo_id: str,
    model_type: str = "pytorch",
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload model",
    model_card: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Upload a model to HuggingFace Hub.

    Args:
        model_path: Path to the model file or directory
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        model_type: Type of model ("pytorch", "onnx", "lora")
        token: HuggingFace API token
        private: Whether to make the repo private
        commit_message: Git commit message
        model_card: Optional model card metadata

    Returns:
        URL of the uploaded model
    """
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Create repo if needed
    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create repo: {e}")

    model_path = Path(model_path)

    # Upload based on model type
    if model_path.is_dir():
        # Upload entire directory
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            commit_message=commit_message,
            token=token,
        )
    else:
        # Upload single file
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_path.name,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token,
        )

    # Create and upload model card
    if model_card:
        card_content = generate_model_card(model_card)
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add model card",
            token=token,
        )

    return f"https://huggingface.co/{repo_id}"


def upload_lora_adapter(
    adapter_path: Path,
    repo_id: str,
    base_model: str,
    token: Optional[str] = None,
    private: bool = False,
    tags: List[str] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Upload a LoRA adapter to HuggingFace Hub.

    Args:
        adapter_path: Path to the adapter directory
        repo_id: HuggingFace repo ID
        base_model: Base model the adapter was trained on
        token: HuggingFace API token
        private: Whether to make the repo private
        tags: Optional tags for the model
        metrics: Optional metrics to include in model card

    Returns:
        URL of the uploaded adapter
    """
    if tags is None:
        tags = ["lora", "peft", "ocr"]

    model_card = {
        "base_model": base_model,
        "library_name": "peft",
        "tags": tags,
        "metrics": metrics or {},
        "model_type": "lora_adapter",
    }

    return upload_to_hub(
        model_path=adapter_path,
        repo_id=repo_id,
        model_type="lora",
        token=token,
        private=private,
        model_card=model_card,
    )


def upload_onnx_model(
    onnx_path: Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    model_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Upload an ONNX model to HuggingFace Hub.

    Args:
        onnx_path: Path to the ONNX file
        repo_id: HuggingFace repo ID
        token: HuggingFace API token
        private: Whether to make the repo private
        model_info: Optional model information for the card

    Returns:
        URL of the uploaded model
    """
    model_card = {
        "tags": ["onnx", "ocr"],
        "library_name": "onnx",
        "model_type": "onnx",
    }

    if model_info:
        model_card.update(model_info)

    return upload_to_hub(
        model_path=onnx_path,
        repo_id=repo_id,
        model_type="onnx",
        token=token,
        private=private,
        model_card=model_card,
    )


def generate_model_card(metadata: Dict[str, Any]) -> str:
    """
    Generate a model card (README.md) from metadata.

    Args:
        metadata: Dictionary with model information

    Returns:
        Model card content as string
    """
    # YAML frontmatter
    frontmatter = "---\n"

    if "tags" in metadata:
        frontmatter += "tags:\n"
        for tag in metadata["tags"]:
            frontmatter += f"  - {tag}\n"

    if "base_model" in metadata:
        frontmatter += f"base_model: {metadata['base_model']}\n"

    if "library_name" in metadata:
        frontmatter += f"library_name: {metadata['library_name']}\n"

    if "license" in metadata:
        frontmatter += f"license: {metadata.get('license', 'apache-2.0')}\n"

    if "language" in metadata:
        frontmatter += "language:\n"
        langs = metadata["language"]
        if isinstance(langs, list):
            for lang in langs:
                frontmatter += f"  - {lang}\n"
        else:
            frontmatter += f"  - {langs}\n"

    frontmatter += "---\n\n"

    # Main content
    content = frontmatter

    # Title
    content += f"# {metadata.get('title', 'OCR Model')}\n\n"

    # Description
    if "description" in metadata:
        content += f"{metadata['description']}\n\n"

    # Model type
    model_type = metadata.get("model_type", "unknown")
    content += "## Model Details\n\n"
    content += f"- **Model Type**: {model_type}\n"

    if "base_model" in metadata:
        content += f"- **Base Model**: {metadata['base_model']}\n"

    if "training_data" in metadata:
        content += f"- **Training Data**: {metadata['training_data']}\n"

    content += "\n"

    # Metrics
    if "metrics" in metadata and metadata["metrics"]:
        content += "## Evaluation Results\n\n"
        content += "| Metric | Value |\n"
        content += "|--------|-------|\n"
        for name, value in metadata["metrics"].items():
            if isinstance(value, float):
                content += f"| {name} | {value:.4f} |\n"
            else:
                content += f"| {name} | {value} |\n"
        content += "\n"

    # Usage
    content += "## Usage\n\n"

    if model_type == "lora_adapter":
        content += """```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel

# Load base model
processor = TrOCRProcessor.from_pretrained("{base_model}")
model = VisionEncoderDecoderModel.from_pretrained("{base_model}")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "{repo_id}")

# Inference
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
""".format(
            base_model=metadata.get("base_model", "microsoft/trocr-base-handwritten"),
            repo_id=metadata.get("repo_id", "your-username/model-name"),
        )
    elif model_type == "onnx":
        content += """```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image_array})
```
"""
    else:
        content += """```python
import torch
model = torch.load("model.pt")
model.eval()
output = model(input_tensor)
```
"""

    content += "\n"

    # License
    content += "## License\n\n"
    content += f"This model is released under the {metadata.get('license', 'Apache 2.0')} license.\n"

    return content


def update_registry(
    registry_path: Path,
    model_name: str,
    variant: str,
    repo_id: str,
    model_type: str = "huggingface",
) -> None:
    """
    Update the model registry with a new model.

    Args:
        registry_path: Path to registry.json
        model_name: Name of the model (e.g., "trocr")
        variant: Variant name (e.g., "finetuned")
        repo_id: HuggingFace repo ID or URL
        model_type: Type of model for registry
    """
    registry_path = Path(registry_path)

    # Load existing registry
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    # Add or update model
    if model_name not in registry:
        registry[model_name] = {}

    registry[model_name][variant] = {
        "url": repo_id,
        "type": model_type,
    }

    # Save updated registry
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Updated registry: {model_name}/{variant} -> {repo_id}")
