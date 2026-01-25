"""
PP-OCR ONNX Recognizer - ONNX Runtime inference for PaddleOCR models

Uses pre-exported ONNX models from PaddleOCR for inference.
Useful when you have official PaddleOCR ONNX exports.

Advantages:
- Uses exact PaddleOCR model weights (no conversion needed)
- ONNX Runtime is well-optimized for inference
- Supports CUDA, TensorRT, OpenVINO, etc.

Disadvantages:
- Cannot fine-tune (inference only)
- Requires ONNX Runtime installation
"""
import numpy as np
from typing import List, Optional
from PIL import Image
from pathlib import Path

from ..base import Word, TextRegion
from .base import TextRecognizer
from ml.models import CTCDecoder, get_charset


class PPOCROnnxRecognizer(TextRecognizer):
    """
    PP-OCR ONNX Recognizer

    Loads and runs PaddleOCR ONNX models using ONNX Runtime.
    Supports GPU inference via CUDA or TensorRT execution providers.

    Args:
        model_path: Path to ONNX model file (.onnx)
        device: Device hint ('cuda', 'cpu', or None for auto)
        charset: Character set string or name ('greek', 'latin', etc.)
        img_height: Input image height (default: 48 for PP-OCRv4)
        img_width: Max input image width (default: 320)
        batch_size: Batch size for recognition
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = None,
        charset: str = "greek",
        img_height: int = 48,
        img_width: int = 320,
        batch_size: int = 16,
        **kwargs
    ):
        # Don't call parent __init__ with device (ONNX handles its own)
        self.model_path = model_path
        self.is_loaded = False
        self.config = kwargs

        # Handle charset name or string
        if charset in ("greek", "latin", "latin_lower"):
            self.charset = get_charset(charset)
        else:
            self.charset = charset

        self.num_classes = len(self.charset) + 1  # +1 for CTC blank

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.device_hint = device

        # CTC decoder (shared utility)
        self.decoder = CTCDecoder(self.charset)

        self.session = None

    @property
    def name(self) -> str:
        return "ppocr_onnx"

    def load_model(self):
        """Load ONNX model with ONNX Runtime"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Install with:\n"
                "  pip install onnxruntime        # CPU only\n"
                "  pip install onnxruntime-gpu    # With CUDA support"
            )

        if not self.model_path or not Path(self.model_path).exists():
            raise ValueError(
                f"ONNX model not found: {self.model_path}\n"
                "Download a PaddleOCR ONNX model or export one with paddle2onnx."
            )

        print(f"Loading PP-OCR ONNX model: {self.model_path}")

        # Select execution providers based on device hint
        if self.device_hint == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                }),
                'CPUExecutionProvider'
            ]
        elif self.device_hint == 'tensorrt':
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 2147483648,
                    'trt_fp16_enable': True,
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        else:
            # Auto-detect or CPU
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available and self.device_hint != 'cpu':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        # Create session with options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=providers
        )

        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        active_providers = self.session.get_providers()
        print(f"ONNX Runtime providers: {active_providers}")
        print(f"Input: {self.input_name} {self.input_shape}")
        print(f"Output: {self.output_name}")

        self.is_loaded = True
        print("PP-OCR ONNX recognizer loaded successfully")

    def recognize_regions(self, regions: List[TextRegion]) -> List[Word]:
        """
        Recognize text from multiple regions

        Args:
            regions: List of TextRegion objects with cropped images

        Returns:
            List of Word objects with recognized text
        """
        if not self.is_loaded:
            self.load_model()

        if not regions:
            return []

        words = []

        for i in range(0, len(regions), self.batch_size):
            batch_regions = regions[i:i + self.batch_size]

            # Preprocess images
            batch_images = [self._preprocess_image(r.crop) for r in batch_regions]
            batch_array = np.stack(batch_images).astype(np.float32)

            # Run ONNX inference
            outputs = self.session.run(None, {self.input_name: batch_array})
            logits = outputs[0]  # [B, T, num_classes]

            # Decode using shared decoder (convert to torch tensor)
            import torch
            logits_tensor = torch.from_numpy(logits)
            texts, confidences = self.decoder.decode_batch(logits_tensor)

            # Create Word objects
            for region, text, conf in zip(batch_regions, texts, confidences):
                word = Word(
                    text=text,
                    bbox=region.bbox,
                    confidence=conf if conf >= 0 else region.confidence
                )
                words.append(word)

        return words

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for PP-OCR ONNX model

        PaddleOCR preprocessing:
        1. Resize to fixed height, variable width
        2. Normalize to [-1, 1]
        3. Pad to max width
        """
        # Convert to RGB
        image = image.convert('RGB')
        img = np.array(image)

        # Resize to target height, maintain aspect ratio
        h, w = img.shape[:2]
        ratio = self.img_height / h
        new_w = min(int(w * ratio), self.img_width)

        img = Image.fromarray(img).resize((new_w, self.img_height), Image.BILINEAR)
        img = np.array(img)

        # Pad to target width (right padding)
        if new_w < self.img_width:
            pad_img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            pad_img[:, :new_w, :] = img
            img = pad_img

        # Normalize to [-1, 1] (PaddleOCR standard)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        # To CHW format [C, H, W]
        img = img.transpose(2, 0, 1)

        return img
