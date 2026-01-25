"""
PP-OCR Recognizer - PyTorch implementation of PaddleOCR's recognition model

Architecture based on PP-OCRv3/v4:
- Backbone: MobileNetV3-like feature extractor
- Neck: BiLSTM for sequence modeling
- Head: CTC decoder

Reference: https://github.com/PaddlePaddle/PaddleOCR
"""
import torch
import numpy as np
from typing import List, Optional
from PIL import Image
from pathlib import Path

from ..base import Word, TextRegion
from .base import TextRecognizer
from ml.models import PPOCRModel, CTCDecoder, get_charset


class PPOCRRecognizer(TextRecognizer):
    """
    PP-OCR Recognizer - PyTorch implementation

    Uses MobileNetV3 + BiLSTM + CTC architecture from PaddleOCR.
    Shares BiLSTM and CTC components with CRNN.

    Args:
        model_path: Path to PyTorch weights (.pth file)
        device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
        charset: Character set string or name ('greek', 'latin', 'latin_lower')
        img_height: Input image height (default: 32)
        img_width: Max input image width (default: 320)
        batch_size: Batch size for recognition
        backbone_scale: MobileNet width multiplier (default: 0.5)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = None,
        charset: str = "greek",
        img_height: int = 32,
        img_width: int = 320,
        batch_size: int = 16,
        backbone_scale: float = 0.5,
        **kwargs
    ):
        super().__init__(model_path=model_path, device=device, **kwargs)

        # Handle charset name or string
        if charset in ("greek", "latin", "latin_lower"):
            self.charset = get_charset(charset)
        else:
            self.charset = charset

        self.num_classes = len(self.charset) + 1  # +1 for CTC blank

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.backbone_scale = backbone_scale

        # CTC decoder (shared utility)
        self.decoder = CTCDecoder(self.charset)

        self.model = None

    @property
    def name(self) -> str:
        return "ppocr"

    def load_model(self):
        """Load PP-OCR PyTorch model"""
        print(f"Loading PP-OCR recognizer...")
        print(f"Device: {self.device}")
        print(f"Charset size: {len(self.charset)} + 1 blank = {self.num_classes}")

        self.model = PPOCRModel(
            in_channels=3,
            num_classes=self.num_classes,
            backbone_scale=self.backbone_scale,
            hidden_size=256
        )

        if self.model_path and Path(self.model_path).exists():
            print(f"Loading weights from: {self.model_path}")
            state_dict = torch.load(self.model_path, map_location='cpu')

            # Handle different formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # Remove 'module.' prefix if present
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            print("Weights loaded")
        else:
            print("Warning: No pretrained weights - using random initialization")

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        print("PP-OCR recognizer loaded successfully")

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
            batch_tensor = torch.stack(batch_images).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(batch_tensor)

            # Decode
            texts, confidences = self.decoder.decode_batch(logits.cpu())

            # Create Word objects
            for region, text, conf in zip(batch_regions, texts, confidences):
                word = Word(
                    text=text,
                    bbox=region.bbox,
                    confidence=conf if conf >= 0 else region.confidence
                )
                words.append(word)

        return words

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for PP-OCR"""
        # Convert to RGB
        image = image.convert('RGB')
        img = np.array(image)

        # Resize to target height, maintain aspect ratio
        h, w = img.shape[:2]
        ratio = self.img_height / h
        new_w = min(int(w * ratio), self.img_width)

        img = Image.fromarray(img).resize((new_w, self.img_height), Image.BILINEAR)
        img = np.array(img)

        # Pad to target width
        if new_w < self.img_width:
            pad_img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            pad_img[:, :new_w, :] = img
            img = pad_img

        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        # To tensor [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img
