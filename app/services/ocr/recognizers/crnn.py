"""
CRNN (Convolutional Recurrent Neural Network) Recognizer

CNN for feature extraction + BiLSTM for sequence modeling + CTC decoding
Fast and accurate for printed text recognition.

Based on: "An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition" (TPAMI 2016)
"""
import torch
import numpy as np
from typing import List, Optional
from PIL import Image
from pathlib import Path

from ..base import Word, TextRegion
from .base import TextRecognizer
from ml.models import CRNN, CTCDecoder, get_charset


class CRNNRecognizer(TextRecognizer):
    """
    CRNN (Convolutional Recurrent Neural Network) recognizer

    CNN for feature extraction + BiLSTM for sequence modeling + CTC decoding.
    Fast and accurate for printed text recognition.

    Args:
        model_path: Path to pretrained CRNN weights (.pth file)
        device: Device to run model on ('auto', 'cuda', 'cpu')
        batch_size: Batch size for recognition (default: 16)
        charset: Character set string (default: digits + lowercase letters)
        img_height: Height to resize images to (default: 32)
        img_width: Maximum width to pad images to (default: 100)
        num_channels: Number of input channels (1 for grayscale, 3 for RGB)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        batch_size: int = 16,
        charset: str = "0123456789abcdefghijklmnopqrstuvwxyz",
        img_height: int = 32,
        img_width: int = 100,
        num_channels: int = 1,
        **kwargs
    ):
        super().__init__(model_path=model_path, device=device, batch_size=batch_size)
        self.batch_size = batch_size

        # Add blank token for CTC
        self.charset = charset
        self.num_classes = len(charset) + 1  # +1 for blank
        self.blank_token = len(charset)  # Blank is last index

        # Character mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(charset)}

        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels

        # CTC decoder (shared utility)
        self.decoder = CTCDecoder(charset)

        self.model = None

    @property
    def name(self) -> str:
        return "crnn"

    def load_model(self):
        """Load CRNN model with pretrained weights"""
        print(f"Loading CRNN recognizer...")
        print(f"Using device: {self.device}")
        print(f"Charset: {self.charset}")
        print(f"Num classes: {self.num_classes} (including blank)")

        # Create model
        self.model = CRNN(
            img_height=self.img_height,
            num_channels=self.num_channels,
            num_classes=self.num_classes,
            hidden_size=256
        )

        # Load pretrained weights if provided
        if self.model_path and Path(self.model_path).exists():
            print(f"Loading pretrained weights from: {self.model_path}")
            state_dict = torch.load(self.model_path, map_location='cpu')

            # Handle different state dict formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Remove 'module.' prefix if present (from DataParallel)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict)
            print("Pretrained weights loaded successfully")
        else:
            print("Warning: No pretrained weights provided. Using random initialization.")

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
        print("CRNN recognizer loaded successfully")

    def recognize_regions(self, regions: List[TextRegion]) -> List[Word]:
        """
        Recognize text from multiple regions (batch processing)

        Args:
            regions: List of TextRegion objects with cropped images

        Returns:
            List of Word objects with recognized text and confidences
        """
        if not self.is_loaded:
            self.load_model()

        if not regions:
            return []

        words = []

        # Process in batches
        for i in range(0, len(regions), self.batch_size):
            batch_regions = regions[i:i + self.batch_size]

            # Extract and preprocess images
            batch_images = []
            for region in batch_regions:
                img = self._preprocess_image(region.crop)
                batch_images.append(img)

            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)

            # Run inference
            with torch.no_grad():
                preds = self.model(batch_tensor)  # [T, B, num_classes]

            # Decode predictions
            texts = self._decode_batch(preds)

            # Create Word objects
            for region, text in zip(batch_regions, texts):
                word = Word(
                    text=text,
                    bbox=region.bbox,
                    confidence=region.confidence  # Use detection confidence
                )
                words.append(word)

        return words

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for CRNN

        Args:
            image: PIL Image

        Returns:
            Preprocessed tensor [C, H, W]
        """
        # Convert to grayscale or RGB based on num_channels
        if self.num_channels == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        # Convert to numpy
        img = np.array(image)

        # Resize to target height while maintaining aspect ratio
        h, w = img.shape[:2]
        ratio = self.img_height / h
        new_w = int(w * ratio)

        # Resize
        if self.num_channels == 1:
            img = Image.fromarray(img).resize((new_w, self.img_height), Image.BILINEAR)
            img = np.array(img)
        else:
            img = Image.fromarray(img).resize((new_w, self.img_height), Image.BILINEAR)
            img = np.array(img)

        # Pad or crop to target width
        if new_w < self.img_width:
            # Pad right
            if self.num_channels == 1:
                pad_img = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
                pad_img[:, :new_w] = img
            else:
                pad_img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
                pad_img[:, :new_w, :] = img
            img = pad_img
        elif new_w > self.img_width:
            # Crop
            img = img[:, :self.img_width]

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor
        if self.num_channels == 1:
            img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        # Normalize (mean=0.5, std=0.5)
        img = (img - 0.5) / 0.5

        return img

    def _decode_batch(self, preds: torch.Tensor) -> List[str]:
        """
        Decode CTC predictions to text strings

        Args:
            preds: CTC output tensor [T, B, num_classes]

        Returns:
            List of decoded text strings
        """
        # Convert from [T, B, C] to [B, T, C] for CTCDecoder
        preds = preds.permute(1, 0, 2)

        # Use shared CTC decoder
        texts, _ = self.decoder.decode_batch(preds)
        return texts

    def _ctc_decode(self, indices: np.ndarray) -> str:
        """
        Decode a single CTC sequence (greedy decoding).

        Args:
            indices: Array of class indices [T]

        Returns:
            Decoded text string
        """
        if len(indices) == 0:
            return ""

        # Remove blanks and merge repeated characters
        chars = []
        prev_idx = None

        for idx in indices:
            if idx != self.blank_token:
                if idx != prev_idx:
                    if idx in self.idx_to_char:
                        chars.append(self.idx_to_char[idx])
            prev_idx = idx

        return "".join(chars)
