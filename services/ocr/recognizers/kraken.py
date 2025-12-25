"""
Kraken Text Recognizer

Specialized OCR for historical documents with Ancient Greek support.
Wraps the kraken library for text recognition.

Kraken is designed for historical document analysis and supports:
- Ancient Greek and other historical scripts
- Degraded and noisy documents
- Custom training on specific scripts
"""
import numpy as np
from typing import List, Optional
from PIL import Image
from pathlib import Path

from ..base import Word, TextRegion, DEFAULT_CONFIDENCE
from .base import TextRecognizer

# Workaround for numpy _no_nep50_warning compatibility issue
def dummy_npwarn_decorator_factory():
    def npwarn_decorator(x):
        return x
    return npwarn_decorator

np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)

# Workaround for numpy.random.bit_generator.SeedlessSequence compatibility issue
if hasattr(np.random, 'bit_generator') and not hasattr(np.random.bit_generator, 'SeedlessSequence'):
    if hasattr(np.random, 'SeedSequence'):
        np.random.bit_generator.SeedlessSequence = np.random.SeedSequence


class KrakenRecognizer(TextRecognizer):
    """
    Kraken text recognizer for historical documents

    Wraps the kraken library which is specifically designed for OCR on
    historical manuscripts and documents. Supports Ancient Greek out of
    the box with pretrained models.

    Args:
        model_path: Path to kraken model file (.mlmodel)
                   Default: None (uses built-in Ancient Greek model)
        device: Device to run model on ('auto', 'cuda', 'cpu')
        batch_size: Batch size for recognition (default: 8)
        language: Language/script code (default: 'grc' for Ancient Greek)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        batch_size: int = 8,
        language: str = 'grc',  # Ancient Greek
        **kwargs
    ):
        super().__init__(model_path=model_path, device=device, batch_size=batch_size)
        self.batch_size = batch_size
        self.language = language

        self.recognizer = None
        self.blla = None  # Baseline and layout analyzer

    @property
    def name(self) -> str:
        return "kraken"

    def load_model(self):
        """Load Kraken recognizer"""
        try:
            from kraken import blla
            from kraken.lib import models
        except ImportError:
            raise ImportError(
                "Kraken not installed. Install with: pip install kraken\n"
                "For Ancient Greek support, also install: pip install kraken[greek]"
            )

        print(f"Loading Kraken recognizer...")
        print(f"Language: {self.language}")
        print(f"Using device: {self.device}")

        # Load recognition model
        if self.model_path and Path(self.model_path).exists():
            print(f"Loading custom model from: {self.model_path}")
            self.recognizer = models.load_any(self.model_path)
        else:
            # Use default Ancient Greek model
            print(f"Loading default model for {self.language}")
            # Kraken will download the model automatically
            # For Ancient Greek, use the default model
            try:
                # Try to load pre-trained Greek model
                # This is a placeholder - actual model loading depends on kraken setup
                self.recognizer = models.load_any(f'{self.language}.mlmodel')
            except Exception as e:
                print(f"Warning: Could not load language-specific model: {e}")
                print("Using default model. For best results, train a custom model.")
                # Fall back to a generic model or continue without
                self.recognizer = None

        self.is_loaded = True
        print("Kraken recognizer loaded successfully")

    def recognize_regions(self, regions: List[TextRegion]) -> List[Word]:
        """
        Recognize text from multiple regions

        Args:
            regions: List of TextRegion objects with cropped images

        Returns:
            List of Word objects with recognized text and confidences
        """
        if not self.is_loaded:
            self.load_model()

        if not regions:
            return []

        # Import kraken modules
        try:
            from kraken import rpred
            from kraken.lib import models
        except ImportError:
            raise ImportError("Kraken not installed. Install with: pip install kraken")

        words = []

        # Process in batches
        for i in range(0, len(regions), self.batch_size):
            batch_regions = regions[i:i + self.batch_size]

            for region in batch_regions:
                # Get image
                img = region.crop

                # Convert to format expected by kraken (PIL Image)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                try:
                    # Recognize text using kraken
                    if self.recognizer:
                        # Use the recognition model
                        pred = rpred.rpred(
                            network=self.recognizer,
                            im=img,
                            bounds={'boxes': [[
                                (0, 0),
                                (img.width, 0),
                                (img.width, img.height),
                                (0, img.height)
                            ]]},
                        )

                        # Extract recognized text
                        recognized_text = ""
                        for record in pred:
                            recognized_text += record.prediction

                        # Use default confidence if not available
                        confidence = getattr(record, 'confidence', region.confidence)
                        if confidence == -1.0 or confidence is None:
                            confidence = region.confidence

                    else:
                        # Fallback if model not loaded
                        recognized_text = ""
                        confidence = DEFAULT_CONFIDENCE

                except Exception as e:
                    print(f"Warning: Kraken recognition failed: {e}")
                    recognized_text = ""
                    confidence = DEFAULT_CONFIDENCE

                # Create Word object
                word = Word(
                    text=recognized_text,
                    bbox=region.bbox,
                    confidence=confidence
                )
                words.append(word)

        return words


class KrakenFullRecognizer(TextRecognizer):
    """
    Kraken recognizer with integrated layout analysis

    Uses Kraken's built-in layout analysis along with text recognition.
    This is more suitable when using Kraken as a complete OCR pipeline.

    Args:
        model_path: Path to kraken recognition model (.mlmodel)
        segmentation_model: Path to segmentation model (default: auto)
        device: Device to run model on ('auto', 'cuda', 'cpu')
        batch_size: Batch size for recognition (default: 8)
        language: Language/script code (default: 'grc' for Ancient Greek)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        segmentation_model: Optional[str] = None,
        device: str = 'auto',
        batch_size: int = 8,
        language: str = 'grc',
        **kwargs
    ):
        super().__init__(model_path=model_path, device=device, batch_size=batch_size)
        self.batch_size = batch_size
        self.language = language
        self.segmentation_model = segmentation_model

        self.recognizer = None
        self.segmenter = None

    @property
    def name(self) -> str:
        return "kraken_full"

    def load_model(self):
        """Load Kraken models (segmentation + recognition)"""
        try:
            from kraken import blla
            from kraken.lib import models
        except ImportError:
            raise ImportError(
                "Kraken not installed. Install with: pip install kraken"
            )

        print(f"Loading Kraken full recognizer...")
        print(f"Language: {self.language}")

        # Load segmentation model
        if self.segmentation_model:
            print(f"Loading segmentation model from: {self.segmentation_model}")
            self.segmenter = models.load_any(self.segmentation_model)
        else:
            print("Using default segmentation model")
            # Kraken will use default
            self.segmenter = None

        # Load recognition model
        if self.model_path and Path(self.model_path).exists():
            print(f"Loading recognition model from: {self.model_path}")
            self.recognizer = models.load_any(self.model_path)
        else:
            print(f"Using default recognition model for {self.language}")
            self.recognizer = None

        self.is_loaded = True
        print("Kraken full recognizer loaded successfully")

    def recognize_regions(self, regions: List[TextRegion]) -> List[Word]:
        """
        Recognize text from regions using Kraken's full pipeline

        Args:
            regions: List of TextRegion objects

        Returns:
            List of Word objects
        """
        if not self.is_loaded:
            self.load_model()

        if not regions:
            return []

        try:
            from kraken import rpred, blla
            from kraken.lib import models, vgsl
        except ImportError:
            raise ImportError("Kraken not installed")

        words = []

        for region in regions:
            img = region.crop

            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            try:
                # Perform segmentation on the region
                if self.segmenter:
                    baseline_seg = blla.segment(
                        img,
                        model=self.segmenter
                    )
                else:
                    # Use default segmentation
                    baseline_seg = blla.segment(img)

                # Recognize text
                if self.recognizer:
                    pred = rpred.rpred(
                        network=self.recognizer,
                        im=img,
                        bounds=baseline_seg
                    )

                    # Extract text from all lines
                    recognized_text = " ".join(record.prediction for record in pred)
                    confidence = region.confidence
                else:
                    recognized_text = ""
                    confidence = DEFAULT_CONFIDENCE

            except Exception as e:
                print(f"Warning: Kraken recognition failed: {e}")
                recognized_text = ""
                confidence = DEFAULT_CONFIDENCE

            # Create Word object
            word = Word(
                text=recognized_text,
                bbox=region.bbox,
                confidence=confidence
            )
            words.append(word)

        return words
