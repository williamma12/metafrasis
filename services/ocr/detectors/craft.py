"""
CRAFT (Character Region Awareness For Text detection) Detector

Based on the paper: "Character Region Awareness for Text Detection" (CVPR 2019)
Reference: https://github.com/clovaai/CRAFT-pytorch
"""
import torch
import numpy as np
import cv2
from typing import List, Optional
from PIL import Image
from pathlib import Path

from ..base import BoundingBox, TextRegion, DEFAULT_CONFIDENCE
from .base import TextDetector
from models import CRAFT


class CRAFTDetector(TextDetector):
    """
    CRAFT (Character Region Awareness For Text detection) detector

    Detects text at character level and links them into words using
    region and affinity maps.

    Args:
        model_path: Path to pretrained CRAFT weights (.pth file)
        device: Device to run model on ('auto', 'cuda', 'cpu')
        text_threshold: Threshold for region score (default: 0.7)
        link_threshold: Threshold for affinity score (default: 0.4)
        low_text: Low threshold for text detection (default: 0.4)
        canvas_size: Maximum image dimension for processing (default: 1280)
        mag_ratio: Magnification ratio for image resizing (default: 1.5)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
        **kwargs
    ):
        super().__init__(model_path=model_path, device=device)
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio

        self.model = None

    @property
    def name(self) -> str:
        return "craft"

    def load_model(self):
        """Load CRAFT model with pretrained weights"""
        print(f"Loading CRAFT detector...")
        print(f"Using device: {self.device}")

        # Create model
        self.model = CRAFT(pretrained=False)

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
        print("CRAFT detector loaded successfully")

    def detect(self, image: Image.Image) -> List[TextRegion]:
        """
        Detect text regions in an image

        Args:
            image: PIL Image

        Returns:
            List of TextRegion objects with bounding boxes and crops
        """
        if not self.is_loaded:
            self.load_model()

        # Convert PIL to numpy array
        img_np = np.array(image)
        if len(img_np.shape) == 2:  # Grayscale
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        # Resize image
        img_resized, target_ratio, size_heatmap = self._resize_image(img_np)
        ratio_h = ratio_w = 1 / target_ratio

        # Preprocess for model
        x = self._normalize_mean_variance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        x = x.to(self.device)

        # Forward pass
        with torch.no_grad():
            region_score, affinity_score = self.model(x)

        # Convert to numpy
        region_score = region_score[0, 0].cpu().numpy()
        affinity_score = affinity_score[0, 0].cpu().numpy()

        # Post-process to get bounding boxes
        boxes = self._get_bboxes(
            region_score,
            affinity_score,
            self.text_threshold,
            self.link_threshold,
            self.low_text
        )

        # Scale boxes back to original image size
        boxes = self._adjust_boxes(boxes, ratio_w, ratio_h)

        # Convert boxes to TextRegion objects
        text_regions = []
        original_image = np.array(image)

        for box in boxes:
            # Extract bounding box coordinates
            x_coords = box[:, 0]
            y_coords = box[:, 1]

            left = int(max(0, np.min(x_coords)))
            top = int(max(0, np.min(y_coords)))
            right = int(min(original_image.shape[1], np.max(x_coords)))
            bottom = int(min(original_image.shape[0], np.max(y_coords)))

            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0:
                continue

            # Create BoundingBox
            bbox = BoundingBox(left=left, top=top, width=width, height=height)

            # Crop image region
            crop = image.crop((left, top, right, bottom))

            # Create TextRegion with polygon
            polygon = [(int(x), int(y)) for x, y in box]

            text_region = TextRegion(
                bbox=bbox,
                crop=crop,
                confidence=DEFAULT_CONFIDENCE,  # CRAFT doesn't provide confidence
                polygon=polygon
            )

            text_regions.append(text_region)

        return text_regions

    def _resize_image(self, img):
        """Resize image for CRAFT processing"""
        img_resized, target_ratio = self._resize_aspect_ratio(
            img,
            self.canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio

        # Make dimensions divisible by 32 (for U-Net)
        target_h, target_w = img_resized.shape[:2]
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)

        img_resized = cv2.resize(img_resized, (target_w32, target_h32))
        size_heatmap = (target_w32 // 2, target_h32 // 2)

        return img_resized, target_ratio, size_heatmap

    def _resize_aspect_ratio(self, img, square_size, interpolation, mag_ratio=1):
        """Resize image while maintaining aspect ratio"""
        height, width, channel = img.shape

        # Magnify image
        target_size = mag_ratio * max(height, width)

        # Limit to max size
        if target_size > square_size:
            target_size = square_size

        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

        return img, ratio

    def _normalize_mean_variance(self, in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        """Normalize image using ImageNet statistics"""
        img = in_img.copy().astype(np.float32)
        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img

    def _get_bboxes(self, score_text, score_link, text_threshold, link_threshold, low_text):
        """
        Extract bounding boxes from score maps using connected components

        Args:
            score_text: Region score map
            score_link: Affinity score map
            text_threshold: Threshold for text regions
            link_threshold: Threshold for link regions
            low_text: Low threshold for text

        Returns:
            List of bounding boxes as numpy arrays of shape [4, 2]
        """
        # Threshold text and link maps
        text_score_comb = np.clip(score_text + score_link, 0, 1)

        # Binarize
        ret, text_score = cv2.threshold(text_score_comb, low_text, 1, cv2.THRESH_BINARY)

        # Connected components
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_score.astype(np.uint8),
            connectivity=4
        )

        boxes = []
        for k in range(1, nLabels):  # Skip background (label 0)
            # Size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:  # Min area threshold
                continue

            # Get segmentation map for this component
            segmap = np.zeros(text_score.shape, dtype=np.uint8)
            segmap[labels == k] = 255

            # Find contours
            contours, _ = cv2.findContours(segmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            # Get minimum area rectangle
            contour = contours[0]
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            # Add box if it passes threshold
            if np.max(score_text[segmap > 0]) > text_threshold:
                boxes.append(box)

        return boxes

    def _adjust_boxes(self, boxes, ratio_w, ratio_h):
        """Scale boxes back to original image coordinates"""
        if len(boxes) == 0:
            return boxes

        adjusted_boxes = []
        for box in boxes:
            box = np.array(box)
            box[:, 0] *= ratio_w
            box[:, 1] *= ratio_h
            adjusted_boxes.append(box)

        return adjusted_boxes
