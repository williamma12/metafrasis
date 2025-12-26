"""
DB (Differentiable Binarization) Detector

Real-time scene text detection using differentiable binarization.
Fast and accurate for document and scene text detection.

Based on: "Real-time Scene Text Detection with Differentiable Binarization"
(AAAI 2020) https://arxiv.org/abs/1911.08947
"""
import torch
import numpy as np
import cv2
from typing import List, Optional, Tuple
from PIL import Image
from pathlib import Path

from ..base import TextRegion, BoundingBox, DEFAULT_CONFIDENCE
from .base import TextDetector
from models import DBNet


class DBDetector(TextDetector):
    """
    DB (Differentiable Binarization) text detector

    Fast and accurate scene text detection using differentiable binarization.
    Works well for both document and scene text.

    Args:
        model_path: Path to pretrained DB weights (.pth file)
        device: Device to run model on ('auto', 'cuda', 'cpu')
        thresh: Binary threshold (default: 0.3)
        box_thresh: Box score threshold (default: 0.7)
        max_candidates: Maximum number of text boxes to detect (default: 1000)
        unclip_ratio: Unclip ratio for polygon expansion (default: 1.5)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        thresh: float = 0.3,
        box_thresh: float = 0.7,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        **kwargs
    ):
        super().__init__(model_path=model_path, device=device)

        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

        self.model = None

    @property
    def name(self) -> str:
        return "db"

    def load_model(self):
        """Load DB model with pretrained weights"""
        print(f"Loading DB detector...")
        print(f"Using device: {self.device}")

        # Create model
        self.model = DBNet(pretrained=False, k=50)

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

            self.model.load_state_dict(new_state_dict, strict=False)
            print("Pretrained weights loaded successfully")
        else:
            print("Warning: No pretrained weights provided. Using random initialization.")

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
        print("DB detector loaded successfully")

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

        # Preprocess
        img_tensor, scale_w, scale_h = self._preprocess_image(image)

        # Forward pass
        with torch.no_grad():
            prob_map, thresh_map, binary_map = self.model(img_tensor)

        # Post-process: extract boxes from binary map
        binary_np = binary_map[0, 0].cpu().numpy()  # [H, W]
        boxes = self._get_boxes(binary_np, scale_w, scale_h)

        # Convert to TextRegions
        regions = []
        for box in boxes:
            x_min = int(min(box[:, 0]))
            y_min = int(min(box[:, 1]))
            x_max = int(max(box[:, 0]))
            y_max = int(max(box[:, 1]))

            # Clip to image bounds
            width, height = image.size
            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            x_max = max(0, min(x_max, width))
            y_max = max(0, min(y_max, height))

            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min:
                continue

            # Create bounding box
            bbox = BoundingBox(
                left=x_min,
                top=y_min,
                width=x_max - x_min,
                height=y_max - y_min
            )

            # Crop region
            crop = image.crop((x_min, y_min, x_max, y_max))

            # Create TextRegion
            region = TextRegion(
                bbox=bbox,
                crop=crop,
                confidence=DEFAULT_CONFIDENCE,  # DB doesn't provide per-box confidence
                polygon=[(int(p[0]), int(p[1])) for p in box]
            )
            regions.append(region)

        return regions

    def _preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, float, float]:
        """
        Preprocess image for DB

        Args:
            image: PIL Image

        Returns:
            img_tensor: Preprocessed tensor [1, 3, H, W]
            scale_w: Width scaling factor
            scale_h: Height scaling factor
        """
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy
        img = np.array(image)

        # Get original size
        orig_h, orig_w = img.shape[:2]

        # Resize to multiple of 32 (required by architecture)
        target_h = (orig_h // 32) * 32
        target_w = (orig_w // 32) * 32

        # Ensure minimum size
        target_h = max(32, target_h)
        target_w = max(32, target_w)

        # Resize
        img = cv2.resize(img, (target_w, target_h))

        # Calculate scale factors
        scale_w = orig_w / target_w
        scale_h = orig_h / target_h

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # Convert to tensor [3, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # Add batch dimension [1, 3, H, W]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor, scale_w, scale_h

    def _get_boxes(self, binary_map: np.ndarray, scale_w: float, scale_h: float) -> List[np.ndarray]:
        """
        Extract bounding boxes from binary map

        Args:
            binary_map: Binary segmentation map [H, W]
            scale_w: Width scaling factor
            scale_h: Height scaling factor

        Returns:
            List of boxes, each box is [4, 2] array of corners
        """
        # Threshold
        bitmap = (binary_map > self.thresh).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours[:self.max_candidates]:
            # Get minimum area rectangle
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get bounding box
            if len(approx) < 4:
                continue

            # Use minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            # Calculate box score
            box_score = self._box_score(binary_map, contour)

            if box_score < self.box_thresh:
                continue

            # Unclip (expand) the box
            box = self._unclip(box)

            # Scale back to original image size
            box[:, 0] = box[:, 0] * scale_w
            box[:, 1] = box[:, 1] * scale_h

            boxes.append(box)

        return boxes

    def _box_score(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
        """Calculate average score inside the box"""
        mask = np.zeros_like(bitmap, dtype=np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 1)

        score = cv2.mean(bitmap, mask)[0]
        return score

    def _unclip(self, box: np.ndarray) -> np.ndarray:
        """Expand the box by unclip_ratio"""
        import pyclipper

        poly = pyclipper.PyclipperOffset()
        poly.AddPath(box.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        # Calculate distance
        area = cv2.contourArea(box)
        perimeter = cv2.arcLength(box, True)
        distance = area * self.unclip_ratio / (perimeter + 1e-5)

        # Expand
        expanded = poly.Execute(distance)

        if len(expanded) == 0:
            return box

        expanded = np.array(expanded[0]).reshape(-1, 2)

        # Get minimum area rectangle
        rect = cv2.minAreaRect(expanded)
        expanded_box = cv2.boxPoints(rect)

        return expanded_box
