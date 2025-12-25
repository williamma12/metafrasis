"""
DB (Differentiable Binarization) Detector

Real-time scene text detection using differentiable binarization.
Fast and accurate for document and scene text detection.

Based on: "Real-time Scene Text Detection with Differentiable Binarization"
(AAAI 2020) https://arxiv.org/abs/1911.08947
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Optional, Tuple
from PIL import Image
from pathlib import Path

from ..base import TextRegion, BoundingBox, DEFAULT_CONFIDENCE
from .base import TextDetector


class ResNetBackbone(nn.Module):
    """ResNet-18 backbone for feature extraction"""

    def __init__(self, pretrained: bool = False):
        super(ResNetBackbone, self).__init__()

        # Initial conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks (simplified ResNet-18)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        """Create a ResNet layer with basic blocks"""
        layers = []

        # First block may downsample
        layers.append(BasicBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            Multi-scale features: [f2, f3, f4, f5]
        """
        # Stem
        x = self.conv1(x)      # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        f1 = x

        x = self.maxpool(x)    # [B, 64, H/4, W/4]

        # ResNet layers
        f2 = self.layer1(x)    # [B, 64, H/4, W/4]
        f3 = self.layer2(f2)   # [B, 128, H/8, W/8]
        f4 = self.layer3(f3)   # [B, 256, H/16, W/16]
        f5 = self.layer4(f4)   # [B, 512, H/32, W/32]

        return f2, f3, f4, f5


class BasicBlock(nn.Module):
    """Basic ResNet block"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super(FPN, self).__init__()

        # Lateral connections (reduce channels)
        self.lateral4 = nn.Conv2d(in_channels[3], out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)

        # Smooth layers (after upsampling)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        """
        Args:
            features: [f2, f3, f4, f5] multi-scale features

        Returns:
            Fused feature map
        """
        f2, f3, f4, f5 = features

        # Top-down pathway
        p5 = self.lateral4(f5)
        p4 = self.lateral3(f4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral2(f3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral1(f2) + F.interpolate(p3, scale_factor=2, mode='nearest')

        # Smooth
        p5 = self.smooth4(p5)
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth1(p2)

        # Upsample all to same size as p2
        p3 = F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=p2.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate
        fused = torch.cat([p2, p3, p4, p5], dim=1)

        return fused


class DBHead(nn.Module):
    """DB detection head - outputs probability and threshold maps"""

    def __init__(self, in_channels: int = 1024, k: int = 50):
        super(DBHead, self).__init__()

        self.k = k  # Amplification factor for differentiable binarization

        # Probability map head
        self.prob_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        # Threshold map head
        self.thresh_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Fused features [B, C, H, W]

        Returns:
            prob_map: Probability map [B, 1, H*4, W*4]
            thresh_map: Threshold map [B, 1, H*4, W*4]
            binary_map: Differentiable binary map [B, 1, H*4, W*4]
        """
        prob_map = self.prob_conv(x)
        thresh_map = self.thresh_conv(x)

        # Differentiable binarization
        binary_map = self._diff_binarize(prob_map, thresh_map)

        return prob_map, thresh_map, binary_map

    def _diff_binarize(self, prob_map, thresh_map):
        """Differentiable binarization using approximate step function"""
        return torch.reciprocal(1 + torch.exp(-self.k * (prob_map - thresh_map)))


class DBNet(nn.Module):
    """
    DB (Differentiable Binarization) Network

    Architecture:
    - ResNet backbone for feature extraction
    - FPN for multi-scale feature fusion
    - Dual heads for probability and threshold maps
    - Differentiable binarization for end-to-end training
    """

    def __init__(self, pretrained: bool = False, k: int = 50):
        super(DBNet, self).__init__()

        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)
        self.head = DBHead(in_channels=256*4, k=k)  # 4 scales concatenated

    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            prob_map: Probability map [B, 1, H, W]
            thresh_map: Threshold map [B, 1, H, W]
            binary_map: Binary map [B, 1, H, W]
        """
        # Extract features
        features = self.backbone(x)

        # Fuse features
        fused = self.fpn(features)

        # Detection heads
        prob_map, thresh_map, binary_map = self.head(fused)

        return prob_map, thresh_map, binary_map


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
