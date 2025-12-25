"""
CRAFT (Character Region Awareness For Text detection) Detector

Based on the paper: "Character Region Awareness for Text Detection" (CVPR 2019)
Reference: https://github.com/clovaai/CRAFT-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Optional
from PIL import Image
from pathlib import Path

from ..base import BoundingBox, TextRegion, DEFAULT_CONFIDENCE
from .base import TextDetector


class VGG16BN(nn.Module):
    """
    VGG16 with Batch Normalization backbone for feature extraction

    Used as the encoder part of CRAFT detector.
    """

    def __init__(self, pretrained=True):
        super(VGG16BN, self).__init__()

        # Conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        # Conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)

        # Conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)

        # Conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)

        # Conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass returning intermediate feature maps for skip connections

        Returns:
            Dictionary of feature maps at different scales
        """
        # Conv1
        h = self.relu(self.bn1_1(self.conv1_1(x)))
        h = self.relu(self.bn1_2(self.conv1_2(h)))
        h = self.pool(h)
        pool1 = h

        # Conv2
        h = self.relu(self.bn2_1(self.conv2_1(h)))
        h = self.relu(self.bn2_2(self.conv2_2(h)))
        h = self.pool(h)
        pool2 = h

        # Conv3
        h = self.relu(self.bn3_1(self.conv3_1(h)))
        h = self.relu(self.bn3_2(self.conv3_2(h)))
        h = self.relu(self.bn3_3(self.conv3_3(h)))
        h = self.pool(h)
        pool3 = h

        # Conv4
        h = self.relu(self.bn4_1(self.conv4_1(h)))
        h = self.relu(self.bn4_2(self.conv4_2(h)))
        h = self.relu(self.bn4_3(self.conv4_3(h)))
        h = self.pool(h)
        pool4 = h

        # Conv5
        h = self.relu(self.bn5_1(self.conv5_1(h)))
        h = self.relu(self.bn5_2(self.conv5_2(h)))
        h = self.relu(self.bn5_3(self.conv5_3(h)))
        h = self.pool(h)
        pool5 = h

        return {
            'pool1': pool1,  # 1/2
            'pool2': pool2,  # 1/4
            'pool3': pool3,  # 1/8
            'pool4': pool4,  # 1/16
            'pool5': pool5   # 1/32
        }


class CRAFT(nn.Module):
    """
    CRAFT: Character Region Awareness For Text detection

    Architecture: VGG16-BN backbone + U-Net decoder
    Output: Region score map + Affinity score map
    """

    def __init__(self, pretrained=False):
        super(CRAFT, self).__init__()

        # Backbone
        self.basenet = VGG16BN(pretrained=pretrained)

        # U-Net decoder with skip connections
        # Upsampling from 1/32 to 1/16
        self.upconv1 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Merge with pool4 (1/16) and upsample to 1/8
        self.upconv2 = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Merge with pool3 (1/8) and upsample to 1/4
        self.upconv3 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Merge with pool2 (1/4) and upsample to 1/2
        self.upconv4 = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Merge with pool1 (1/2) and upsample to 1/1
        self.upconv5 = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final convolution to produce 2 output channels
        # Channel 0: Region score
        # Channel 1: Affinity score
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1)  # 2 channels: region + affinity
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Tuple of (region_score, affinity_score) maps [B, 1, H, W] each
        """
        # Get feature maps from backbone
        features = self.basenet(x)

        # Decoder with skip connections
        # 1/32 -> 1/16
        h = self.upconv1(features['pool5'])
        h = F.interpolate(h, size=features['pool4'].size()[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, features['pool4']], dim=1)

        # 1/16 -> 1/8
        h = self.upconv2(h)
        h = F.interpolate(h, size=features['pool3'].size()[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, features['pool3']], dim=1)

        # 1/8 -> 1/4
        h = self.upconv3(h)
        h = F.interpolate(h, size=features['pool2'].size()[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, features['pool2']], dim=1)

        # 1/4 -> 1/2
        h = self.upconv4(h)
        h = F.interpolate(h, size=features['pool1'].size()[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, features['pool1']], dim=1)

        # 1/2 -> 1/1
        h = self.upconv5(h)
        h = F.interpolate(h, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # Final classification
        h = self.conv_cls(h)

        # Split into region and affinity scores
        region_score = h[:, 0:1, :, :]
        affinity_score = h[:, 1:2, :, :]

        # Apply sigmoid to get probabilities
        region_score = torch.sigmoid(region_score)
        affinity_score = torch.sigmoid(affinity_score)

        return region_score, affinity_score


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
