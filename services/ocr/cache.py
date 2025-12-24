"""
Image caching utility for storing processed images temporarily
"""
import os
import tempfile
from pathlib import Path
from typing import List, Optional
from PIL import Image
import hashlib
import shutil


class ImageCache:
    """
    Temporary disk cache for images during OCR processing session
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize image cache

        Args:
            cache_dir: Directory for cached images. If None, creates temp directory
        """
        if cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="metafrasis_cache_"))
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._image_paths: List[Path] = []

    def add_image(self, image: Image.Image, prefix: str = "image") -> Path:
        """
        Save image to cache and return path

        Args:
            image: PIL Image to cache
            prefix: Prefix for cached filename

        Returns:
            Path to cached image file
        """
        # Generate unique filename based on image hash
        img_hash = self._hash_image(image)
        filename = f"{prefix}_{img_hash}.png"
        filepath = self.cache_dir / filename

        # Save if not already cached
        if not filepath.exists():
            image.save(filepath, format="PNG")

        self._image_paths.append(filepath)
        return filepath

    def add_images(self, images: List[Image.Image], prefix: str = "image") -> List[Path]:
        """
        Save multiple images to cache

        Args:
            images: List of PIL Images to cache
            prefix: Prefix for cached filenames

        Returns:
            List of paths to cached image files
        """
        return [self.add_image(img, f"{prefix}_{i}") for i, img in enumerate(images)]

    def get_image(self, index: int) -> Optional[Image.Image]:
        """
        Load cached image by index

        Args:
            index: Index of image in cache

        Returns:
            PIL Image or None if index out of range
        """
        if 0 <= index < len(self._image_paths):
            return Image.open(self._image_paths[index])
        return None

    def clear(self):
        """Remove all cached images and clean up cache directory"""
        try:
            shutil.rmtree(self.cache_dir)
            self._image_paths = []
        except Exception as e:
            print(f"Warning: Failed to clean cache directory: {e}")

    def __len__(self) -> int:
        """Return number of cached images"""
        return len(self._image_paths)

    def __del__(self):
        """Cleanup cache on deletion"""
        self.clear()

    @staticmethod
    def _hash_image(image: Image.Image) -> str:
        """
        Generate hash of image for unique filename

        Args:
            image: PIL Image

        Returns:
            MD5 hash string (first 8 chars)
        """
        # Convert image to bytes
        from io import BytesIO
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Generate hash
        return hashlib.md5(img_bytes).hexdigest()[:8]
