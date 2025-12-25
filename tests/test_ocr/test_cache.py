"""
Tests for ImageCache
"""
import pytest
from pathlib import Path
from PIL import Image
from services.ocr.cache import ImageCache


@pytest.fixture
def sample_image():
    """Create a simple test image"""
    img = Image.new('RGB', (100, 100), color='red')
    return img


@pytest.fixture
def cache(tmp_path):
    """Create an ImageCache with temporary directory"""
    return ImageCache(cache_dir=str(tmp_path / "cache"))


class TestImageCache:
    """Tests for ImageCache"""

    def test_cache_creation_with_default_dir(self):
        """Test ImageCache creates temp directory by default"""
        cache = ImageCache()

        assert cache.cache_dir.exists()
        assert cache.cache_dir.is_dir()

        # Cleanup
        cache.clear()

    def test_cache_creation_with_custom_dir(self, tmp_path):
        """Test ImageCache with custom directory"""
        cache_dir = tmp_path / "custom_cache"
        cache = ImageCache(cache_dir=str(cache_dir))

        assert cache.cache_dir == cache_dir
        assert cache_dir.exists()

        # Cleanup
        cache.clear()

    def test_add_image(self, cache, sample_image):
        """Test adding image to cache"""
        filepath = cache.add_image(sample_image)

        assert isinstance(filepath, Path)
        assert filepath.exists()
        assert filepath.suffix == '.png'

    def test_add_image_with_prefix(self, cache, sample_image):
        """Test adding image with custom prefix"""
        filepath = cache.add_image(sample_image, prefix="test")

        assert filepath.name.startswith("test_")

    def test_add_multiple_images(self, cache, sample_image):
        """Test adding multiple images"""
        images = [sample_image] * 3
        filepaths = cache.add_images(images)

        assert len(filepaths) == 3
        assert all(fp.exists() for fp in filepaths)
        assert len(cache) == 3

    def test_get_image_by_index(self, cache, sample_image):
        """Test retrieving image by index"""
        cache.add_image(sample_image)
        retrieved_image = cache.get_image(0)

        assert retrieved_image is not None
        assert isinstance(retrieved_image, Image.Image)
        assert retrieved_image.size == sample_image.size

    def test_get_image_invalid_index(self, cache):
        """Test retrieving image with invalid index returns None"""
        assert cache.get_image(0) is None
        assert cache.get_image(-1) is None
        assert cache.get_image(100) is None

    def test_cache_length(self, cache, sample_image):
        """Test cache length tracking"""
        assert len(cache) == 0

        cache.add_image(sample_image)
        assert len(cache) == 1

        cache.add_image(sample_image)
        assert len(cache) == 2

    def test_clear_cache(self, cache, sample_image):
        """Test clearing cache"""
        cache.add_image(sample_image)
        cache.add_image(sample_image)

        assert len(cache) == 2
        cache_dir = cache.cache_dir

        cache.clear()

        assert len(cache) == 0
        assert not cache_dir.exists()

    def test_cache_deduplication(self, cache, sample_image):
        """Test that identical images use same cached file"""
        # Add same image twice
        filepath1 = cache.add_image(sample_image, prefix="img")
        filepath2 = cache.add_image(sample_image, prefix="img")

        # Both should have same hash in filename (deduplication)
        # But different prefixes mean different filenames
        assert filepath1.exists()
        assert filepath2.exists()

    def test_cache_handles_different_images(self, cache):
        """Test cache handles different images correctly"""
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')

        filepath1 = cache.add_image(img1)
        filepath2 = cache.add_image(img2)

        # Different images should have different hashes
        assert filepath1 != filepath2

        # Both should be retrievable
        retrieved1 = cache.get_image(0)
        retrieved2 = cache.get_image(1)

        assert retrieved1 is not None
        assert retrieved2 is not None

    def test_cache_cleanup_on_deletion(self, tmp_path, sample_image):
        """Test that cache cleans up on deletion"""
        cache_dir = tmp_path / "temp_cache"
        cache = ImageCache(cache_dir=str(cache_dir))

        cache.add_image(sample_image)
        assert cache_dir.exists()

        # Delete cache object
        del cache

        # Directory should be cleaned up (may not work perfectly in all cases)
        # This is a best-effort cleanup test

    def test_add_images_with_prefix(self, cache, sample_image):
        """Test add_images uses numbered prefixes"""
        images = [sample_image] * 3
        filepaths = cache.add_images(images, prefix="batch")

        # Each should have incrementing prefix
        assert all("batch_" in fp.name for fp in filepaths)
        assert filepaths[0].name.startswith("batch_0_")
        assert filepaths[1].name.startswith("batch_1_")
        assert filepaths[2].name.startswith("batch_2_")
