"""
Image preprocessing utilities and PDF conversion

PDF conversion is separate from OCR engines - engines only process PIL Images
"""
import numpy as np
from PIL import Image
from typing import List
import cv2


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images

    Args:
        pdf_path: Path to PDF file
        dpi: DPI for image conversion (higher = better quality, larger files)

    Returns:
        List of PIL Images (one per page)
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF support. "
            "Install it with: uv sync or pip install pdf2image"
        )

    return convert_from_path(pdf_path, dpi=dpi)


def ensure_pil_image(image) -> Image.Image:
    """
    Convert various image formats to PIL Image

    Args:
        image: numpy array, file path, or PIL Image

    Returns:
        PIL Image object
    """
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, str):
        return Image.open(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def resize_image(
    image: Image.Image,
    max_width: int = 2048,
    max_height: int = 2048
) -> Image.Image:
    """
    Resize image if it exceeds maximum dimensions (maintains aspect ratio)

    Args:
        image: PIL Image
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels

    Returns:
        Resized PIL Image
    """
    width, height = image.size

    if width <= max_width and height <= max_height:
        return image

    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def grayscale(image: Image.Image) -> Image.Image:
    """
    Convert image to grayscale

    Args:
        image: PIL Image

    Returns:
        Grayscale PIL Image
    """
    return image.convert('L')


def binarize(image: Image.Image, threshold: int = 128) -> Image.Image:
    """
    Convert image to binary (black and white)

    Args:
        image: PIL Image
        threshold: Threshold value (0-255)

    Returns:
        Binary PIL Image
    """
    gray = grayscale(image)
    return gray.point(lambda x: 0 if x < threshold else 255, '1')


def denoise(image: Image.Image, kernel_size: int = 3) -> Image.Image:
    """
    Apply denoising filter to image

    Args:
        image: PIL Image
        kernel_size: Size of denoising kernel (odd number)

    Returns:
        Denoised PIL Image
    """
    # Convert to numpy array
    img_array = np.array(image)

    # Apply median filter for denoising
    denoised = cv2.medianBlur(img_array, kernel_size)

    # Convert back to PIL
    return Image.fromarray(denoised)


def normalize_orientation(image: Image.Image) -> Image.Image:
    """
    Normalize image orientation based on EXIF data

    Args:
        image: PIL Image

    Returns:
        Oriented PIL Image
    """
    try:
        # Get EXIF orientation tag
        exif = image.getexif()
        orientation = exif.get(0x0112)  # Orientation tag

        if orientation:
            # Apply rotation based on orientation
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError):
        pass

    return image


def prepare_for_ocr(
    image: Image.Image,
    resize: bool = True,
    convert_grayscale: bool = False,
    apply_denoise: bool = False
) -> Image.Image:
    """
    Apply standard preprocessing pipeline for OCR

    Args:
        image: PIL Image
        resize: Resize if too large
        convert_grayscale: Convert to grayscale
        apply_denoise: Apply denoising filter

    Returns:
        Preprocessed PIL Image
    """
    # Ensure PIL image
    image = ensure_pil_image(image)

    # Normalize orientation
    image = normalize_orientation(image)

    # Resize if needed
    if resize:
        image = resize_image(image)

    # Convert to grayscale
    if convert_grayscale:
        image = grayscale(image)

    # Denoise
    if apply_denoise:
        image = denoise(image)

    return image
