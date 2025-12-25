#!/usr/bin/env python3
"""
Model download utility for OCR engines

Downloads pretrained weights from various sources:
- Direct URLs (HTTP/HTTPS)
- Google Drive
- Hugging Face Hub (handled by transformers library)

Usage:
    python models/download_models.py --all
    python models/download_models.py --craft base
    python models/download_models.py --crnn base
    python models/download_models.py --tesseract base
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional
import urllib.request
import urllib.error


# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_FILE = MODELS_DIR / "registry.json"


def load_registry() -> Dict:
    """Load model registry from JSON file"""
    with open(REGISTRY_FILE) as f:
        return json.load(f)


def download_file(url: str, output_path: Path, show_progress: bool = True):
    """
    Download a file from a URL with progress indication

    Args:
        url: URL to download from
        output_path: Path to save the file
        show_progress: Whether to show download progress
    """
    def progress_hook(count, block_size, total_size):
        if not show_progress or total_size <= 0:
            return

        percent = int(count * block_size * 100 / total_size)
        percent = min(percent, 100)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '=' * filled + '-' * (bar_length - filled)

        mb_downloaded = count * block_size / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)

        sys.stdout.write(f'\r[{bar}] {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()

    try:
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        if show_progress:
            print()  # New line after progress bar
        print(f"✓ Downloaded to: {output_path}")
    except urllib.error.HTTPError as e:
        print(f"✗ HTTP Error {e.code}: {e.reason}")
        raise
    except urllib.error.URLError as e:
        print(f"✗ URL Error: {e.reason}")
        raise
    except Exception as e:
        print(f"✗ Error downloading file: {e}")
        raise


def download_from_gdrive(file_id: str, output_path: Path):
    """
    Download a file from Google Drive

    Args:
        file_id: Google Drive file ID
        output_path: Path to save the file
    """
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"

    # For large files, we may need to handle the virus scan warning
    # Try direct download first
    try:
        download_file(url, output_path)
    except Exception as e:
        print(f"✗ Direct download failed: {e}")
        print("Note: For large files, you may need to download manually from Google Drive")
        print(f"URL: https://drive.google.com/file/d/{file_id}/view")
        raise


def download_from_direct_url(url: str, output_path: Path):
    """
    Download a file from a direct URL

    Args:
        url: Direct download URL
        output_path: Path to save the file
    """
    download_file(url, output_path)


def download_model(
    model_type: str,
    variant: str = "base",
    force: bool = False,
    registry: Optional[Dict] = None
) -> Optional[Path]:
    """
    Download a model from the registry

    Args:
        model_type: Type of model (craft, crnn, tesseract, trocr)
        variant: Model variant (base, large, etc.)
        force: Force re-download even if file exists
        registry: Model registry dict (loaded if None)

    Returns:
        Path to downloaded model file, or None if failed
    """
    if registry is None:
        registry = load_registry()

    # Check if model type exists
    if model_type not in registry:
        print(f"✗ Unknown model type: {model_type}")
        print(f"Available types: {', '.join(registry.keys())}")
        return None

    # Check if variant exists
    if variant not in registry[model_type]:
        print(f"✗ Unknown variant '{variant}' for {model_type}")
        print(f"Available variants: {', '.join(registry[model_type].keys())}")
        return None

    model_info = registry[model_type][variant]
    model_url = model_info["url"]
    model_download_type = model_info["type"]
    description = model_info.get("description", "")

    print("=" * 70)
    print(f"Model: {model_type}/{variant}")
    print(f"Description: {description}")
    print(f"Type: {model_download_type}")
    print("=" * 70)

    # Handle Hugging Face models
    if model_download_type == "huggingface":
        print("ℹ Hugging Face models are downloaded automatically by transformers")
        print(f"Model ID: {model_url}")
        print("No manual download needed - will download on first use")
        return None

    # Determine output path
    if "filename" in model_info:
        filename = model_info["filename"]
    else:
        # Extract filename from URL
        filename = model_url.split("/")[-1]

    # Create model-specific subdirectory
    model_dir = MODELS_DIR / model_type
    model_dir.mkdir(exist_ok=True)

    output_path = model_dir / filename

    # Check if already exists
    if output_path.exists() and not force:
        print(f"✓ Model already exists: {output_path}")
        print("  Use --force to re-download")
        return output_path

    # Download based on type
    try:
        if model_download_type == "gdrive":
            # Extract Google Drive file ID
            file_id = model_url.split("id=")[1].split("&")[0] if "id=" in model_url else model_url
            download_from_gdrive(file_id, output_path)

        elif model_download_type == "direct":
            download_from_direct_url(model_url, output_path)

        elif model_download_type == "traineddata":
            # Tesseract trained data
            download_from_direct_url(model_url, output_path)

        elif model_download_type == "archive":
            # Download and extract archive (e.g., .tar)
            import tarfile
            import tempfile

            # Download to temporary location
            temp_archive = Path(tempfile.mkdtemp()) / "model.tar"
            download_from_direct_url(model_url, temp_archive)

            # Extract the archive
            print(f"Extracting archive...")
            with tarfile.open(temp_archive) as tar:
                tar.extractall(model_dir)

            # Remove temp file
            temp_archive.unlink()
            print(f"✓ Archive extracted to: {model_dir}")

            # Note: The actual model file might be in a subdirectory
            # Return the directory instead of a specific file
            return model_dir

        else:
            print(f"✗ Unknown download type: {model_download_type}")
            return None

        return output_path

    except Exception as e:
        print(f"✗ Failed to download {model_type}/{variant}: {e}")
        return None


def download_all_models(force: bool = False):
    """Download all models from the registry"""
    registry = load_registry()

    print("\n" + "=" * 70)
    print("Downloading all models from registry")
    print("=" * 70 + "\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for model_type, variants in registry.items():
        for variant in variants:
            print()
            result = download_model(model_type, variant, force=force, registry=registry)

            if result is None:
                # HuggingFace models return None (they auto-download)
                if registry[model_type][variant]["type"] == "huggingface":
                    skip_count += 1
                else:
                    fail_count += 1
            else:
                success_count += 1

    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"✓ Downloaded: {success_count}")
    print(f"⊘ Skipped: {skip_count}")
    if fail_count > 0:
        print(f"✗ Failed: {fail_count}")
    print()


def list_models():
    """List all available models in the registry"""
    registry = load_registry()

    print("\n" + "=" * 70)
    print("Available Models")
    print("=" * 70 + "\n")

    for model_type, variants in registry.items():
        print(f"{model_type}:")
        for variant, info in variants.items():
            desc = info.get("description", "No description")
            model_type_str = info.get("type", "unknown")
            print(f"  {variant:12} ({model_type_str:12}) - {desc}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained models for OCR engines"
    )

    # Model selection
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models"
    )
    parser.add_argument(
        "--craft",
        type=str,
        metavar="VARIANT",
        help="Download CRAFT detector (variants: base, icdar)"
    )
    parser.add_argument(
        "--crnn",
        type=str,
        metavar="VARIANT",
        help="Download CRNN recognizer (variants: base)"
    )
    parser.add_argument(
        "--tesseract",
        type=str,
        metavar="VARIANT",
        help="Download Tesseract trained data (variants: base)"
    )
    parser.add_argument(
        "--db",
        type=str,
        metavar="VARIANT",
        help="Download DB detector (variants: base, mobilenet)"
    )
    parser.add_argument(
        "--kraken",
        type=str,
        metavar="VARIANT",
        help="Download Kraken recognizer (variants: greek, greek_medieval, polytonic)"
    )

    # Options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )

    args = parser.parse_args()

    # List models
    if args.list:
        list_models()
        return

    # Download all
    if args.all:
        download_all_models(force=args.force)
        return

    # Download specific models
    downloaded_any = False

    if args.craft:
        download_model("craft", args.craft, force=args.force)
        downloaded_any = True

    if args.crnn:
        download_model("crnn", args.crnn, force=args.force)
        downloaded_any = True

    if args.tesseract:
        download_model("tesseract", args.tesseract, force=args.force)
        downloaded_any = True

    if args.db:
        download_model("db", args.db, force=args.force)
        downloaded_any = True

    if args.kraken:
        download_model("kraken", args.kraken, force=args.force)
        downloaded_any = True

    # Show help if no action specified
    if not downloaded_any:
        parser.print_help()
        print("\nExamples:")
        print("  python models/download_models.py --list")
        print("  python models/download_models.py --craft base")
        print("  python models/download_models.py --crnn base")
        print("  python models/download_models.py --db base")
        print("  python models/download_models.py --kraken greek")
        print("  python models/download_models.py --all")


if __name__ == "__main__":
    main()
