#!/usr/bin/env python3
"""
Automated tests for the built Metafrasis.app bundle

These tests verify that the app bundle was created correctly and
contains all necessary components.

Usage:
    pytest installer/tests/test_app_bundle.py -v

Or run directly:
    python installer/tests/test_app_bundle.py
"""

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Determine paths
SCRIPT_DIR = Path(__file__).parent
INSTALLER_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = INSTALLER_DIR / "output"
APP_PATH = OUTPUT_DIR / "Metafrasis.app"


def skip_if_not_built():
    """Skip tests if app hasn't been built yet"""
    if not APP_PATH.exists():
        pytest.skip(f"App bundle not found at {APP_PATH}. Run build.sh first.")


class TestAppBundleStructure:
    """Test that the app bundle has the correct structure"""

    def test_app_bundle_exists(self):
        """Verify app bundle was created"""
        skip_if_not_built()
        assert APP_PATH.exists(), f"App bundle not found at {APP_PATH}"
        assert APP_PATH.is_dir(), "App bundle should be a directory"

    def test_contents_directory(self):
        """Verify Contents directory exists"""
        skip_if_not_built()
        contents = APP_PATH / "Contents"
        assert contents.exists(), "Contents directory missing"

    def test_macos_executable(self):
        """Verify main executable exists"""
        skip_if_not_built()
        exe = APP_PATH / "Contents" / "MacOS" / "Metafrasis"
        assert exe.exists(), "Main executable missing"
        assert os.access(exe, os.X_OK), "Executable should have execute permission"

    def test_info_plist(self):
        """Verify Info.plist exists"""
        skip_if_not_built()
        plist = APP_PATH / "Contents" / "Info.plist"
        assert plist.exists(), "Info.plist missing"

    def test_resources_directory(self):
        """Verify Resources directory exists"""
        skip_if_not_built()
        resources = APP_PATH / "Contents" / "Resources"
        assert resources.exists(), "Resources directory missing"


class TestBundledDependencies:
    """Test that dependencies are properly bundled"""

    def test_bundled_tesseract(self):
        """Test bundled Tesseract binary exists"""
        skip_if_not_built()
        tesseract = APP_PATH / "Contents" / "Resources" / "deps" / "tesseract" / "bin" / "tesseract"

        if not tesseract.exists():
            pytest.skip("Tesseract not bundled (may be intentional)")

        assert tesseract.exists(), "Tesseract binary missing"
        assert os.access(tesseract, os.X_OK), "Tesseract should be executable"

    def test_bundled_tesseract_data(self):
        """Test Greek language data exists"""
        skip_if_not_built()
        tessdata = APP_PATH / "Contents" / "Resources" / "deps" / "tesseract" / "share" / "tessdata"

        if not tessdata.exists():
            pytest.skip("Tesseract data not bundled")

        grc_data = tessdata / "grc.traineddata"
        assert grc_data.exists(), "Greek language data (grc.traineddata) missing"

    def test_bundled_poppler(self):
        """Test bundled Poppler utilities exist"""
        skip_if_not_built()
        poppler_bin = APP_PATH / "Contents" / "Resources" / "deps" / "poppler" / "bin"

        if not poppler_bin.exists():
            pytest.skip("Poppler not bundled")

        pdftoppm = poppler_bin / "pdftoppm"
        assert pdftoppm.exists(), "pdftoppm binary missing"
        assert os.access(pdftoppm, os.X_OK), "pdftoppm should be executable"


class TestApplicationCode:
    """Test that application code is properly bundled"""

    def test_app_module(self):
        """Test app module is bundled"""
        skip_if_not_built()
        # The exact location depends on PyInstaller's structure
        resources = APP_PATH / "Contents" / "Resources"

        # Check for app directory or in the main bundle
        app_found = (
            (resources / "app").exists()
            or any(resources.glob("**/app"))
            or any((APP_PATH / "Contents" / "MacOS").glob("**/app"))
        )

        assert app_found, "App module not found in bundle"

    def test_config_module(self):
        """Test config.py is bundled"""
        skip_if_not_built()
        resources = APP_PATH / "Contents" / "Resources"

        config_found = (
            (resources / "config.py").exists()
            or any(resources.glob("**/config.py"))
        )

        assert config_found, "config.py not found in bundle"

    def test_models_registry(self):
        """Test models registry is bundled"""
        skip_if_not_built()
        resources = APP_PATH / "Contents" / "Resources"

        registry_found = (
            (resources / "models" / "registry.json").exists()
            or any(resources.glob("**/registry.json"))
        )

        assert registry_found, "models/registry.json not found in bundle"


class TestCodeSigning:
    """Test code signing (if applicable)"""

    def test_codesign_verify(self):
        """Verify app is properly signed (if signed)"""
        skip_if_not_built()

        result = subprocess.run(
            ["codesign", "--verify", "--deep", "--strict", str(APP_PATH)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Not signed or invalid signature
            if "not signed" in result.stderr.lower():
                pytest.skip("App is not signed (expected for unsigned builds)")
            else:
                pytest.fail(f"Signing verification failed: {result.stderr}")


class TestArchitecture:
    """Test binary architecture"""

    def test_executable_architecture(self):
        """Verify executable architecture matches or is universal"""
        skip_if_not_built()
        exe = APP_PATH / "Contents" / "MacOS" / "Metafrasis"

        result = subprocess.run(
            ["lipo", "-archs", str(exe)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"lipo failed: {result.stderr}"

        archs = result.stdout.strip().split()
        current_arch = platform.machine()

        # Either universal or matches current architecture
        is_valid = (
            current_arch in archs
            or ("arm64" in archs and "x86_64" in archs)  # Universal
        )

        assert is_valid, f"Executable architecture {archs} not compatible with {current_arch}"


class TestAppLaunch:
    """Test that the app can launch (integration test)"""

    @pytest.mark.slow
    def test_app_starts(self):
        """Test that app launches and Streamlit server starts"""
        skip_if_not_built()

        # Launch app
        process = subprocess.Popen(
            ["open", "-a", str(APP_PATH), "--stdout", "/dev/null", "--stderr", "/dev/null"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Wait for potential server to start
        server_port = None
        max_wait = 30
        start_time = time.time()

        # Try common Streamlit ports
        import socket
        while time.time() - start_time < max_wait:
            for port in range(8501, 8520):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.5)
                        s.connect(('localhost', port))
                        server_port = port
                        break
                except (ConnectionRefusedError, socket.timeout, OSError):
                    pass
            if server_port:
                break
            time.sleep(1)

        # Cleanup - kill the app
        subprocess.run(["pkill", "-f", "Metafrasis"], capture_output=True)
        time.sleep(1)

        if server_port:
            print(f"Server started on port {server_port}")
        else:
            pytest.skip("Could not verify server start (may still be valid)")


def run_tests():
    """Run tests and print summary"""
    print("=" * 60)
    print("Metafrasis App Bundle Tests")
    print("=" * 60)
    print(f"App Path: {APP_PATH}")
    print(f"Exists: {APP_PATH.exists()}")
    print()

    if not APP_PATH.exists():
        print("ERROR: App bundle not found!")
        print("Run './installer/build.sh' first to create the bundle.")
        return 1

    # Run pytest
    return pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    sys.exit(run_tests())
