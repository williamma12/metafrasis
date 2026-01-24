#!/usr/bin/env python3
"""
Metafrasis Launcher - Manages Streamlit server lifecycle for desktop app

This script:
1. Finds an available port
2. Starts Streamlit server in background
3. Opens default browser
4. Monitors for app closure
5. Cleanly terminates server on exit
"""
import atexit
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Global reference to subprocess
streamlit_process = None


def get_bundle_paths():
    """
    Determine paths based on execution context (bundled vs development)

    Returns:
        tuple: (bundle_dir, app_dir, deps_dir)
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        # sys._MEIPASS is the temp extraction directory
        bundle_dir = Path(sys._MEIPASS)

        # The .app structure: Contents/MacOS/Metafrasis (executable)
        # Resources are at: Contents/Resources/
        executable_dir = Path(os.path.dirname(sys.executable))
        app_dir = executable_dir.parent / "Resources"

        # Bundled dependencies (Tesseract, Poppler)
        deps_dir = app_dir / "deps"
    else:
        # Running in development mode
        bundle_dir = Path(__file__).parent.parent.parent
        app_dir = bundle_dir
        deps_dir = None

    return bundle_dir, app_dir, deps_dir


def find_free_port():
    """Find an available port for Streamlit server"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockname()
        return s.getsockname()[1]


def setup_environment(app_dir, deps_dir):
    """
    Configure environment for bundled execution

    Args:
        app_dir: Path to application resources
        deps_dir: Path to bundled dependencies (Tesseract, Poppler)
    """
    # Set root for config.py to find models/registry.json etc.
    os.environ["METAFRASIS_ROOT"] = str(app_dir)

    # Force Streamlit to use production React components
    os.environ["VIEWER_RELEASE"] = "true"
    os.environ["ANNOTATION_CANVAS_RELEASE"] = "true"

    # Disable Streamlit's browser auto-open (we handle it ourselves)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

    # Disable Streamlit usage stats
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    if deps_dir and deps_dir.exists():
        # Set up paths for bundled Tesseract
        tesseract_dir = deps_dir / "tesseract"
        if tesseract_dir.exists():
            tesseract_bin = tesseract_dir / "bin"
            tessdata = tesseract_dir / "share" / "tessdata"

            # Add to PATH
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{tesseract_bin}:{current_path}"

            # Set Tesseract data directory
            os.environ["TESSDATA_PREFIX"] = str(tessdata)

            # Set library path for dylibs
            tesseract_lib = tesseract_dir / "lib"
            dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = f"{tesseract_lib}:{dyld_path}"

        # Set up paths for bundled Poppler (pdf2image)
        poppler_dir = deps_dir / "poppler"
        if poppler_dir.exists():
            poppler_bin = poppler_dir / "bin"

            # Add to PATH
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{poppler_bin}:{current_path}"

            # Set library path
            poppler_lib = poppler_dir / "lib"
            dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = f"{poppler_lib}:{dyld_path}"


def setup_architecture_specific_paths(app_dir):
    """
    Set up architecture-specific library paths (for universal binary)

    On universal builds, PyTorch is stored separately for each architecture.
    This function adds the correct one to sys.path based on the current arch.
    """
    arch = platform.machine()

    if arch == "x86_64":
        # Intel Mac - check for x86_64-specific torch
        torch_x86 = app_dir / "torch_x86_64"
        if torch_x86.exists():
            sys.path.insert(0, str(torch_x86))
    # arm64 (Apple Silicon) uses the default bundled torch


def cleanup():
    """Clean up Streamlit process on exit"""
    global streamlit_process
    if streamlit_process:
        print("Shutting down Streamlit server...")
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing Streamlit server...")
            streamlit_process.kill()
        streamlit_process = None


def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    cleanup()
    sys.exit(0)


def wait_for_server(port, timeout=30):
    """
    Wait for Streamlit server to be ready

    Args:
        port: Port to check
        timeout: Maximum wait time in seconds

    Returns:
        bool: True if server is ready, False if timed out
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('localhost', port))
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    return False


def main():
    global streamlit_process

    print("Starting Metafrasis...")

    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Get paths based on execution context
    bundle_dir, app_dir, deps_dir = get_bundle_paths()

    # Set up environment
    setup_environment(app_dir, deps_dir)
    setup_architecture_specific_paths(app_dir)

    # Find free port
    port = find_free_port()
    url = f"http://localhost:{port}"

    print(f"Starting Streamlit on port {port}...")

    # Determine the app.py location
    app_py = app_dir / "app.py"
    if not app_py.exists():
        print(f"Error: Could not find app.py at {app_py}")
        sys.exit(1)

    # Build Streamlit command
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_py),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light",
        "--server.fileWatcherType", "none",  # Disable file watcher in bundled mode
    ]

    # Start Streamlit server
    try:
        streamlit_process = subprocess.Popen(
            streamlit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(app_dir),
            env=os.environ.copy(),
        )
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        sys.exit(1)

    # Wait for server to start
    print("Waiting for server to be ready...")
    if not wait_for_server(port):
        print("Error: Streamlit server failed to start within timeout")
        # Try to get any error output
        if streamlit_process.poll() is not None:
            stdout, _ = streamlit_process.communicate()
            print(f"Server output: {stdout.decode()}")
        cleanup()
        sys.exit(1)

    print(f"Server ready! Opening browser to {url}")

    # Open browser
    webbrowser.open(url)

    # Keep running until the Streamlit process ends or we receive a signal
    try:
        # Wait for Streamlit to exit
        return_code = streamlit_process.wait()
        if return_code != 0:
            print(f"Streamlit exited with code {return_code}")
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
