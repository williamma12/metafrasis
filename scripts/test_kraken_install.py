#!/usr/bin/env python3
"""
Test script to verify kraken OCR library installation
"""

def test_kraken_installation():
    """Test if kraken is properly installed"""

    print("=" * 60)
    print("Kraken OCR Installation Test")
    print("=" * 60)

    # Test 1: Can we import kraken at all?
    print("\n1. Testing basic kraken import...")
    try:
        import kraken
        print("   ✓ kraken package found")
        print(f"   Location: {kraken.__file__}")

        # Try to get version
        try:
            version = kraken.__version__
            print(f"   Version: {version}")
        except AttributeError:
            print("   Version: unknown (no __version__ attribute)")
    except ImportError as e:
        print(f"   ✗ Failed to import kraken: {e}")
        print("\n   Install with: pip install kraken")
        return False

    # Test 2: Check for kraken.lib module
    print("\n2. Testing kraken.lib module...")
    try:
        from kraken import lib
        print("   ✓ kraken.lib module found")
    except ImportError as e:
        print(f"   ✗ Failed to import kraken.lib: {e}")
        print("   This might not be the OCR library!")
        return False

    # Test 3: Check for kraken.lib.models
    print("\n3. Testing kraken.lib.models...")
    try:
        from kraken.lib import models
        print("   ✓ kraken.lib.models found")
    except ImportError as e:
        print(f"   ✗ Failed to import kraken.lib.models: {e}")
        return False

    # Test 4: Check for blla (baseline and layout analysis)
    print("\n4. Testing kraken.blla module...")
    try:
        from kraken import blla
        print("   ✓ kraken.blla found")
    except ImportError as e:
        print(f"   ⚠ Failed to import kraken.blla: {e}")
        print("   This is optional for basic recognition")

    # Test 5: Check for rpred (recognition predictor)
    print("\n5. Testing kraken.rpred module...")
    try:
        from kraken import rpred
        print("   ✓ kraken.rpred found")
    except ImportError as e:
        print(f"   ✗ Failed to import kraken.rpred: {e}")
        return False

    # Test 6: List available functions/classes
    print("\n6. Available kraken modules:")
    import kraken
    kraken_attrs = [attr for attr in dir(kraken) if not attr.startswith('_')]
    for attr in kraken_attrs[:10]:  # Show first 10
        print(f"   - {attr}")
    if len(kraken_attrs) > 10:
        print(f"   ... and {len(kraken_attrs) - 10} more")

    # Test 7: Check dependencies
    print("\n7. Checking key dependencies...")
    dependencies = [
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('torch', 'pytorch'),
    ]

    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"   ✓ {package_name} installed")
        except ImportError:
            print(f"   ✗ {package_name} not installed")

    print("\n" + "=" * 60)
    print("✓ Kraken OCR appears to be properly installed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    success = test_kraken_installation()
    sys.exit(0 if success else 1)
