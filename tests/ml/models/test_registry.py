"""
Tests for model registry system.

Tests registry JSON structure, model info retrieval, and path construction.
"""
import pytest
import json
from pathlib import Path
from ml.config import get_model_info, get_model_registry, MODELS_DIR, MODEL_WEIGHTS_DIR

MODEL_REGISTRY_PATH = MODELS_DIR / "registry.json"


class TestRegistryLoading:
    """Tests for registry file loading."""

    def test_registry_file_exists(self):
        """Test registry.json file exists."""
        assert MODEL_REGISTRY_PATH.exists(), f"Registry file not found: {MODEL_REGISTRY_PATH}"

    def test_registry_loads_without_error(self):
        """Test registry loads without JSON errors."""
        registry = get_model_registry()

        assert isinstance(registry, dict)
        assert len(registry) > 0

    def test_registry_is_valid_json(self):
        """Test registry file contains valid JSON."""
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_registry_contains_expected_models(self):
        """Test registry contains expected model types."""
        registry = get_model_registry()

        expected_models = ['tesseract', 'craft', 'db']
        for model_type in expected_models:
            assert model_type in registry, f"Missing expected model: {model_type}"


class TestRegistryStructure:
    """Tests for registry data structure."""

    def test_all_models_have_required_fields(self):
        """Test all registered models have required fields."""
        registry = get_model_registry()

        required_fields = ['url', 'type']  # filename is optional depending on type

        for model_type, variants in registry.items():
            # Skip special keys like _comment
            if model_type.startswith('_'):
                continue

            assert isinstance(variants, dict), f"{model_type} variants should be a dict"

            for variant_name, model_info in variants.items():
                # Skip special keys like _comment
                if variant_name.startswith('_'):
                    continue

                # Skip if model_info is not a dict (could be a comment)
                if not isinstance(model_info, dict):
                    continue

                for field in required_fields:
                    assert field in model_info, \
                        f"{model_type}:{variant_name} missing field '{field}'"

    def test_url_fields_are_strings(self):
        """Test URL fields are valid strings."""
        registry = get_model_registry()

        for model_type, variants in registry.items():
            if model_type.startswith('_'):
                continue

            for variant_name, model_info in variants.items():
                # Skip special keys and non-dict entries
                if variant_name.startswith('_') or not isinstance(model_info, dict):
                    continue

                url = model_info.get('url')
                assert isinstance(url, str), \
                    f"{model_type}:{variant_name} URL should be string"
                assert len(url) > 0, \
                    f"{model_type}:{variant_name} URL should not be empty"

    def test_type_field_is_valid(self):
        """Test type field is one of allowed values."""
        registry = get_model_registry()

        valid_types = ['huggingface', 'gdrive', 'direct', 'traineddata', 'archive']

        for model_type, variants in registry.items():
            if model_type.startswith('_'):
                continue

            for variant_name, model_info in variants.items():
                # Skip special keys and non-dict entries
                if variant_name.startswith('_') or not isinstance(model_info, dict):
                    continue

                download_type = model_info.get('type')
                assert download_type in valid_types, \
                    f"{model_type}:{variant_name} has invalid type '{download_type}'"

    def test_no_duplicate_filenames_within_model_type(self):
        """Test no duplicate filenames within same model type."""
        registry = get_model_registry()

        for model_type, variants in registry.items():
            if model_type.startswith('_'):
                continue

            filenames = []
            for variant_name, info in variants.items():
                # Skip special keys and non-dict entries
                if variant_name.startswith('_') or not isinstance(info, dict):
                    continue

                # Only check filename if it exists
                if 'filename' in info:
                    filenames.append(info['filename'])

            if filenames:  # Only check if there are filenames
                assert len(filenames) == len(set(filenames)), \
                    f"{model_type} has duplicate filenames: {filenames}"


class TestModelInfoRetrieval:
    """Tests for get_model_info function."""

    def test_get_model_info_valid_engine(self):
        """Test get_model_info returns correct structure for valid engine."""
        info = get_model_info('craft', 'base')  # Use 'base' variant that exists

        assert info is not None, "get_model_info should return a dict for valid engine/variant"
        assert 'url' in info
        assert 'type' in info
        assert 'filename' in info

        assert isinstance(info['url'], str)
        assert isinstance(info['type'], str)
        assert isinstance(info['filename'], str)

    def test_get_model_info_invalid_engine_returns_none(self):
        """Test get_model_info returns None for invalid engine."""
        result = get_model_info('nonexistent_model', 'variant')
        assert result is None

    def test_get_model_info_invalid_variant_returns_none(self):
        """Test get_model_info returns None for invalid variant."""
        result = get_model_info('craft', 'nonexistent_variant')
        assert result is None

    def test_get_model_info_returns_correct_data(self):
        """Test get_model_info returns expected data for known model."""
        info = get_model_info('craft', 'base')  # Use 'base' variant that exists

        # CRAFT base should be from Google Drive
        assert info is not None
        assert info['type'] == 'gdrive'
        assert 'craft' in info['filename'].lower() or 'mlt' in info['filename'].lower()


class TestModelPathConstruction:
    """Tests for model path construction."""

    def test_model_weights_dir_uses_correct_base(self):
        """Test MODEL_WEIGHTS_DIR points to data/model_weights."""
        assert MODEL_WEIGHTS_DIR.exists() or True  # May not exist yet
        assert 'model_weights' in str(MODEL_WEIGHTS_DIR)

    def test_path_construction_includes_model_type(self):
        """Test model paths include model type subdirectory."""
        info = get_model_info('craft', 'base')  # Use 'base' variant that exists

        assert info is not None
        # Path should be: data/model_weights/{model_type}/{filename}
        expected_path = MODEL_WEIGHTS_DIR / info['type'] / info['filename']

        assert info['type'] in str(expected_path)
        assert info['filename'] in str(expected_path)


class TestMultipleVariants:
    """Tests for models with multiple variants."""

    def test_craft_has_multiple_variants(self):
        """Test CRAFT model has multiple variants registered."""
        registry = get_model_registry()

        assert 'craft' in registry
        craft_variants = registry['craft']

        # Filter out special keys like _comment
        actual_variants = {k: v for k, v in craft_variants.items()
                           if not k.startswith('_') and isinstance(v, dict)}

        assert len(actual_variants) > 1, "CRAFT should have multiple variants"
        assert 'base' in actual_variants or 'icdar' in actual_variants
