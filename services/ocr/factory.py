"""
OCR Engine Factory - simple dict-based engine creation
"""
from typing import Dict, Type, List
from .base import OCREngine


class OCREngineFactory:
    """Factory for creating OCR engine instances"""

    # Simple dict mapping engine names to classes
    # Engines are imported and added to this dict in __init__.py
    _engines: Dict[str, Type[OCREngine]] = {}

    @classmethod
    def create(cls, engine_name: str, **kwargs) -> OCREngine:
        """
        Create an OCR engine instance

        Args:
            engine_name: Name of the engine to create
            **kwargs: Engine-specific configuration

        Returns:
            OCREngine instance

        Raises:
            ValueError: If engine name is not found
        """
        if engine_name not in cls._engines:
            available = ', '.join(cls._engines.keys()) or 'none'
            raise ValueError(
                f"Unknown engine: '{engine_name}'. "
                f"Available engines: {available}"
            )

        engine_class = cls._engines[engine_name]
        return engine_class(**kwargs)

    @classmethod
    def available_engines(cls) -> List[str]:
        """
        Get list of available engine names

        Returns:
            List of engine names
        """
        return list(cls._engines.keys())
