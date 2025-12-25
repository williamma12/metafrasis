"""
OCR Engine Factory - creates engines from detectors + recognizers
"""
from typing import Dict, Type, List, Optional, Union
from .base import OCREngine
from .types import DetectorType, RecognizerType, EngineType


class OCREngineFactory:
    """
    Factory for creating OCR engine instances

    Supports two creation modes:
    1. Monolithic engine: create(engine=EngineType.TESSERACT)
    2. Composed engine: create(detector=DetectorType.WHOLE_IMAGE, recognizer=RecognizerType.TROCR)
    """

    # Monolithic engines
    _engines: Dict[str, Type[OCREngine]] = {}

    # Modular components
    _detectors: Dict[str, Type] = {}
    _recognizers: Dict[str, Type] = {}

    @classmethod
    def create(
        cls,
        engine: Optional[Union[EngineType, str]] = None,
        detector: Optional[Union[DetectorType, str]] = None,
        recognizer: Optional[Union[RecognizerType, str]] = None,
        **kwargs
    ) -> OCREngine:
        """
        Create an OCR engine instance

        Args:
            engine: Monolithic engine (e.g., EngineType.TESSERACT or 'tesseract')
            detector: Detector (e.g., DetectorType.WHOLE_IMAGE or 'whole_image')
            recognizer: Recognizer (e.g., RecognizerType.TROCR or 'trocr')
            **kwargs: Engine-specific configuration

        Returns:
            OCREngine instance

        Examples:
            # Monolithic engine
            engine = OCREngineFactory.create(engine=EngineType.TESSERACT)
            engine = OCREngineFactory.create(engine='tesseract')

            # Composed engine
            engine = OCREngineFactory.create(
                detector=DetectorType.WHOLE_IMAGE,
                recognizer=RecognizerType.TROCR,
                device='cuda'
            )

        Raises:
            ValueError: If parameters are invalid or not found
        """
        # Convert enums to strings
        if isinstance(engine, EngineType):
            engine = engine.value
        if isinstance(detector, DetectorType):
            detector = detector.value
        if isinstance(recognizer, RecognizerType):
            recognizer = recognizer.value

        # Validate parameters
        if engine is not None and (detector is not None or recognizer is not None):
            raise ValueError(
                "Cannot specify both 'engine' and 'detector'/'recognizer'. "
                "Use 'engine' for monolithic engines OR 'detector'+'recognizer' for composed engines."
            )

        if engine is None and (detector is None or recognizer is None):
            raise ValueError(
                "Must specify either 'engine' OR both 'detector' and 'recognizer'. "
                f"Available engines: {', '.join(cls._engines.keys())}. "
                f"Available detectors: {', '.join(cls._detectors.keys())}. "
                f"Available recognizers: {', '.join(cls._recognizers.keys())}."
            )

        # Create monolithic engine
        if engine is not None:
            if engine not in cls._engines:
                available = ', '.join(cls._engines.keys()) or 'none'
                raise ValueError(
                    f"Unknown engine: '{engine}'. "
                    f"Available engines: {available}"
                )
            return cls._engines[engine](**kwargs)

        # Create composed engine
        if detector not in cls._detectors:
            available = ', '.join(cls._detectors.keys()) or 'none'
            raise ValueError(
                f"Unknown detector: '{detector}'. "
                f"Available detectors: {available}"
            )

        if recognizer not in cls._recognizers:
            available = ', '.join(cls._recognizers.keys()) or 'none'
            raise ValueError(
                f"Unknown recognizer: '{recognizer}'. "
                f"Available recognizers: {available}"
            )

        # Import PyTorchOCREngine here to avoid circular import
        from .engines.pytorch_engine import PyTorchOCREngine

        # Create detector and recognizer instances
        detector_instance = cls._detectors[detector](**kwargs)
        recognizer_instance = cls._recognizers[recognizer](**kwargs)

        # Compose into PyTorchOCREngine
        return PyTorchOCREngine(
            detector=detector_instance,
            recognizer=recognizer_instance,
            **kwargs
        )

    @classmethod
    def available_engines(cls) -> List[str]:
        """Get list of available monolithic engine names"""
        return sorted(cls._engines.keys())

    @classmethod
    def available_detectors(cls) -> List[str]:
        """Get list of available detector names"""
        return sorted(cls._detectors.keys())

    @classmethod
    def available_recognizers(cls) -> List[str]:
        """Get list of available recognizer names"""
        return sorted(cls._recognizers.keys())

    @classmethod
    def register_engine(cls, name: str, engine_class: Type[OCREngine]):
        """Register a monolithic engine"""
        cls._engines[name] = engine_class

    @classmethod
    def register_detector(cls, name: str, detector_class: Type):
        """Register a detector"""
        cls._detectors[name] = detector_class

    @classmethod
    def register_recognizer(cls, name: str, recognizer_class: Type):
        """Register a recognizer"""
        cls._recognizers[name] = recognizer_class
