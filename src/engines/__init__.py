"""
OCR Engines Package

This package contains implementations of various OCR engines that conform
to the OcrEngine interface defined in ocr_engine.py.
"""

# Import engine classes for easy access
try:
    from .doctr_engine import DoctrEngine, DoctrEngineFactory
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

try:
    from .mmocr_engine import MmocrEngine, MmocrEngineFactory
    MMOCR_AVAILABLE = True
except ImportError:
    MMOCR_AVAILABLE = False

try:
    from .kraken_engine import KrakenEngine, KrakenEngineFactory
    KRAKEN_AVAILABLE = True
except ImportError:
    KRAKEN_AVAILABLE = False

# Engine availability flags
ENGINE_AVAILABILITY = {
    'doctr': DOCTR_AVAILABLE,
    'mmocr': MMOCR_AVAILABLE,
    'kraken': KRAKEN_AVAILABLE,
    'paddle': True  # Always available in our pipeline
}

# Engine factory mapping
ENGINE_FACTORIES = {}

if DOCTR_AVAILABLE:
    ENGINE_FACTORIES['doctr'] = DoctrEngineFactory

if MMOCR_AVAILABLE:
    ENGINE_FACTORIES['mmocr'] = MmocrEngineFactory

if KRAKEN_AVAILABLE:
    ENGINE_FACTORIES['kraken'] = KrakenEngineFactory


def get_available_engines():
    """Get list of available OCR engines."""
    return [name for name, available in ENGINE_AVAILABILITY.items() if available]


def create_engine(engine_name: str, profile: str = "balanced", device: str = "auto"):
    """
    Create an OCR engine instance.
    
    Args:
        engine_name: Name of the engine ('doctr', 'mmocr', 'kraken', 'paddle')
        profile: Performance profile ('fast', 'balanced', 'quality')
        device: Device to use ('auto', 'cpu', 'cuda')
        
    Returns:
        OCR engine instance
        
    Raises:
        ValueError: If engine is not available
        NotImplementedError: If profile is not supported
    """
    if engine_name == 'paddle':
        # Return mock for now - will be implemented in pipeline integration
        from ..ocr_engine import MockOcrEngine
        return MockOcrEngine({'device': device})
    
    if engine_name not in ENGINE_AVAILABILITY or not ENGINE_AVAILABILITY[engine_name]:
        available = get_available_engines()
        raise ValueError(f"Engine '{engine_name}' not available. Available engines: {available}")
    
    factory = ENGINE_FACTORIES[engine_name]
    
    # Create engine based on profile
    if profile == "fast":
        if hasattr(factory, 'create_fast_engine'):
            return factory.create_fast_engine(device)
        else:
            raise NotImplementedError(f"Fast profile not implemented for {engine_name}")
    elif profile == "quality":
        if hasattr(factory, 'create_quality_engine'):
            return factory.create_quality_engine(device)
        else:
            raise NotImplementedError(f"Quality profile not implemented for {engine_name}")
    elif profile == "balanced":
        # Try quality first, fall back to fast, then default
        if hasattr(factory, 'create_quality_engine'):
            return factory.create_quality_engine(device)
        elif hasattr(factory, 'create_fast_engine'):
            return factory.create_fast_engine(device)
        else:
            # Create with default config
            engine_classes = {
                'doctr': DoctrEngine if DOCTR_AVAILABLE else None,
                'mmocr': MmocrEngine if MMOCR_AVAILABLE else None,
                'kraken': KrakenEngine if KRAKEN_AVAILABLE else None
            }
            engine_class = engine_classes[engine_name]
            if engine_class:
                return engine_class({'device': device})
            else:
                raise ValueError(f"Could not create {engine_name} engine")
    else:
        raise ValueError(f"Unknown profile: {profile}. Supported: fast, balanced, quality")


def check_engine_dependencies():
    """
    Check which OCR engine dependencies are installed.
    
    Returns:
        Dictionary with dependency status for each engine
    """
    status = {}
    
    # Check docTR
    try:
        import doctr
        status['doctr'] = {'available': True, 'version': doctr.__version__}
    except ImportError:
        status['doctr'] = {'available': False, 'error': 'doctr not installed'}
    
    # Check MMOCR
    try:
        import mmocr
        import mmcv
        import mmdet
        status['mmocr'] = {
            'available': True, 
            'version': mmocr.__version__,
            'mmcv_version': mmcv.__version__,
            'mmdet_version': mmdet.__version__
        }
    except ImportError as e:
        status['mmocr'] = {'available': False, 'error': str(e)}
    
    # Check Kraken
    try:
        import kraken
        status['kraken'] = {'available': True, 'version': kraken.__version__}
    except ImportError:
        status['kraken'] = {'available': False, 'error': 'kraken not installed'}
    
    # PaddleOCR (always available in our setup)
    try:
        import paddleocr
        status['paddle'] = {'available': True, 'version': paddleocr.__version__}
    except ImportError:
        status['paddle'] = {'available': False, 'error': 'paddleocr not installed'}
    
    return status


# Export main classes and functions
__all__ = [
    'ENGINE_AVAILABILITY',
    'get_available_engines', 
    'create_engine',
    'check_engine_dependencies'
]

# Conditionally export engine classes if available
if DOCTR_AVAILABLE:
    __all__.extend(['DoctrEngine', 'DoctrEngineFactory'])

if MMOCR_AVAILABLE:
    __all__.extend(['MmocrEngine', 'MmocrEngineFactory'])

if KRAKEN_AVAILABLE:
    __all__.extend(['KrakenEngine', 'KrakenEngineFactory'])