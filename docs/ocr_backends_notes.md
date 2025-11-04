# OCR Backend Implementation Notes

## Overview
This document details the implementation of multiple OCR backends for the TokenWorks OCR pipeline. The implementation follows a factory pattern design that allows seamless switching between OCR engines while maintaining backward compatibility with existing PaddleOCR functionality.

## Architecture

### Base Interface (`src/ocr_engine.py`)
All OCR engines implement the `OcrEngine` abstract base class:

```python
@abstractmethod
def infer_page(self, image_path: str) -> List[Span]:
    """Infer OCR results for a single page image."""
    pass
```

The `Span` schema normalizes output across all engines:
- `bbox`: [x1, y1, x2, y2] coordinates (floats, 0.0-1.0 normalized)
- `confidence`: Float 0.0-1.0
- `text`: Extracted text string

### Engine Implementations

#### 1. docTR Engine (`src/engines/doctr_engine.py`)
- **License**: Apache-2.0
- **Version**: 1.0.0
- **Documentation**: https://mindee.github.io/doctr/
- **Features**: End-to-end OCR with DBNet detection and SAR/CRNN recognition
- **Dependencies**: PyTorch, torchvision
- **Configuration**: Supports detection and recognition model selection

**Implementation Details**:
- Uses `ocr_predictor` with configurable models
- Converts docTR geometry to normalized bboxes
- Handles both GPU and CPU execution
- Default models: DBNet + SAR for robust performance

#### 2. MMOCR Engine (`src/engines/mmocr_engine.py`)
- **License**: Apache-2.0  
- **Version**: 1.0.1
- **Documentation**: https://mmocr.readthedocs.io/
- **Features**: OpenMMLab framework with state-of-the-art models
- **Dependencies**: PyTorch, MMCV, MMEngine, MMDetection
- **Configuration**: Supports detection and recognition model pairs

**Implementation Details**:
- Uses `MMOCRInferencer` with model pairs (det + rec)
- Complex dependency chain: mmengine → mmcv → mmdet → mmocr
- Default models: DBNet++ detection + ABINet recognition
- Supports extensive model zoo for specialized use cases

#### 3. Kraken Engine (`src/engines/kraken_engine.py`)
- **License**: Apache-2.0
- **Version**: 6.0.0
- **Documentation**: https://kraken.re/master/
- **Features**: Specialized for historical documents and non-Latin scripts
- **Dependencies**: PyTorch, PIL, lxml
- **Configuration**: Supports segmentation and recognition models

**Implementation Details**:
- Two-stage process: segmentation (`blla.segment`) + recognition
- BiDi text support for Arabic/Hebrew scripts
- Optimized for manuscript and historical document processing
- Default: baseline detection + simple character recognition

### Factory Pattern (`src/engines/__init__.py`)
The engine factory provides clean instantiation:

```python
from src.engines import create_engine

engine = create_engine('doctr', config={'det_arch': 'db_resnet50'})
spans = engine.infer_page('path/to/image.jpg')
```

## Configuration System

### Engine Selection (`src/config.py`)
Extended configuration supports engine selection:

```python
OCR_ENGINE = {
    'engine': 'paddle',  # Default: paddle, doctr, mmocr, kraken
    'fallback_to_paddle': True,  # Graceful degradation
    'config': {}  # Engine-specific configuration
}
```

### Profile System
Pre-configured profiles for different use cases:

- **Fast Profile**: `{'engine': 'doctr', 'config': {'det_arch': 'db_mobilenet_v3_large'}}`
- **Quality Profile**: `{'engine': 'mmocr', 'config': {'det': 'DBNet_r50', 'rec': 'ABINet'}}`
- **Historical Profile**: `{'engine': 'kraken', 'config': {'seg_model': 'blla'}}`

## Pipeline Integration

### Backward Compatibility
The implementation maintains full backward compatibility:

1. **Default Behavior**: Existing code continues to use PaddleOCR ensemble
2. **Graceful Fallback**: If specified engine unavailable, falls back to PaddleOCR
3. **Optional Parameters**: New engine parameters are optional in all functions

### Integration Points

#### `src/ocr_utils.py`
New `ocr_with_engine()` function provides engine selection:

```python
def ocr_with_engine(image_path, config=None, engine_name=None, engine_config=None):
    """OCR with configurable engine, fallback to ensemble if engine unavailable."""
```

#### `src/pipeline.py`
Enhanced `process_image()` supports engine parameters:

```python
def process_image(image_path, config, engine_name=None, engine_config=None):
    """Process single image with optional engine selection."""
```

## Evaluation Framework

### Enhanced Evaluation Tool (`tools/run_enhanced_eval.py`)
Comprehensive evaluation supporting all engines:

**Features**:
- Multi-engine comparison
- CER/WER metrics calculation
- Confidence analysis
- Performance benchmarking
- Markdown report generation

**Usage**:
```bash
python tools/run_enhanced_eval.py --engines paddle,doctr,mmocr,kraken \
                                  --gold-csv data/gold_data/gold_pages.csv \
                                  --output-dir results/
```

**Output**:
- Per-engine accuracy metrics
- Processing time comparisons  
- Confidence distribution analysis
- Detailed error analysis

## Docker Integration

### Multi-Engine Support (`Dockerfile`)
Enhanced Dockerfile supports all engines:

1. **Base**: CUDA 12.1 + Python 3.11
2. **PaddleOCR**: Maintained existing installation
3. **PyTorch**: Added CUDA-compatible PyTorch for docTR/MMOCR/Kraken
4. **Optional Engines**: Graceful installation with error handling

### Smoke Tests (`Makefile`)
Added engine-specific smoke tests:

- `make gpu-smoke-doctr`: Test docTR availability
- `make gpu-smoke-mmocr`: Test MMOCR availability  
- `make gpu-smoke-kraken`: Test Kraken availability

## License Analysis

### Approved Engines (Apache-2.0)
All integrated engines use Apache-2.0 license, ensuring:
- ✅ Commercial use permitted
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Patent grant included
- ✅ Compatible with existing codebase

### Rejected Engines
- **Calamari v2.3.1**: GPL-3.0 license (copyleft incompatible)

## Usage Examples

### Basic Engine Selection
```python
from src.config import get_ocr_engine_config
from src.ocr_utils import ocr_with_engine

# Use docTR engine
config = get_ocr_engine_config('doctr')
spans = ocr_with_engine('image.jpg', engine_name='doctr')

# Use MMOCR with custom model
spans = ocr_with_engine('image.jpg', 
                       engine_name='mmocr',
                       engine_config={'det': 'DBNet_r50', 'rec': 'PARSeq'})
```

### Pipeline Integration
```python
from src.pipeline import process_image

# Process with Kraken for historical documents
result = process_image('manuscript.jpg', config, 
                      engine_name='kraken',
                      engine_config={'seg_model': 'blla'})
```

### Evaluation Workflow
```bash
# Compare all engines
python tools/run_enhanced_eval.py --engines all --limit 10

# Test specific engine
python tools/run_enhanced_eval.py --engines doctr --gold-csv gold.csv
```

## Implementation References

### Documentation Sources
- **docTR**: https://mindee.github.io/doctr/using_models.html
- **MMOCR**: https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html
- **Kraken**: https://kraken.re/master/ketos.html

### Key Implementation Patterns
1. **Conditional Imports**: All engines use try/except imports with availability flags
2. **Normalization**: Consistent bbox/confidence/text normalization across engines
3. **Error Handling**: Graceful degradation when engines unavailable
4. **Configuration**: Flexible engine-specific configuration system
5. **Factory Pattern**: Clean instantiation and engine management

## Testing Strategy

### Smoke Tests
- Engine availability verification
- Version checking
- Basic model loading
- Error handling validation

### Integration Tests
- Cross-engine output comparison
- Performance benchmarking
- Accuracy validation against gold standard
- Configuration system testing

### Acceptance Criteria
- ✅ All engines implement OcrEngine interface
- ✅ Backward compatibility maintained
- ✅ Graceful fallback to PaddleOCR
- ✅ Docker builds successfully with all dependencies
- ✅ Evaluation tools support all engines
- ✅ Documentation complete and accurate