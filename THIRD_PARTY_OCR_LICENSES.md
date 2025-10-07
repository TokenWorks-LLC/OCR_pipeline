# Third-Party OCR Engine Licenses and Integration Decisions

This document summarizes the license analysis and integration decisions for candidate OCR engines.

## License Analysis Summary

| Engine | Repository | License | License File | Decision | Reason |
|--------|------------|---------|--------------|----------|---------|
| docTR | https://github.com/mindee/doctr | Apache-2.0 | LICENSE | ✅ **INTEGRATE** | Permissive Apache-2.0 license, compatible with our project |
| MMOCR | https://github.com/open-mmlab/mmocr | Apache-2.0 | LICENSE | ✅ **INTEGRATE** | Permissive Apache-2.0 license, compatible with our project |
| Kraken | https://github.com/mittagessen/kraken | Apache-2.0 | LICENSE | ✅ **INTEGRATE** | Permissive Apache-2.0 license, compatible with our project |
| Calamari | https://github.com/Calamari-OCR/calamari | GPL-3.0 | LICENSE | ❌ **SKIP** | GPL-3.0 is copyleft license - incompatible with commercial use |

## Documentation References

### docTR (Mindee) - Version v1.0.0
- **Documentation URL:** https://mindee.github.io/doctr/
- **Repository:** https://github.com/mindee/doctr (commit: main branch)
- **License:** Apache-2.0
- **Installation:** `pip install python-doctr`
- **Key Features:**
  - End-to-end OCR with text detection + recognition
  - Default models: DBNet (detection) + CRNN/SAR (recognition)
  - PyTorch backend with GPU support
  - Pre-trained models available
- **Integration Notes:**
  - Use `ocr_predictor(det_arch='db_resnet50', reco_arch='sar', pretrained=True)`
  - GPU support via PyTorch CUDA
  - Returns structured document with pages/blocks/lines/words

### MMOCR (OpenMMLab) - Version v1.0.1
- **Documentation URL:** https://mmocr.readthedocs.io/en/dev-1.x/
- **Repository:** https://github.com/open-mmlab/mmocr (commit: main branch)
- **License:** Apache-2.0
- **Installation:** Requires MMEngine, MMCV, MMDetection
- **Key Features:**
  - Modular design with state-of-the-art models
  - DBNet++, ABINet, PARSeq support
  - Comprehensive model zoo
- **Integration Notes:**
  - Detector: DBNet++ (recommended)
  - Recognizer: ABINet (quality) or PARSeq (speed)
  - Requires specific MMCV version compatibility
  - GPU support through MMDetection framework

### Kraken - Version 6.0.0  
- **Documentation URL:** https://kraken.re/
- **Repository:** https://github.com/mittagessen/kraken (commit: main branch)
- **License:** Apache-2.0
- **Installation:** `pip install kraken`
- **Key Features:**
  - Optimized for historical and non-Latin scripts
  - Right-to-Left, BiDi script support  
  - Public model repository on Zenodo
  - Variable network architectures
- **Integration Notes:**
  - Line-level recognition engine
  - Best suited for Akkadian/specialized scripts
  - Models available via `kraken get <model_id>`
  - Command-line and Python API available

### Calamari - Version v2.3.1 (SKIPPED)
- **Documentation URL:** https://calamari-ocr.readthedocs.io/
- **Repository:** https://github.com/Calamari-OCR/calamari (commit: master branch)  
- **License:** GPL-3.0 (Copyleft)
- **Integration Decision:** ❌ **SKIP** - GPL-3.0 license incompatible with commercial use
- **License Excerpt:**
  ```
  GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
  Copyright 2020 The tfaip Authors. All rights reserved.
  ```
- **Reason:** GPL-3.0 is a copyleft license that would require our entire codebase to be GPL-licensed, incompatible with commercial usage and Apache-2.0 components.

## Integration Plan

Based on license compatibility, we will integrate:

1. **docTR** - Primary alternative OCR engine with excellent documentation
2. **MMOCR** - Research-grade engine with advanced models  
3. **Kraken** - Specialized engine for historical/non-Latin scripts (optional, behind feature flag)

All three engines use Apache-2.0 licenses and are fully compatible with our project's licensing requirements.

## Version Pinning Strategy

- **docTR:** Use latest stable (v1.0.0+) 
- **MMOCR:** Pin to v1.0.1 with compatible MMCV version
- **Kraken:** Use latest stable (v6.0.0+)

All engines will be optional dependencies to maintain lean installation for users who only need Paddle OCR.