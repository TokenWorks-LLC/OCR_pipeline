"""
Configuration system with tuning profiles for fast vs quality OCR processing.
Provides documented parameter sets for different use cases.
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class OCRParameters:
    """OCR engine parameters."""
    # PaddleOCR detection parameters
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.6
    det_db_unclip_ratio: float = 1.5
    det_limit_side_len: int = 960
    
    # PaddleOCR recognition parameters  
    rec_score_thresh: float = 0.5
    rec_batch_num: int = 6
    
    # GPU settings
    gpu_mem: int = 4000
    use_tensorrt: bool = False

@dataclass
class PreprocessingParameters:
    """Image preprocessing parameters."""
    enable_deskew: bool = True
    enable_denoise: bool = True
    enable_contrast: bool = True
    enable_column_detection: bool = True
    enable_footnote_detection: bool = True
    enable_reading_order: bool = True
    
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    
    # Bilateral filter parameters  
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75

@dataclass
class LLMParameters:
    """LLM correction parameters."""
    enable_llm_correction: bool = True
    provider: str = "ollama"
    model: str = "mistral:latest"
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    max_concurrent_corrections: int = 3
    
    # Confidence thresholds for LLM routing by language
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'en': 0.86,
        'de': 0.85, 
        'fr': 0.84,
        'it': 0.84,
        'tr': 0.83
    })
    
    min_text_length: int = 8

@dataclass
class OutputParameters:
    """Output generation parameters."""
    create_html_overlays: bool = True
    include_csv_metadata: bool = True
    enable_telemetry: bool = True
    
    # Overlay parameters
    overlay_show_confidence: bool = True
    overlay_show_bboxes: bool = True
    overlay_confidence_threshold: float = 0.5

@dataclass
class PipelineProfile:
    """Complete pipeline configuration profile."""
    profile_name: str
    description: str
    
    # Language and OCR settings
    ocr_languages: List[str] = field(default_factory=lambda: ['en', 'de', 'fr', 'it', 'tr'])
    ocr_params: Dict[str, OCRParameters] = field(default_factory=dict)
    default_ocr_params: OCRParameters = field(default_factory=OCRParameters)
    
    # Processing settings
    preprocessing: PreprocessingParameters = field(default_factory=PreprocessingParameters)
    llm: LLMParameters = field(default_factory=LLMParameters)
    output: OutputParameters = field(default_factory=OutputParameters)
    
    # Performance settings
    dpi: int = 300
    max_pages: Optional[int] = None
    enable_gpu: bool = True

class ProfileManager:
    """Manager for pipeline configuration profiles."""
    
    def __init__(self):
        """Initialize profile manager with built-in profiles."""
        self.profiles = {}
        self._create_builtin_profiles()
    
    def _create_builtin_profiles(self):
        """Create built-in fast and quality profiles."""
        # Fast profile - optimized for speed
        fast_profile = PipelineProfile(
            profile_name="fast",
            description="Optimized for speed with reduced accuracy",
            
            # Faster OCR parameters
            default_ocr_params=OCRParameters(
                det_db_thresh=0.5,  # Higher threshold = fewer boxes
                det_db_box_thresh=0.7,  # Higher threshold = fewer boxes
                det_limit_side_len=640,  # Lower resolution
                rec_score_thresh=0.6,  # Higher threshold = skip low confidence
                rec_batch_num=12,  # Larger batches
                gpu_mem=2000,  # Less GPU memory
                use_tensorrt=False
            ),
            
            # Minimal preprocessing
            preprocessing=PreprocessingParameters(
                enable_deskew=False,  # Skip deskewing
                enable_denoise=False,  # Skip denoising
                enable_contrast=True,  # Keep contrast enhancement
                enable_column_detection=False,  # Skip column detection
                enable_footnote_detection=False,  # Skip footnote detection
                enable_reading_order=False  # Skip reading order
            ),
            
            # Disabled LLM correction
            llm=LLMParameters(
                enable_llm_correction=False,
                max_concurrent_corrections=1,
                timeout=10,
                confidence_thresholds={
                    'en': 0.9,  # Very high thresholds
                    'de': 0.9,
                    'fr': 0.9,
                    'it': 0.9,
                    'tr': 0.9
                }
            ),
            
            # Minimal output
            output=OutputParameters(
                create_html_overlays=False,  # Skip overlays
                include_csv_metadata=False,  # Skip metadata
                overlay_show_confidence=False,
                overlay_show_bboxes=False
            ),
            
            dpi=200,  # Lower DPI for speed
            enable_gpu=True
        )
        
        # Quality profile - optimized for accuracy
        quality_profile = PipelineProfile(
            profile_name="quality",
            description="Optimized for maximum accuracy with full preprocessing",
            
            # High-quality OCR parameters
            default_ocr_params=OCRParameters(
                det_db_thresh=0.2,  # Lower threshold = more boxes
                det_db_box_thresh=0.5,  # Lower threshold = more boxes
                det_limit_side_len=1280,  # Higher resolution
                rec_score_thresh=0.4,  # Lower threshold = keep more text
                rec_batch_num=3,  # Smaller batches for accuracy
                gpu_mem=6000,  # More GPU memory
                use_tensorrt=False
            ),
            
            # Full preprocessing
            preprocessing=PreprocessingParameters(
                enable_deskew=True,
                enable_denoise=True,
                enable_contrast=True,
                enable_column_detection=True,
                enable_footnote_detection=True,
                enable_reading_order=True,
                clahe_clip_limit=1.5,  # Gentler contrast
                clahe_tile_grid_size=(16, 16)  # Finer grid
            ),
            
            # Enabled LLM correction with lower thresholds
            llm=LLMParameters(
                enable_llm_correction=True,
                max_concurrent_corrections=3,
                timeout=45,
                confidence_thresholds={
                    'en': 0.86,
                    'de': 0.85,
                    'fr': 0.84,
                    'it': 0.84,
                    'tr': 0.83
                },
                min_text_length=6  # Correct shorter spans
            ),
            
            # Full output
            output=OutputParameters(
                create_html_overlays=True,
                include_csv_metadata=True,
                enable_telemetry=True,
                overlay_show_confidence=True,
                overlay_show_bboxes=True,
                overlay_confidence_threshold=0.3  # Show more boxes
            ),
            
            dpi=300,  # Standard DPI
            enable_gpu=True
        )
        
        # Language-specific OCR parameters for quality profile
        quality_profile.ocr_params = {
            'en': OCRParameters(
                det_db_thresh=0.3,
                det_db_box_thresh=0.6,
                rec_score_thresh=0.5
            ),
            'tr': OCRParameters(
                det_db_thresh=0.2,  # Turkish needs lower thresholds
                det_db_box_thresh=0.5,
                rec_score_thresh=0.4,
                det_db_unclip_ratio=1.8  # More aggressive unclipping
            ),
            'de': OCRParameters(
                det_db_thresh=0.25,
                det_db_box_thresh=0.55,
                rec_score_thresh=0.45
            ),
            'fr': OCRParameters(
                det_db_thresh=0.3,
                det_db_box_thresh=0.6,
                rec_score_thresh=0.5
            ),
            'it': OCRParameters(
                det_db_thresh=0.3,
                det_db_box_thresh=0.6,
                rec_score_thresh=0.5
            )
        }
        
        self.profiles['fast'] = fast_profile
        self.profiles['quality'] = quality_profile
        
        # Add a balanced profile
        balanced_profile = PipelineProfile(
            profile_name="balanced",
            description="Balanced speed and accuracy for general use",
            
            # Moderate OCR parameters
            default_ocr_params=OCRParameters(
                det_db_thresh=0.3,
                det_db_box_thresh=0.6,
                det_limit_side_len=960,
                rec_score_thresh=0.5,
                rec_batch_num=6,
                gpu_mem=4000,
                use_tensorrt=False
            ),
            
            # Selective preprocessing
            preprocessing=PreprocessingParameters(
                enable_deskew=True,
                enable_denoise=False,  # Skip denoise for speed
                enable_contrast=True,
                enable_column_detection=True,
                enable_footnote_detection=True,
                enable_reading_order=True
            ),
            
            # Selective LLM correction
            llm=LLMParameters(
                enable_llm_correction=True,
                max_concurrent_corrections=2,
                timeout=30,
                confidence_thresholds={
                    'en': 0.88,  # Slightly higher thresholds
                    'de': 0.87,
                    'fr': 0.86,
                    'it': 0.86,
                    'tr': 0.85
                },
                min_text_length=10
            ),
            
            # Standard output
            output=OutputParameters(
                create_html_overlays=True,
                include_csv_metadata=True,
                enable_telemetry=True,
                overlay_show_confidence=True,
                overlay_show_bboxes=True
            ),
            
            dpi=300,
            enable_gpu=True
        )
        
        self.profiles['balanced'] = balanced_profile
    
    def get_profile(self, profile_name: str) -> Optional[PipelineProfile]:
        """Get a profile by name.
        
        Args:
            profile_name: Name of the profile ('fast', 'quality', 'balanced')
            
        Returns:
            PipelineProfile or None if not found
        """
        return self.profiles.get(profile_name)
    
    def list_profiles(self) -> List[str]:
        """Get list of available profile names.
        
        Returns:
            List of profile names
        """
        return list(self.profiles.keys())
    
    def get_profile_info(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get profile information summary.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Profile info dictionary or None
        """
        profile = self.get_profile(profile_name)
        if not profile:
            return None
        
        return {
            'name': profile.profile_name,
            'description': profile.description,
            'languages': profile.ocr_languages,
            'llm_enabled': profile.llm.enable_llm_correction,
            'preprocessing_features': {
                'deskew': profile.preprocessing.enable_deskew,
                'denoise': profile.preprocessing.enable_denoise,
                'column_detection': profile.preprocessing.enable_column_detection,
                'reading_order': profile.preprocessing.enable_reading_order
            },
            'output_features': {
                'html_overlays': profile.output.create_html_overlays,
                'csv_metadata': profile.output.include_csv_metadata,
                'telemetry': profile.output.enable_telemetry
            },
            'performance': {
                'dpi': profile.dpi,
                'gpu_enabled': profile.enable_gpu,
                'batch_size': profile.default_ocr_params.rec_batch_num
            }
        }
    
    def save_profile(self, profile: PipelineProfile, filepath: str):
        """Save a profile to JSON file.
        
        Args:
            profile: Profile to save
            filepath: Path to save the profile
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(profile), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved profile '{profile.profile_name}' to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
    
    def load_profile(self, filepath: str) -> Optional[PipelineProfile]:
        """Load a profile from JSON file.
        
        Args:
            filepath: Path to the profile file
            
        Returns:
            Loaded PipelineProfile or None if failed
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert nested dicts back to dataclasses
            if 'default_ocr_params' in data:
                data['default_ocr_params'] = OCRParameters(**data['default_ocr_params'])
            if 'preprocessing' in data:
                data['preprocessing'] = PreprocessingParameters(**data['preprocessing'])
            if 'llm' in data:
                data['llm'] = LLMParameters(**data['llm'])
            if 'output' in data:
                data['output'] = OutputParameters(**data['output'])
            
            # Convert ocr_params dict
            if 'ocr_params' in data:
                ocr_params = {}
                for lang, params in data['ocr_params'].items():
                    ocr_params[lang] = OCRParameters(**params)
                data['ocr_params'] = ocr_params
            
            profile = PipelineProfile(**data)
            logger.info(f"Loaded profile '{profile.profile_name}' from {filepath}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to load profile from {filepath}: {e}")
            return None
    
    def export_profiles_documentation(self, output_path: str):
        """Export profile documentation to markdown.
        
        Args:
            output_path: Path to save the documentation
        """
        try:
            doc = "# OCR Pipeline Profiles\n\n"
            doc += "This document describes the available OCR pipeline profiles and their parameters.\n\n"
            
            for profile_name in ['fast', 'balanced', 'quality']:
                profile = self.get_profile(profile_name)
                if not profile:
                    continue
                
                doc += f"## {profile.profile_name.title()} Profile\n\n"
                doc += f"**Description:** {profile.description}\n\n"
                
                # OCR Parameters
                doc += "### OCR Parameters\n\n"
                doc += "| Parameter | Value | Description |\n"
                doc += "|-----------|-------|-------------|\n"
                ocr = profile.default_ocr_params
                doc += f"| det_db_thresh | {ocr.det_db_thresh} | Detection threshold (lower = more text boxes) |\n"
                doc += f"| det_db_box_thresh | {ocr.det_db_box_thresh} | Box threshold (lower = more boxes) |\n"
                doc += f"| rec_score_thresh | {ocr.rec_score_thresh} | Recognition threshold (lower = keep more text) |\n"
                doc += f"| rec_batch_num | {ocr.rec_batch_num} | Batch size (higher = faster, more memory) |\n"
                doc += f"| det_limit_side_len | {ocr.det_limit_side_len} | Max image side length (higher = more detail) |\n"
                doc += f"| gpu_mem | {ocr.gpu_mem} | GPU memory allocation (MB) |\n\n"
                
                # Preprocessing
                doc += "### Preprocessing Features\n\n"
                doc += "| Feature | Enabled | Description |\n"
                doc += "|---------|---------|-------------|\n"
                prep = profile.preprocessing
                doc += f"| Deskew | {prep.enable_deskew} | Correct image rotation |\n"
                doc += f"| Denoise | {prep.enable_denoise} | Reduce image noise |\n"
                doc += f"| Contrast | {prep.enable_contrast} | Enhance text contrast |\n"
                doc += f"| Column Detection | {prep.enable_column_detection} | Detect multi-column layouts |\n"
                doc += f"| Footnote Detection | {prep.enable_footnote_detection} | Separate footnotes |\n"
                doc += f"| Reading Order | {prep.enable_reading_order} | Calculate text reading order |\n\n"
                
                # LLM Settings
                doc += "### LLM Correction\n\n"
                llm = profile.llm
                doc += f"**Enabled:** {llm.enable_llm_correction}\n\n"
                if llm.enable_llm_correction:
                    doc += f"**Max Concurrent:** {llm.max_concurrent_corrections}\n\n"
                    doc += f"**Timeout:** {llm.timeout}s\n\n"
                    doc += "**Confidence Thresholds by Language:**\n\n"
                    for lang, thresh in llm.confidence_thresholds.items():
                        doc += f"- {lang}: {thresh}\n"
                doc += "\n"
                
                # Performance
                doc += "### Performance Settings\n\n"
                doc += f"- **DPI:** {profile.dpi}\n"
                doc += f"- **GPU Enabled:** {profile.enable_gpu}\n"
                doc += f"- **HTML Overlays:** {profile.output.create_html_overlays}\n"
                doc += f"- **Telemetry:** {profile.output.enable_telemetry}\n\n"
                
                doc += "---\n\n"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc)
            
            logger.info(f"Exported profile documentation to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export documentation: {e}")

# Global profile manager instance
profile_manager = ProfileManager()