#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) for OCR.

Decodes page at multiple rotations and scales, then fuses results via ROVER.

REQUIREMENTS:
- Rotation augmentation: {-2, -1, 0, 1, 2} degrees
- Scale augmentation: {0.95, 1.0, 1.05}
- Per-page time budget: 8 seconds
- ROVER fusion of augmented results
- Abort if budget exceeded

Author: Senior OCR Engineer
Date: 2025-10-07
"""

import logging
import time
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import cv2


@dataclass
class TTAConfig:
    """Configuration for test-time augmentation."""
    rot_deg: List[float] = None  # Rotation angles in degrees
    scales: List[float] = None  # Scale factors
    time_budget: float = 8.0  # Per-page time budget in seconds
    max_augments: int = 5  # Maximum augmented decodes
    
    def __post_init__(self):
        if self.rot_deg is None:
            self.rot_deg = [-2, -1, 0, 1, 2]
        if self.scales is None:
            self.scales = [0.95, 1.0, 1.05]


@dataclass
class AugmentedResult:
    """Result from one augmented decode."""
    text: str
    confidence: float
    rotation: float
    scale: float
    decode_time: float
    engine: str


class TTAAugmenter:
    """
    Test-time augmentation for OCR.
    
    Applies geometric transformations and fuses results.
    """
    
    def __init__(self, config: Optional[TTAConfig] = None):
        """
        Initialize TTA augmenter.
        
        Args:
            config: TTA configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or TTAConfig()
        
        # Calculate total augmentations
        self.total_augments = len(self.config.rot_deg) * len(self.config.scales)
        
        # Cap at max_augments
        if self.total_augments > self.config.max_augments:
            self.logger.warning(f"Total augments ({self.total_augments}) exceeds max ({self.config.max_augments})")
            self.logger.warning("Will prioritize central augments (rot=0, scale=1.0)")
            self.total_augments = self.config.max_augments
        
        self.logger.info(f"TTA initialized: {self.total_augments} augments, {self.config.time_budget}s budget")
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by angle (degrees).
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counterclockwise)
            
        Returns:
            Rotated image
        """
        if abs(angle) < 0.01:
            # No rotation needed
            return image
        
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate with white background
        rotated = cv2.warpAffine(
            image, M, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return rotated
    
    def scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """
        Scale image by factor.
        
        Args:
            image: Input image
            scale: Scale factor (1.0 = no change, >1 = larger, <1 = smaller)
            
        Returns:
            Scaled image
        """
        if abs(scale - 1.0) < 0.01:
            # No scaling needed
            return image
        
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        scaled = cv2.resize(
            image, (new_width, new_height),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        )
        
        # If scaled down, pad to original size
        if scale < 1.0:
            padded = np.full((height, width, *image.shape[2:]), 255, dtype=image.dtype)
            y_offset = (height - new_height) // 2
            x_offset = (width - new_width) // 2
            padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled
            return padded
        
        # If scaled up, crop to original size
        elif scale > 1.0:
            y_offset = (new_height - height) // 2
            x_offset = (new_width - width) // 2
            cropped = scaled[y_offset:y_offset+height, x_offset:x_offset+width]
            return cropped
        
        return scaled
    
    def generate_augmentations(self, image: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """
        Generate augmented versions of image.
        
        Args:
            image: Input image
            
        Returns:
            List of (augmented_image, rotation, scale) tuples
        """
        augmentations = []
        
        # Generate all combinations
        for rotation in self.config.rot_deg:
            for scale in self.config.scales:
                # Apply transformations
                aug_image = image.copy()
                
                # Scale first, then rotate (order matters)
                aug_image = self.scale_image(aug_image, scale)
                aug_image = self.rotate_image(aug_image, rotation)
                
                augmentations.append((aug_image, rotation, scale))
        
        # Prioritize central augment (rot=0, scale=1.0)
        # Move it to front if not already there
        central_idx = None
        for i, (_, rot, sc) in enumerate(augmentations):
            if abs(rot) < 0.01 and abs(sc - 1.0) < 0.01:
                central_idx = i
                break
        
        if central_idx is not None and central_idx > 0:
            # Move central to front
            central = augmentations.pop(central_idx)
            augmentations.insert(0, central)
        
        # Limit to max_augments
        augmentations = augmentations[:self.config.max_augments]
        
        return augmentations
    
    def decode_with_tta(self, image: np.ndarray, 
                       decode_fn: Callable[[np.ndarray], Tuple[str, float]],
                       engine_name: str = 'unknown') -> List[AugmentedResult]:
        """
        Decode image with TTA.
        
        Args:
            image: Input image
            decode_fn: Decoding function (image -> (text, confidence))
            engine_name: Name of OCR engine
            
        Returns:
            List of augmented results
        """
        start_time = time.time()
        results = []
        
        # Generate augmentations
        augmentations = self.generate_augmentations(image)
        
        self.logger.debug(f"Generated {len(augmentations)} augmentations")
        
        # Decode each augmentation
        for aug_image, rotation, scale in augmentations:
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > self.config.time_budget:
                self.logger.warning(f"TTA time budget ({self.config.time_budget}s) exceeded")
                self.logger.warning(f"Completed {len(results)}/{len(augmentations)} augments")
                break
            
            # Decode
            decode_start = time.time()
            
            try:
                text, confidence = decode_fn(aug_image)
                decode_time = time.time() - decode_start
                
                result = AugmentedResult(
                    text=text,
                    confidence=confidence,
                    rotation=rotation,
                    scale=scale,
                    decode_time=decode_time,
                    engine=engine_name
                )
                
                results.append(result)
                
                self.logger.debug(f"  Aug rot={rotation:.1f}°, scale={scale:.2f}: "
                                f"'{text[:30]}...' (conf={confidence:.3f}, {decode_time:.2f}s)")
            
            except Exception as e:
                self.logger.error(f"TTA decode failed for rot={rotation}, scale={scale}: {e}")
                continue
        
        total_time = time.time() - start_time
        self.logger.info(f"TTA completed: {len(results)} augments in {total_time:.2f}s")
        
        return results
    
    def fuse_with_rover(self, results: List[AugmentedResult], 
                       rover_fusion) -> Tuple[str, float]:
        """
        Fuse TTA results using ROVER.
        
        Args:
            results: List of augmented results
            rover_fusion: ROVER fusion instance
            
        Returns:
            Tuple of (consensus_text, confidence)
        """
        if not results:
            return '', 0.0
        
        if len(results) == 1:
            # Single result - no fusion needed
            return results[0].text, results[0].confidence
        
        # Convert to ROVER hypotheses
        from rover_fusion import Hypothesis
        
        hypotheses = []
        for i, result in enumerate(results):
            # Create unique engine name for each augmentation
            aug_engine = f"{result.engine}_rot{result.rotation:.1f}_sc{result.scale:.2f}"
            
            hyp = Hypothesis(
                text=result.text,
                confidence=result.confidence,
                engine=aug_engine
            )
            
            hypotheses.append(hyp)
        
        # Fuse with ROVER
        consensus, confidence, _ = rover_fusion.fuse(hypotheses)
        
        self.logger.info(f"ROVER fusion: '{consensus[:50]}...' (conf={confidence:.4f})")
        
        return consensus, confidence


if __name__ == '__main__':
    # Test TTA augmenter
    print("=== TTA Augmenter Test ===\n")
    
    # Create test image (white background with black text)
    test_image = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Test Text", (50, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Create TTA config
    config = TTAConfig(
        rot_deg=[-1, 0, 1],
        scales=[0.95, 1.0, 1.05],
        time_budget=5.0,
        max_augments=5
    )
    
    # Create augmenter
    augmenter = TTAAugmenter(config)
    
    # Generate augmentations
    print("[1] Generating augmentations...")
    augmentations = augmenter.generate_augmentations(test_image)
    
    print(f"  Generated {len(augmentations)} augmentations:")
    for i, (_, rot, scale) in enumerate(augmentations):
        print(f"    {i+1}. rot={rot:+.1f}°, scale={scale:.2f}")
    
    # Mock decode function
    def mock_decode(image):
        """Mock decoder that returns dummy results."""
        import time
        time.sleep(0.1)  # Simulate decode time
        
        # Slightly different results based on image hash
        image_hash = hash(image.tobytes()) % 3
        
        texts = [
            "Test Text",
            "Test Texl",  # Small error
            "Test Text",
        ]
        
        confidences = [0.92, 0.88, 0.90]
        
        return texts[image_hash], confidences[image_hash]
    
    # Decode with TTA
    print("\n[2] Decoding with TTA...")
    results = augmenter.decode_with_tta(test_image, mock_decode, engine_name='mock_ocr')
    
    print(f"  Completed {len(results)} decodes:")
    for result in results:
        print(f"    rot={result.rotation:+.1f}°, scale={result.scale:.2f}: "
              f"'{result.text}' (conf={result.confidence:.3f})")
    
    print("\n=== Test Complete ===")
