#!/usr/bin/env python3
"""
Layout band classifier for header/footer detection.

Uses LogisticRegression with weak labeling from band positions.
Auto-fallback to rule-based filter if classifier underperforms.

REQUIREMENTS:
- LogisticRegression for binary classification (header/footer vs content)
- Weak labeling: top 6% & bottom 8% = negative, mid bands = positive
- Auto-fallback if rules perform better on dev slice
- Gated by enable_header_footer_lr flag

Author: Senior OCR Engineer
Date: 2025-10-07
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class BoundingBox:
    """Bounding box for text line."""
    x: float
    y: float
    width: float
    height: float


class LayoutBandClassifier:
    """
    Machine learning classifier for header/footer detection.
    
    Falls back to rule-based approach if ML underperforms.
    """
    
    def __init__(self, header_band_pct: float = 0.06, 
                 footer_band_pct: float = 0.08,
                 use_ml: bool = True):
        """
        Initialize layout classifier.
        
        Args:
            header_band_pct: Top band percentage for headers (default: 6%)
            footer_band_pct: Bottom band percentage for footers (default: 8%)
            use_ml: Whether to use ML classifier (default: True)
        """
        self.logger = logging.getLogger(__name__)
        
        self.header_band_pct = header_band_pct
        self.footer_band_pct = footer_band_pct
        self.use_ml = use_ml and SKLEARN_AVAILABLE
        
        if use_ml and not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available - falling back to rule-based")
            self.use_ml = False
        
        self.classifier = None
        self.rule_accuracy = None
        self.ml_accuracy = None
        
        if self.use_ml:
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        
        self.logger.info(f"Layout classifier initialized (ML={'enabled' if self.use_ml else 'disabled'})")
    
    def extract_features(self, bbox: BoundingBox, page_height: float) -> np.ndarray:
        """
        Extract features from bounding box for classification.
        
        Features:
        - Normalized y position (0=top, 1=bottom)
        - Normalized height
        - Aspect ratio
        - Distance from top edge
        - Distance from bottom edge
        
        Args:
            bbox: Bounding box
            page_height: Page height for normalization
            
        Returns:
            Feature vector
        """
        # Normalized y position (vertical center)
        y_center = bbox.y + bbox.height / 2
        norm_y = y_center / page_height if page_height > 0 else 0.5
        
        # Normalized height
        norm_h = bbox.height / page_height if page_height > 0 else 0.1
        
        # Aspect ratio
        aspect = bbox.width / bbox.height if bbox.height > 0 else 1.0
        
        # Distance from edges
        dist_top = bbox.y / page_height if page_height > 0 else 0.0
        dist_bottom = (page_height - (bbox.y + bbox.height)) / page_height if page_height > 0 else 0.0
        
        return np.array([norm_y, norm_h, aspect, dist_top, dist_bottom])
    
    def rule_based_prediction(self, bbox: BoundingBox, page_height: float) -> bool:
        """
        Rule-based prediction (header/footer = True, content = False).
        
        Args:
            bbox: Bounding box
            page_height: Page height
            
        Returns:
            True if header/footer, False if content
        """
        y_center = bbox.y + bbox.height / 2
        norm_y = y_center / page_height if page_height > 0 else 0.5
        
        # Check if in header band (top X%)
        if norm_y < self.header_band_pct:
            return True
        
        # Check if in footer band (bottom X%)
        if norm_y > (1.0 - self.footer_band_pct):
            return True
        
        return False
    
    def train(self, boxes: List[BoundingBox], page_heights: List[float],
             validation_split: float = 0.2) -> Tuple[float, float]:
        """
        Train ML classifier with weak labels.
        
        Weak labeling strategy:
        - Top 6% & bottom 8% bands → negative (header/footer)
        - Middle bands → positive (content)
        
        Args:
            boxes: List of bounding boxes
            page_heights: Corresponding page heights
            validation_split: Validation split ratio
            
        Returns:
            Tuple of (rule_accuracy, ml_accuracy) on validation set
        """
        if not self.use_ml:
            self.logger.warning("ML not available - skipping training")
            return 0.0, 0.0
        
        # Extract features and weak labels
        X = []
        y_weak = []
        
        for bbox, page_h in zip(boxes, page_heights):
            features = self.extract_features(bbox, page_h)
            X.append(features)
            
            # Weak label (1 = content, 0 = header/footer)
            is_header_footer = self.rule_based_prediction(bbox, page_h)
            y_weak.append(0 if is_header_footer else 1)
        
        X = np.array(X)
        y_weak = np.array(y_weak)
        
        # Split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_weak, test_size=validation_split, random_state=42
        )
        
        # Train classifier
        self.logger.info(f"Training on {len(X_train)} samples...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate both on validation set
        ml_preds = self.classifier.predict(X_val)
        self.ml_accuracy = np.mean(ml_preds == y_val)
        
        # Rule-based predictions on validation set
        rule_preds = []
        for features, page_h in zip(X_val, [page_heights[0]] * len(X_val)):
            # Reconstruct bbox from features (approximate)
            norm_y = features[0]
            bbox_approx = BoundingBox(
                x=0, y=norm_y * page_h, width=100, height=features[1] * page_h
            )
            rule_pred = self.rule_based_prediction(bbox_approx, page_h)
            rule_preds.append(0 if rule_pred else 1)
        
        self.rule_accuracy = np.mean(np.array(rule_preds) == y_val)
        
        self.logger.info(f"Validation accuracy:")
        self.logger.info(f"  Rule-based: {self.rule_accuracy:.4f}")
        self.logger.info(f"  ML: {self.ml_accuracy:.4f}")
        
        # Decide whether to use ML or fallback to rules
        if self.ml_accuracy <= self.rule_accuracy:
            self.logger.warning("ML does not outperform rules - will use fallback")
            self.use_ml = False
        else:
            self.logger.info("ML outperforms rules - using ML classifier")
        
        return self.rule_accuracy, self.ml_accuracy
    
    def predict(self, bbox: BoundingBox, page_height: float) -> bool:
        """
        Predict if bounding box is header/footer.
        
        Args:
            bbox: Bounding box
            page_height: Page height
            
        Returns:
            True if header/footer, False if content
        """
        if not self.use_ml or self.classifier is None:
            # Fallback to rules
            return self.rule_based_prediction(bbox, page_height)
        
        # ML prediction
        features = self.extract_features(bbox, page_height)
        pred = self.classifier.predict(features.reshape(1, -1))[0]
        
        # pred = 1 → content, pred = 0 → header/footer
        return pred == 0


if __name__ == '__main__':
    # Test layout classifier
    print("=== Layout Band Classifier Test ===\n")
    
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not installed")
        print("Run: pip install scikit-learn")
        exit(1)
    
    # Create classifier
    classifier = LayoutBandClassifier(
        header_band_pct=0.06,
        footer_band_pct=0.08,
        use_ml=True
    )
    
    # Generate synthetic training data
    print("[1] Generating synthetic training data...")
    
    page_height = 1000.0
    boxes = []
    page_heights = []
    
    # Headers (top 6%)
    for i in range(20):
        boxes.append(BoundingBox(x=50, y=10 + i*2, width=500, height=15))
        page_heights.append(page_height)
    
    # Content (middle)
    for i in range(100):
        y = 100 + i * 7
        boxes.append(BoundingBox(x=50, y=y, width=500, height=12))
        page_heights.append(page_height)
    
    # Footers (bottom 8%)
    for i in range(20):
        y = 940 + i * 2
        boxes.append(BoundingBox(x=50, y=y, width=500, height=15))
        page_heights.append(page_height)
    
    print(f"  Generated {len(boxes)} boxes")
    
    # Train classifier
    print("\n[2] Training classifier...")
    rule_acc, ml_acc = classifier.train(boxes, page_heights, validation_split=0.3)
    
    print(f"\n  Rule accuracy: {rule_acc:.4f}")
    print(f"  ML accuracy: {ml_acc:.4f}")
    
    # Test predictions
    print("\n[3] Testing predictions...")
    
    test_boxes = [
        BoundingBox(x=50, y=20, width=500, height=15),  # Header
        BoundingBox(x=50, y=500, width=500, height=12),  # Content
        BoundingBox(x=50, y=950, width=500, height=15),  # Footer
    ]
    
    for i, bbox in enumerate(test_boxes):
        pred = classifier.predict(bbox, page_height)
        print(f"  Box {i+1} (y={bbox.y}): {'Header/Footer' if pred else 'Content'}")
    
    print("\n=== Test Complete ===")
