"""
OCR Ensemble System using ROVER (Recognizer Output Voting Error Reduction)

This module implements ensemble voting to combine results from multiple OCR engines.
Uses confidence-weighted voting and bounding box alignment to select the best text.
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextBox:
    """Represents a detected text box from an OCR engine."""
    bbox: Any  # Can be list of points or [x, y, w, h]
    text: str
    confidence: float
    engine: str
    
    def get_center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        if isinstance(self.bbox, (list, tuple)) and len(self.bbox) == 4:
            # Check if it's [x, y, w, h] or [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if isinstance(self.bbox[0], (list, tuple)):
                # Polygon format - calculate centroid
                xs = [p[0] for p in self.bbox]
                ys = [p[1] for p in self.bbox]
                return (sum(xs) / len(xs), sum(ys) / len(ys))
            else:
                # [x, y, w, h] format
                return (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2)
        elif isinstance(self.bbox, np.ndarray):
            # Numpy array - assume polygon
            if self.bbox.shape == (4, 2):
                return (float(np.mean(self.bbox[:, 0])), float(np.mean(self.bbox[:, 1])))
        
        # Fallback
        return (0, 0)
    
    def overlaps(self, other: 'TextBox', threshold: float = 0.5) -> bool:
        """Check if this box overlaps with another box."""
        c1 = self.get_center()
        c2 = other.get_center()
        
        # Simple distance-based overlap detection
        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        
        # Estimate box size
        if isinstance(self.bbox, (list, tuple)) and len(self.bbox) == 4:
            if isinstance(self.bbox[0], (list, tuple)):
                xs = [p[0] for p in self.bbox]
                ys = [p[1] for p in self.bbox]
                size = max(max(xs) - min(xs), max(ys) - min(ys))
            else:
                size = max(self.bbox[2], self.bbox[3])
        else:
            size = 100  # Default
        
        return distance < (size * threshold)


class OCREnsemble:
    """
    Ensemble voting system for combining multiple OCR engine results.
    
    Implements confidence-weighted voting with spatial alignment.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ensemble system.
        
        Args:
            config: Configuration with options:
                - voting_method: 'weighted', 'majority', or 'best' (default: 'weighted')
                - overlap_threshold: IoU threshold for box matching (default: 0.5)
                - min_confidence: Minimum confidence to consider (default: 0.3)
                - engine_weights: Dict of engine weights (default: equal)
        """
        self.config = config or {}
        self.voting_method = self.config.get('voting_method', 'weighted')
        self.overlap_threshold = self.config.get('overlap_threshold', 0.5)
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.engine_weights = self.config.get('engine_weights', {})
    
    def combine_results(self, engine_results: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Combine results from multiple OCR engines using ensemble voting.
        
        Args:
            engine_results: Dict mapping engine name to list of detection dicts
                           Each detection: {'bbox': ..., 'text': ..., 'confidence': ...}
        
        Returns:
            List of combined detection dicts with highest confidence text
        """
        if not engine_results:
            return []
        
        # Convert all results to TextBox objects
        all_boxes = []
        for engine_name, detections in engine_results.items():
            weight = self.engine_weights.get(engine_name, 1.0)
            for det in detections:
                if det.get('confidence', 0) >= self.min_confidence:
                    all_boxes.append(TextBox(
                        bbox=det['bbox'],
                        text=det['text'],
                        confidence=det['confidence'] * weight,
                        engine=engine_name
                    ))
        
        if not all_boxes:
            return []
        
        # Group overlapping boxes
        groups = self._group_overlapping_boxes(all_boxes)
        
        # Vote within each group
        combined_results = []
        for group in groups:
            best_box = self._vote_best_box(group)
            if best_box:
                combined_results.append({
                    'bbox': best_box.bbox,
                    'text': best_box.text,
                    'confidence': best_box.confidence,
                    'engine': best_box.engine,
                    'votes': len(group)
                })
        
        logger.info(f"Ensemble combined {len(all_boxes)} boxes from {len(engine_results)} engines into {len(combined_results)} results")
        
        return combined_results
    
    def _group_overlapping_boxes(self, boxes: List[TextBox]) -> List[List[TextBox]]:
        """Group boxes that overlap into clusters."""
        groups = []
        used = set()
        
        for i, box in enumerate(boxes):
            if i in used:
                continue
            
            # Start a new group
            group = [box]
            used.add(i)
            
            # Find all overlapping boxes
            for j, other_box in enumerate(boxes):
                if j in used:
                    continue
                
                # Check if any box in the group overlaps with this box
                for group_box in group:
                    if group_box.overlaps(other_box, self.overlap_threshold):
                        group.append(other_box)
                        used.add(j)
                        break
            
            groups.append(group)
        
        return groups
    
    def _vote_best_box(self, boxes: List[TextBox]) -> TextBox:
        """
        Vote for the best text from a group of overlapping boxes.
        
        Uses confidence-weighted voting or simple best-confidence selection.
        """
        if not boxes:
            return None
        
        if self.voting_method == 'best':
            # Simply return the box with highest confidence
            return max(boxes, key=lambda b: b.confidence)
        
        elif self.voting_method == 'majority':
            # Majority vote by text, weighted by confidence
            text_votes = defaultdict(float)
            text_boxes = {}
            
            for box in boxes:
                text_votes[box.text] += box.confidence
                if box.text not in text_boxes or box.confidence > text_boxes[box.text].confidence:
                    text_boxes[box.text] = box
            
            best_text = max(text_votes.items(), key=lambda x: x[1])[0]
            return text_boxes[best_text]
        
        elif self.voting_method == 'weighted':
            # Weighted confidence voting
            # If texts are similar, combine confidence; otherwise take highest
            
            # Group by similar text (simple exact match for now)
            text_groups = defaultdict(list)
            for box in boxes:
                text_groups[box.text].append(box)
            
            # Find the text group with highest combined confidence
            best_group = max(text_groups.values(), 
                           key=lambda g: sum(b.confidence for b in g))
            
            # Return the box with highest individual confidence from that group
            return max(best_group, key=lambda b: b.confidence)
        
        # Fallback
        return max(boxes, key=lambda b: b.confidence)
    
    def get_statistics(self, engine_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Get statistics about the ensemble process."""
        stats = {
            'num_engines': len(engine_results),
            'total_detections': sum(len(dets) for dets in engine_results.values()),
            'per_engine_detections': {eng: len(dets) for eng, dets in engine_results.items()},
            'voting_method': self.voting_method
        }
        return stats
