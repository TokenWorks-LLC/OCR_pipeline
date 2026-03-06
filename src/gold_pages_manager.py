#!/usr/bin/env python3
"""
Gold Pages Manager - Ground Truth Integration Framework
Manages Gold Pages (ground truth) data for OCR accuracy evaluation and LLM correction reference.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GoldPageEntry:
    """Single Gold Page entry with Akkadian-translation pair."""
    document_id: str
    page_number: int
    akkadian_text: str
    translation_text: str
    confidence_score: float
    verified_by: str
    created_date: str
    notes: Optional[str] = None

@dataclass
class GoldPageMetrics:
    """Metrics for Gold Page accuracy evaluation."""
    total_entries: int
    verified_entries: int
    average_confidence: float
    document_coverage: Dict[str, int]  # document_id -> page_count
    language_distribution: Dict[str, int]  # language -> count

class GoldPagesManager:
    """Manages Gold Pages (ground truth) data for OCR evaluation and LLM correction."""
    
    def __init__(self, gold_pages_dir: str = "./data/gold_pages"):
        """
        Initialize Gold Pages Manager.
        
        Args:
            gold_pages_dir: Directory to store Gold Pages data
        """
        self.gold_pages_dir = Path(gold_pages_dir)
        self.gold_pages_dir.mkdir(parents=True, exist_ok=True)
        
        # Gold Pages data structure
        self.entries: List[GoldPageEntry] = []
        self.entries_by_document: Dict[str, List[GoldPageEntry]] = {}
        self.entries_by_page: Dict[Tuple[str, int], GoldPageEntry] = {}
        
        # Load existing Gold Pages
        self._load_gold_pages()
    
    def _load_gold_pages(self) -> None:
        """Load existing Gold Pages from storage."""
        gold_pages_file = self.gold_pages_dir / "gold_pages.json"
        
        if not gold_pages_file.exists():
            logger.info("No existing Gold Pages found, starting fresh")
            return
        
        try:
            with open(gold_pages_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load entries
            for entry_data in data.get('entries', []):
                entry = GoldPageEntry(**entry_data)
                self.entries.append(entry)
                
                # Index by document
                if entry.document_id not in self.entries_by_document:
                    self.entries_by_document[entry.document_id] = []
                self.entries_by_document[entry.document_id].append(entry)
                
                # Index by page
                self.entries_by_page[(entry.document_id, entry.page_number)] = entry
            
            logger.info(f"Loaded {len(self.entries)} Gold Pages entries")
            
        except Exception as e:
            logger.error(f"Failed to load Gold Pages: {e}")
            self.entries = []
    
    def save_gold_pages(self) -> None:
        """Save Gold Pages to storage."""
        gold_pages_file = self.gold_pages_dir / "gold_pages.json"
        
        try:
            data = {
                'metadata': {
                    'total_entries': len(self.entries),
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.0'
                },
                'entries': [
                    {
                        'document_id': entry.document_id,
                        'page_number': entry.page_number,
                        'akkadian_text': entry.akkadian_text,
                        'translation_text': entry.translation_text,
                        'confidence_score': entry.confidence_score,
                        'verified_by': entry.verified_by,
                        'created_date': entry.created_date,
                        'notes': entry.notes
                    }
                    for entry in self.entries
                ]
            }
            
            with open(gold_pages_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.entries)} Gold Pages entries")
            
        except Exception as e:
            logger.error(f"Failed to save Gold Pages: {e}")
    
    def add_gold_page(self, document_id: str, page_number: int, 
                     akkadian_text: str, translation_text: str,
                     confidence_score: float = 1.0, verified_by: str = "manual",
                     notes: Optional[str] = None) -> bool:
        """
        Add a new Gold Page entry.
        
        Args:
            document_id: Document identifier
            page_number: Page number
            akkadian_text: Akkadian text (ground truth)
            translation_text: Translation text (ground truth)
            confidence_score: Confidence score (0-1)
            verified_by: Who verified this entry
            notes: Optional notes
        
        Returns:
            True if added successfully, False otherwise
        """
        try:
            # Check if entry already exists
            key = (document_id, page_number)
            if key in self.entries_by_page:
                logger.warning(f"Gold Page entry already exists for {document_id} page {page_number}")
                return False
            
            # Create new entry
            entry = GoldPageEntry(
                document_id=document_id,
                page_number=page_number,
                akkadian_text=akkadian_text,
                translation_text=translation_text,
                confidence_score=confidence_score,
                verified_by=verified_by,
                created_date=datetime.now().isoformat(),
                notes=notes
            )
            
            # Add to collections
            self.entries.append(entry)
            
            if document_id not in self.entries_by_document:
                self.entries_by_document[document_id] = []
            self.entries_by_document[document_id].append(entry)
            
            self.entries_by_page[key] = entry
            
            # Save to storage
            self.save_gold_pages()
            
            logger.info(f"Added Gold Page entry: {document_id} page {page_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add Gold Page entry: {e}")
            return False
    
    def get_gold_page(self, document_id: str, page_number: int) -> Optional[GoldPageEntry]:
        """Get Gold Page entry for specific document and page."""
        return self.entries_by_page.get((document_id, page_number))
    
    def get_document_gold_pages(self, document_id: str) -> List[GoldPageEntry]:
        """Get all Gold Page entries for a document."""
        return self.entries_by_document.get(document_id, [])
    
    def get_all_gold_pages(self) -> List[GoldPageEntry]:
        """Get all Gold Page entries."""
        return self.entries.copy()
    
    def get_metrics(self) -> GoldPageMetrics:
        """Get Gold Pages metrics."""
        if not self.entries:
            return GoldPageMetrics(
                total_entries=0,
                verified_entries=0,
                average_confidence=0.0,
                document_coverage={},
                language_distribution={}
            )
        
        # Calculate metrics
        total_entries = len(self.entries)
        verified_entries = len([e for e in self.entries if e.verified_by != "auto"])
        average_confidence = sum(e.confidence_score for e in self.entries) / total_entries
        
        # Document coverage
        document_coverage = {}
        for entry in self.entries:
            if entry.document_id not in document_coverage:
                document_coverage[entry.document_id] = 0
            document_coverage[entry.document_id] += 1
        
        # Language distribution (simplified - could be enhanced)
        language_distribution = {
            'akkadian': total_entries,
            'english': total_entries  # Assuming all have translations
        }
        
        return GoldPageMetrics(
            total_entries=total_entries,
            verified_entries=verified_entries,
            average_confidence=average_confidence,
            document_coverage=document_coverage,
            language_distribution=language_distribution
        )
    
    def find_similar_entries(self, akkadian_text: str, threshold: float = 0.8) -> List[GoldPageEntry]:
        """
        Find similar Gold Page entries based on Akkadian text similarity.
        
        Args:
            akkadian_text: Text to find similar entries for
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of similar entries
        """
        similar_entries = []
        
        for entry in self.entries:
            # Simple similarity check (could be enhanced with fuzzy matching)
            if self._calculate_similarity(akkadian_text, entry.akkadian_text) >= threshold:
                similar_entries.append(entry)
        
        return similar_entries
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (0-1)."""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def export_for_llm(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export Gold Pages data in format suitable for LLM correction.
        
        Args:
            document_id: Optional document ID to filter by
        
        Returns:
            Dictionary with Gold Pages data for LLM reference
        """
        entries_to_export = self.entries
        
        if document_id:
            entries_to_export = self.get_document_gold_pages(document_id)
        
        # Format for LLM reference
        llm_data = {
            'reference_pairs': [],
            'metadata': {
                'total_pairs': len(entries_to_export),
                'export_date': datetime.now().isoformat()
            }
        }
        
        for entry in entries_to_export:
            llm_data['reference_pairs'].append({
                'akkadian': entry.akkadian_text,
                'translation': entry.translation_text,
                'confidence': entry.confidence_score,
                'document': entry.document_id,
                'page': entry.page_number
            })
        
        return llm_data
    
    def validate_gold_pages(self) -> Dict[str, Any]:
        """Validate Gold Pages data integrity."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for duplicates
        seen_pages = set()
        duplicates = []
        
        for entry in self.entries:
            key = (entry.document_id, entry.page_number)
            if key in seen_pages:
                duplicates.append(f"{entry.document_id} page {entry.page_number}")
            else:
                seen_pages.add(key)
        
        if duplicates:
            validation_results['warnings'].append(f"Duplicate entries found: {duplicates}")
        
        # Check for empty entries
        empty_entries = []
        for entry in self.entries:
            if not entry.akkadian_text.strip() or not entry.translation_text.strip():
                empty_entries.append(f"{entry.document_id} page {entry.page_number}")
        
        if empty_entries:
            validation_results['errors'].append(f"Empty entries found: {empty_entries}")
            validation_results['is_valid'] = False
        
        # Statistics
        validation_results['statistics'] = {
            'total_entries': len(self.entries),
            'unique_documents': len(self.entries_by_document),
            'duplicate_entries': len(duplicates),
            'empty_entries': len(empty_entries)
        }
        
        return validation_results

# Convenience functions for easy integration
def create_gold_pages_manager(gold_pages_dir: str = "./data/gold_pages") -> GoldPagesManager:
    """Create a new Gold Pages Manager instance."""
    return GoldPagesManager(gold_pages_dir)

def load_gold_pages_for_document(document_id: str, gold_pages_dir: str = "./data/gold_pages") -> List[GoldPageEntry]:
    """Load Gold Pages for a specific document."""
    manager = GoldPagesManager(gold_pages_dir)
    return manager.get_document_gold_pages(document_id)

def export_gold_pages_for_llm(document_id: Optional[str] = None, 
                             gold_pages_dir: str = "./data/gold_pages") -> Dict[str, Any]:
    """Export Gold Pages for LLM correction."""
    manager = GoldPagesManager(gold_pages_dir)
    return manager.export_for_llm(document_id)
