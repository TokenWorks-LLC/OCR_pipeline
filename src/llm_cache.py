"""
LLM cache implementation for OCR correction results.
Provides persistent caching with SHA256 keys and configurable storage.
"""
import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cached LLM correction result."""
    corrected_text: str
    applied_edits: List[Dict[str, Any]]
    latency_ms: int
    model_id: str
    prompt_version: str
    created_at: str
    notes: str = ""

class LLMCache:
    """Persistent cache for LLM correction results using SQLite."""
    
    CURRENT_PROMPT_VERSION = "fix-typos-only-v1"
    
    def __init__(self, cache_dir: str = "data/.cache/llm"):
        """Initialize LLM cache.
        
        Args:
            cache_dir: Directory to store cache database
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "llm_cache.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with cache schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        cache_key TEXT PRIMARY KEY,
                        corrected_text TEXT NOT NULL,
                        applied_edits TEXT NOT NULL,
                        latency_ms INTEGER NOT NULL,
                        model_id TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        notes TEXT DEFAULT ''
                    )
                """)
                
                # Create index for faster lookups by prompt version
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prompt_version 
                    ON cache_entries(prompt_version)
                """)
                
                conn.commit()
                logger.debug(f"Initialized LLM cache database at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize cache database: {e}")
    
    def _generate_cache_key(self, model_id: str, language: str, 
                           original_text: str, constraints: Dict[str, Any]) -> str:
        """Generate SHA256 cache key for the input parameters.
        
        Args:
            model_id: LLM model identifier
            language: Language ISO code
            original_text: Original text to be corrected
            constraints: Correction constraints
            
        Returns:
            SHA256 hash as cache key
        """
        # Normalize text for consistent caching
        normalized_text = original_text.strip().lower()
        
        # Create cache key components
        key_components = {
            'model_id': model_id,
            'prompt_version': self.CURRENT_PROMPT_VERSION,
            'language': language,
            'text': normalized_text,
            'constraints': sorted(constraints.items()) if constraints else []
        }
        
        # Generate SHA256 hash
        key_str = json.dumps(key_components, sort_keys=True, separators=(',', ':'))
        cache_key = hashlib.sha256(key_str.encode('utf-8')).hexdigest()
        
        return cache_key
    
    def get(self, model_id: str, language: str, original_text: str, 
            constraints: Dict[str, Any] = None) -> Optional[CacheEntry]:
        """Retrieve cached correction result.
        
        Args:
            model_id: LLM model identifier
            language: Language ISO code
            original_text: Original text to be corrected
            constraints: Correction constraints
            
        Returns:
            CacheEntry if found, None otherwise
        """
        cache_key = self._generate_cache_key(model_id, language, original_text, constraints or {})
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT corrected_text, applied_edits, latency_ms, model_id, 
                           prompt_version, created_at, notes
                    FROM cache_entries 
                    WHERE cache_key = ?
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row:
                    applied_edits = json.loads(row[1]) if row[1] else []
                    entry = CacheEntry(
                        corrected_text=row[0],
                        applied_edits=applied_edits,
                        latency_ms=row[2],
                        model_id=row[3],
                        prompt_version=row[4],
                        created_at=row[5],
                        notes=row[6] or ""
                    )
                    logger.debug(f"Cache hit for key {cache_key[:8]}...")
                    return entry
                    
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def put(self, model_id: str, language: str, original_text: str,
            corrected_text: str, applied_edits: List[Dict[str, Any]],
            latency_ms: int, constraints: Dict[str, Any] = None,
            notes: str = "") -> bool:
        """Store correction result in cache.
        
        Args:
            model_id: LLM model identifier
            language: Language ISO code
            original_text: Original text that was corrected
            corrected_text: LLM-corrected text
            applied_edits: List of edit operations applied
            latency_ms: Correction latency in milliseconds
            constraints: Correction constraints used
            notes: Optional notes about the correction
            
        Returns:
            True if stored successfully, False otherwise
        """
        cache_key = self._generate_cache_key(model_id, language, original_text, constraints or {})
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, corrected_text, applied_edits, latency_ms, 
                     model_id, prompt_version, created_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    corrected_text,
                    json.dumps(applied_edits, separators=(',', ':')),
                    latency_ms,
                    model_id,
                    self.CURRENT_PROMPT_VERSION,
                    datetime.now().isoformat(),
                    notes
                ))
                conn.commit()
                logger.debug(f"Cached result for key {cache_key[:8]}...")
                return True
                
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total entries
                total_cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                total_count = total_cursor.fetchone()[0]
                
                # Entries by prompt version
                version_cursor = conn.execute("""
                    SELECT prompt_version, COUNT(*) 
                    FROM cache_entries 
                    GROUP BY prompt_version
                """)
                version_counts = dict(version_cursor.fetchall())
                
                # Entries by model
                model_cursor = conn.execute("""
                    SELECT model_id, COUNT(*) 
                    FROM cache_entries 
                    GROUP BY model_id
                """)
                model_counts = dict(model_cursor.fetchall())
                
                # Average latency
                latency_cursor = conn.execute("SELECT AVG(latency_ms) FROM cache_entries")
                avg_latency = latency_cursor.fetchone()[0] or 0
                
                return {
                    'total_entries': total_count,
                    'current_version_entries': version_counts.get(self.CURRENT_PROMPT_VERSION, 0),
                    'version_counts': version_counts,
                    'model_counts': model_counts,
                    'average_latency_ms': round(avg_latency, 2),
                    'cache_size_mb': round(self.db_path.stat().st_size / (1024 * 1024), 2)
                }
                
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}
    
    def purge_by_version(self, prompt_version: str) -> int:
        """Purge cache entries by prompt version.
        
        Args:
            prompt_version: Prompt version to purge
            
        Returns:
            Number of entries purged
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE prompt_version = ?
                """, (prompt_version,))
                purged_count = cursor.rowcount
                conn.commit()
                logger.info(f"Purged {purged_count} cache entries for version {prompt_version}")
                return purged_count
                
        except Exception as e:
            logger.error(f"Cache purge failed: {e}")
            return 0
    
    def purge_old_entries(self, days_old: int = 30) -> int:
        """Purge cache entries older than specified days.
        
        Args:
            days_old: Delete entries older than this many days
            
        Returns:
            Number of entries purged
        """
        try:
            cutoff_date = datetime.now()
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
            cutoff_str = cutoff_date.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE created_at < ?
                """, (cutoff_str,))
                purged_count = cursor.rowcount
                conn.commit()
                logger.info(f"Purged {purged_count} cache entries older than {days_old} days")
                return purged_count
                
        except Exception as e:
            logger.error(f"Old entries purge failed: {e}")
            return 0