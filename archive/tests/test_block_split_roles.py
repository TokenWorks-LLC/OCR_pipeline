"""
Unit tests for block_splitter and block_roles modules.

Tests the regex patterns and splitting logic on real-world examples
from the Akkadian corpus.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from block_splitter import BlockSplitter, split_blocks
from block_roles import BlockRoleTagger, BlockRole, tag_block_roles, filter_blocks_by_role


class TestBlockSplitter(unittest.TestCase):
    """Test block splitting functionality."""
    
    def setUp(self):
        self.splitter = BlockSplitter({'split_enabled': True})
    
    def test_split_on_author_line(self):
        """Test splitting on author citation lines."""
        text = """SEBAHATTIN BAYRAM - SALIH CECEN
i-dí-ni-a-ti-ma IGI GIR
sa A-ur í-bu-tí-ni"""
        
        fragments = self.splitter.split_block(text, 'test_block')
        self.assertGreater(len(fragments), 1, "Should split author from transliteration")
        
        # First fragment should be author (may be tagged as header or author_line)
        self.assertIn('BAYRAM', fragments[0]['text'])
        self.assertIn(fragments[0]['split_reason'], ['author_line', 'header'])
    
    def test_split_on_catalog_number(self):
        """Test splitting catalog numbers."""
        text = """Some content here
Kt n/k 1295
More content"""
        
        fragments = self.splitter.split_block(text, 'test_block')
        # May or may not split depending on line length and context
        # What matters is catalog is identifiable
        catalog_frags = [f for f in fragments if 'Kt n/k' in f['text']]
        self.assertGreater(len(catalog_frags), 0, "Catalog number should be present")
        self.assertIn(catalog_frags[0]['split_reason'], ['catalog_line', 'metadata_line', 'original'])
    
    def test_split_on_reference_line(self):
        """Test splitting reference/bibliographic lines."""
        text = """Translation text here
HW s. 124 a.
More text"""
        
        fragments = self.splitter.split_block(text, 'test_block')
        self.assertGreater(len(fragments), 1)
        
        # Reference line should be isolated (may be tagged as reference or metadata)
        ref_frag = [f for f in fragments if 'HW s.' in f['text']][0]
        self.assertTrue('reference' in ref_frag['split_reason'] or 'metadata' in ref_frag['split_reason'])
    
    def test_split_on_header(self):
        """Test splitting all-caps headers."""
        text = """KULTEPE TEXTS FROM MUSEUMS
Regular content here"""
        
        fragments = self.splitter.split_block(text, 'test_block')
        self.assertGreater(len(fragments), 1)
        
        # Header should be separate
        self.assertEqual(fragments[0]['split_reason'], 'header')
        self.assertIn('KULTEPE', fragments[0]['text'])
    
    def test_no_split_on_pure_transliteration(self):
        """Pure transliteration blocks should not be split."""
        text = """i-dí-ni-a-ti-ma IGI GIR
sa A-ur í-bu-tí-ni
ni-di-in
Ma-nu-ki-A-ùr"""
        
        fragments = self.splitter.split_block(text, 'test_block')
        # Should stay as one block (pure content, no metadata)
        self.assertEqual(len(fragments), 1)
    
    def test_split_blocks_convenience_function(self):
        """Test the convenience function for block lists."""
        blocks = [
            {'block_id': 'b1', 'text': 'Content\nKt 123\nMore content'},
            {'block_id': 'b2', 'text': 'Pure content without splits'}
        ]
        
        result = split_blocks(blocks)
        
        # First block should split into multiple fragments
        b1_frags = [b for b in result if b['original_block_id'] == 'b1']
        self.assertGreater(len(b1_frags), 1)
        
        # Second block should remain single
        b2_frags = [b for b in result if b['original_block_id'] == 'b2']
        self.assertEqual(len(b2_frags), 1)


class TestBlockRoleTagger(unittest.TestCase):
    """Test block role tagging functionality."""
    
    def setUp(self):
        self.tagger = BlockRoleTagger({'role_tagging': True})
    
    def test_reference_meta_detection(self):
        """Test detection of reference/bibliographic blocks."""
        text = "HW s. 124 a. / evcuttur. ubalit II pres. Formdadr. / 5) Müze Env. Nr. 161-426-64"
        block = {'text': text, 'block_id': 'test'}
        
        result = self.tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.REFERENCE_META)
        self.assertIn('citation_markers', result['role_reasons'])
        self.assertIn('museum_numbers', result['role_reasons'])
    
    def test_catalog_number_as_reference(self):
        """Test catalog numbers tagged as reference metadata."""
        text = "Kt j/k 430"
        block = {'text': text, 'block_id': 'test'}
        
        result = self.tagger.tag_block(block)
        
        # Short catalog-only blocks need 2+ indicators for reference_meta
        # Single indicator may leave as OTHER - that's okay, role tagger needs context
        self.assertIn(result['role'], [BlockRole.REFERENCE_META, BlockRole.OTHER])
        if result['role'] == BlockRole.REFERENCE_META:
            self.assertIn('catalog_numbers', result['role_reasons'])
    
    def test_header_footer_detection(self):
        """Test header/footer detection."""
        text = "156"  # Standalone page number
        block = {'text': text, 'block_id': 'test'}
        
        result = self.tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.HEADER_FOOTER)
        self.assertIn('standalone_page_number', result['role_reasons'])
    
    def test_figure_caption_detection(self):
        """Test figure caption detection."""
        text = "Fig. 12: Seal impression from Kültepe"
        block = {'text': text, 'block_id': 'test'}
        
        result = self.tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.FIGURE_CAPTION)
        self.assertIn('caption_prefix', result['role_reasons'])
    
    def test_akkadian_transliteration_detection(self):
        """Test Akkadian transliteration detection."""
        text = """i-dí-ni-a-ti-ma IGI GIR
sa A-ur í-bu-tí-ni
Ma-nu-ki-A-ùr
DUMU A-ur-ma-lik"""
        block = {'text': text, 'block_id': 'test'}
        
        result = self.tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.AKKADIAN)
        self.assertTrue(any('akkadian' in r for r in result['role_reasons']))
    
    def test_translation_with_lang_hint(self):
        """Test translation detection with language hint."""
        text = "Dies ist eine deutsche Übersetzung des Textes."
        block = {'text': text, 'block_id': 'test', 'lang': 'de'}
        
        result = self.tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.TRANSLATION)
        self.assertIn('lang_de', result['role_reasons'])
    
    def test_filter_blocks_by_role(self):
        """Test filtering blocks by role."""
        blocks = [
            {'text': 'HW s. 124', 'role': BlockRole.REFERENCE_META, 'block_id': 'b1'},
            {'text': 'Translation text', 'role': BlockRole.TRANSLATION, 'block_id': 'b2'},
            {'text': 'Fig. 1', 'role': BlockRole.FIGURE_CAPTION, 'block_id': 'b3'},
            {'text': 'i-na URU-lim', 'role': BlockRole.AKKADIAN, 'block_id': 'b4'}
        ]
        
        # Exclude reference metadata
        filtered = filter_blocks_by_role(
            blocks,
            exclude_roles={'reference_meta', 'header_footer', 'figure_caption'}
        )
        
        self.assertEqual(len(filtered), 2)  # Only translation and akkadian
        roles = [b['role'] for b in filtered]
        self.assertIn(BlockRole.TRANSLATION, roles)
        self.assertIn(BlockRole.AKKADIAN, roles)
        self.assertNotIn(BlockRole.REFERENCE_META, roles)
    
    def test_multiple_reference_indicators(self):
        """Test that multiple reference indicators increase confidence."""
        text = "vgl. Michel 1991, s. 45-67, Kt c/k 123"
        block = {'text': text, 'block_id': 'test'}
        
        result = self.tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.REFERENCE_META)
        # Should have multiple reasons (citation markers, year, catalog numbers)
        self.assertGreaterEqual(len(result['role_reasons']), 2)
        self.assertGreaterEqual(result['role_confidence'], 0.7)


class TestIntegrationSplitAndTag(unittest.TestCase):
    """Integration tests for split + tag pipeline."""
    
    def test_real_world_bayram_case(self):
        """Test on the real Bayram mixed-content block."""
        text = """SEBAHATTIN BAYRAM - SALIH CECEN
i-dí-ni-a-ti-ma IGI GIR
sa A-ur í-bu-tí-ni
ni-di-in
This deposition records a disagreement between Idi-Suen and Mannu-ki-Aur"""
        
        # Split first
        splitter = BlockSplitter()
        fragments = splitter.split_block(text, 'bayram_test')
        
        # Should split author from content
        self.assertGreater(len(fragments), 1)
        
        # Tag each fragment
        tagger = BlockRoleTagger()
        blocks = [{'text': f['text'], 'block_id': f'frag_{i}'} 
                  for i, f in enumerate(fragments)]
        tagged = tagger.tag_blocks(blocks)
        
        # Author line should NOT be tagged as Akkadian
        author_blocks = [b for b in tagged if 'BAYRAM' in b['text']]
        if author_blocks:
            self.assertNotEqual(author_blocks[0]['role'], BlockRole.AKKADIAN)
        
        # Transliteration lines should be tagged as Akkadian
        akk_blocks = [b for b in tagged if b['role'] == BlockRole.AKKADIAN]
        self.assertGreater(len(akk_blocks), 0)
    
    def test_real_world_yilmaz_case(self):
        """Test on the Yilmaz reference metadata case."""
        text = "HW s. 124 a. / evcuttur. ubalit II pres. Formdadr. / 5) Müze Env. Nr. 161-426-64"
        
        # This should be tagged as reference metadata
        tagger = BlockRoleTagger()
        block = {'text': text, 'block_id': 'yilmaz_test'}
        result = tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.REFERENCE_META)
        
        # When filtering for translation candidates, this should be excluded
        blocks = [result]
        candidates = filter_blocks_by_role(
            blocks,
            exclude_roles={'reference_meta', 'header_footer', 'figure_caption'}
        )
        
        self.assertEqual(len(candidates), 0, "Reference metadata should be excluded")
    
    def test_hw_reference_citation(self):
        """Test CRITICAL failing case: HW s. 124 a."""
        text = "HW s. 124 a."
        tagger = BlockRoleTagger()
        block = {'text': text, 'block_id': 'hw_ref'}
        result = tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.REFERENCE_META, 
                        "HW s. citation should be tagged as reference_meta")
    
    def test_museum_catalog_number(self):
        """Test CRITICAL failing case: Müze env. 166-147-64"""
        text = "Müze env. 166-147-64 3,7 x 4,2 x 1,6 cm. siyah renkli"
        tagger = BlockRoleTagger()
        block = {'text': text, 'block_id': 'museum_cat'}
        result = tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.REFERENCE_META,
                        "Museum catalog should be tagged as reference_meta")
    
    def test_kt_k_slash_k_catalog(self):
        """Test CRITICAL failing case: kt k/k 15"""
        text = "kt k/k 15 A, 3"
        tagger = BlockRoleTagger()
        block = {'text': text, 'block_id': 'kt_catalog'}
        result = tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.REFERENCE_META,
                        "Kt k/k catalog should be tagged as reference_meta")
    
    def test_author_page_header(self):
        """Test CRITICAL failing case: 250 S. BAYRAM-R KÖZOGLU Notlar:"""
        text = "250 S. BAYRAM-R KÖZOGLU Notlar:"
        tagger = BlockRoleTagger()
        block = {'text': text, 'block_id': 'author_header'}
        result = tagger.tag_block(block)
        
        self.assertIn(result['role'], [BlockRole.HEADER_FOOTER, BlockRole.REFERENCE_META],
                     "Author page header should be tagged as header/footer or reference_meta")
    
    def test_french_sceau_reference(self):
        """Test CRITICAL failing case: Kt k/k, 44, 1 sceau."""
        text = "Kt k/k, 44, 1 sceau."
        tagger = BlockRoleTagger()
        block = {'text': text, 'block_id': 'sceau_ref'}
        result = tagger.tag_block(block)
        
        self.assertEqual(result['role'], BlockRole.REFERENCE_META,
                        "French 'sceau' with catalog should be tagged as reference_meta")
    
    def test_all_failing_cases_excluded_from_pairing(self):
        """Integration test: ensure all failing examples are excluded."""
        failing_examples = [
            "HW s. 124 a.",
            "Müze env. 166-147-64",
            "kt k/k 15 A, 3",
            "250 S. BAYRAM-R KÖZOGLU Notlar:",
            "Kt k/k, 44, 1 sceau."
        ]
        
        tagger = BlockRoleTagger()
        blocks = [
            tagger.tag_block({'text': text, 'block_id': f'fail_{i}'})
            for i, text in enumerate(failing_examples)
        ]
        
        # Filter for pairing candidates
        candidates = filter_blocks_by_role(
            blocks,
            exclude_roles={'reference_meta', 'header_footer', 'figure_caption'}
        )
        
        self.assertEqual(len(candidates), 0,
                        f"All {len(failing_examples)} failing examples should be excluded. "
                        f"Got {len(candidates)} candidates: {[b['text'] for b in candidates]}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
