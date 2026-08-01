"""
Unit and Integration Verification Test Suite for the Knowledge Extraction Pipeline.
"""

import unittest
from utils.text_processing import (
    clean_html_content,
    normalize_unicode_and_whitespace,
    normalize_company_name,
)
from data_models import DocumentModel, EntityModel, RelationshipModel, ENTITY_TYPES
from graph import build_pipeline_graph


class TestPipelineComponents(unittest.TestCase):

    def test_text_processing_utilities(self):
        raw_html = "<p>Apple Inc. (AAPL) acquired <b>AI startup</b> &nbsp; in London.</p><script>alert('x')</script>"
        cleaned = clean_html_content(raw_html)
        self.assertIn("Apple Inc.", cleaned)
        self.assertNotIn("<script>", cleaned)

        dirty_str = "Microsoft\u00a0\u00a0Corporation   "
        norm_str = normalize_unicode_and_whitespace(dirty_str)
        self.assertEqual(norm_str, "Microsoft Corporation")

        self.assertEqual(normalize_company_name("AAPL"), "Apple")
        self.assertEqual(normalize_company_name("Tesla Inc."), "Tesla")
        self.assertEqual(normalize_company_name("Alphabet Corp."), "Alphabet")

    def test_data_models(self):
        doc = DocumentModel(title="Test Article", source="Reuters", url="https://example.com/test")
        self.assertIsNotNone(doc.uuid)
        self.assertEqual(doc.title, "Test Article")

        ent = EntityModel(
            name="Apple Inc.",
            normalized_name="Apple",
            entity_type="Company",
            sentence_id="s1",
            document_uuid=doc.uuid,
        )
        self.assertIn(ent.entity_type, ENTITY_TYPES)

        rel = RelationshipModel(
            source_entity_uuid=ent.uuid,
            source_entity_name="Apple Inc.",
            target_entity_uuid="target_uuid",
            target_entity_name="Shazam",
            relationship_type="ACQUIRED",
            supporting_sentence="Apple Inc. acquired Shazam.",
            sentence_id="s1",
            document_uuid=doc.uuid,
        )
        self.assertEqual(rel.relationship_type, "ACQUIRED")

    def test_langgraph_compilation(self):
        graph = build_pipeline_graph()
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
