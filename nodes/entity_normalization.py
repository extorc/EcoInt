"""
Node 4: Entity Normalization
Normalizes casing, removes punctuation, normalizes corporate suffixes, resolves stock tickers/aliases,
and deduplicates repeated entity mentions within the same document.
"""

import logging
from typing import List, Dict, Any
from data_models import PipelineState
from utils.text_processing import (
    clean_entity_text,
    normalize_company_name,
)

logger = logging.getLogger("entity_normalization_node")


def entity_normalization_node(state: PipelineState) -> PipelineState:
    """
    LangGraph Node: Entity Normalization
    Takes raw entities from NER and cleans/normalizes them for graph integrity.
    """
    logger.info("--- [Stage 4/8] Entity Normalization & Alias Resolution ---")
    raw_entities = state.get("raw_entities", [])
    errors: List[str] = list(state.get("errors", []))

    normalized_entities: List[Dict[str, Any]] = []
    # Intra-document deduplication tracking map: (doc_uuid, entity_type, normalized_key) -> canonical_entity_dict
    seen_intra_doc: Dict[tuple, Dict[str, Any]] = {}

    for ent in raw_entities:
        original_name = ent.get("name", "")
        e_type = ent.get("entity_type", "")
        doc_uuid = ent.get("document_uuid")
        sent_id = ent.get("sentence_id")

        if not original_name:
            continue

        clean_name = clean_entity_text(original_name)
        if not clean_name:
            continue

        # Compute canonical normalized name
        if e_type in ["Company", "Organization", "Regulator", "Government"]:
            norm_name = normalize_company_name(clean_name)
        else:
            norm_name = clean_name.title()

        ent["name"] = clean_name
        ent["normalized_name"] = norm_name

        # Deduplication key per document
        dedup_key = (doc_uuid, e_type, norm_name.lower())

        if dedup_key in seen_intra_doc:
            # Existing mention in document: reuse UUID to link co-references within doc
            existing_ent = seen_intra_doc[dedup_key]
            ent["canonical_uuid"] = existing_ent["uuid"]
            ent["description"] = existing_ent.get("description", ent.get("description", ""))
            # Retain higher confidence score
            if ent.get("confidence", 0.0) > existing_ent.get("confidence", 0.0):
                existing_ent["confidence"] = ent.get("confidence", 0.0)
        else:
            ent["canonical_uuid"] = ent["uuid"]
            seen_intra_doc[dedup_key] = ent

        normalized_entities.append(ent)

    logger.info(
        f"Entity normalization completed. Processed {len(normalized_entities)} entity mentions "
        f"({len(seen_intra_doc)} unique intra-doc entities)."
    )
    return {
        **state,
        "normalized_entities": normalized_entities,
        "errors": errors,
    }
