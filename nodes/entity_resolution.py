"""
Node 6: Entity Resolution
Generates entity candidates via RapidFuzz fuzzy matching & Neo4j lookup,
ranks them with BAAI/bge-reranker-large CrossEncoder, and merges co-referent entities.
"""

import logging
from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder
from data_models import PipelineState
import config

logger = logging.getLogger("entity_resolution_node")

_RERANKER_MODEL_INSTANCE = None


def get_reranker_model():
    """Lazy initialization of BGE Reranker (CrossEncoder)."""
    global _RERANKER_MODEL_INSTANCE
    if _RERANKER_MODEL_INSTANCE is None:
        logger.info(f"Loading candidate reranker model: '{config.RERANKER_MODEL}'...")
        try:
            _RERANKER_MODEL_INSTANCE = CrossEncoder(config.RERANKER_MODEL)
        except Exception as err:
            logger.warning(f"Failed to load '{config.RERANKER_MODEL}' ({err}). Falling back to dummy reranker.")
            _RERANKER_MODEL_INSTANCE = None
    return _RERANKER_MODEL_INSTANCE


def entity_resolution_node(state: PipelineState) -> PipelineState:
    """
    LangGraph Node: Entity Resolution
    Identifies synonymous / co-referent entities across documents and unifies their canonical UUIDs.
    """
    logger.info("--- [Stage 6/8] Entity Resolution (RapidFuzz + BGE Reranker) ---")
    normalized_entities = state.get("normalized_entities", [])
    relationships = state.get("relationships", [])
    errors: List[str] = list(state.get("errors", []))

    if not normalized_entities:
        return {**state, "resolved_entities": []}

    # Extract unique entity representations: (normalized_name, entity_type) -> list of entity dicts
    unique_entities: Dict[str, Dict[str, Any]] = {}
    for ent in normalized_entities:
        c_uuid = ent.get("canonical_uuid") or ent.get("uuid")
        if c_uuid not in unique_entities:
            unique_entities[c_uuid] = dict(ent)

    entity_list = list(unique_entities.values())
    n = len(entity_list)

    # Union-Find / Disjoint Set structure for entity merging
    parent = {e["canonical_uuid"]: e["canonical_uuid"] for e in entity_list}

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # 1. Candidate Generation via RapidFuzz token matching
    candidate_pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            e1 = entity_list[i]
            e2 = entity_list[j]

            # Only consider resolution between same entity types
            if e1["entity_type"] != e2["entity_type"]:
                continue

            name1 = e1["normalized_name"]
            name2 = e2["normalized_name"]

            # Compute RapidFuzz token set ratio
            similarity = fuzz.token_set_ratio(name1, name2)
            if similarity >= config.FUZZY_MATCH_THRESHOLD:
                candidate_pairs.append((i, j, similarity))

    # 2. Candidate Ranking using BGE Reranker CrossEncoder
    reranker = get_reranker_model()
    merged_count = 0

    for i, j, fuzzy_score in candidate_pairs:
        e1 = entity_list[i]
        e2 = entity_list[j]

        # Prepare context pairs for CrossEncoder
        text_pair = [
            f"Entity: {e1['name']}, Type: {e1['entity_type']}",
            f"Entity: {e2['name']}, Type: {e2['entity_type']}",
        ]

        if reranker is not None:
            try:
                score = float(reranker.predict([text_pair])[0])
            except Exception as err:
                logger.warning(f"Reranker scoring failed: {err}")
                score = fuzzy_score / 100.0
        else:
            score = fuzzy_score / 100.0

        if score >= config.RERANKER_SCORE_THRESHOLD:
            union(e1["canonical_uuid"], e2["canonical_uuid"])
            merged_count += 1

    # Map canonical UUIDs to root representative UUID
    uuid_mapping = {}
    for c_uuid in parent:
        uuid_mapping[c_uuid] = find(c_uuid)

    # Build resolved entity models
    resolved_entities_dict: Dict[str, Dict[str, Any]] = {}
    for ent in normalized_entities:
        orig_uuid = ent.get("canonical_uuid") or ent.get("uuid")
        root_uuid = uuid_mapping.get(orig_uuid, orig_uuid)
        ent["canonical_uuid"] = root_uuid

        if root_uuid not in resolved_entities_dict:
            resolved_entities_dict[root_uuid] = {
                "uuid": root_uuid,
                "name": ent["name"],
                "normalized_name": ent["normalized_name"],
                "entity_type": ent["entity_type"],
                "description": ent.get("description", ""),
                "confidence": ent["confidence"],
                "sentence_id": ent["sentence_id"],
                "document_uuid": ent["document_uuid"],
            }

    resolved_entities = list(resolved_entities_dict.values())

    # Remap relationship source/target UUIDs to canonical resolved entity UUIDs
    for rel in relationships:
        src_orig = rel["source_entity_uuid"]
        tgt_orig = rel["target_entity_uuid"]
        rel["source_entity_uuid"] = uuid_mapping.get(src_orig, src_orig)
        rel["target_entity_uuid"] = uuid_mapping.get(tgt_orig, tgt_orig)

    logger.info(
        f"Entity resolution completed. Merged {merged_count} candidate pairs down to "
        f"{len(resolved_entities)} unique canonical entities."
    )

    return {
        **state,
        "resolved_entities": resolved_entities,
        "relationships": relationships,
        "errors": errors,
    }
