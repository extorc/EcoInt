"""
Node 7: Entity Embeddings
Generates 768-dimensional dense vector embeddings using BAAI/bge-base-en-v1.5.
Combines entity name, entity type, and local sentence context for high-precision semantic indexing.
"""

import logging
from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from data_models import PipelineState
import config

logger = logging.getLogger("entity_embeddings_node")

_EMBEDDING_MODEL_INSTANCE = None


def get_embedding_model():
    """Lazy initialization of BGE embedding model."""
    global _EMBEDDING_MODEL_INSTANCE
    if _EMBEDDING_MODEL_INSTANCE is None:
        logger.info(f"Loading SentenceTransformer embedding model: '{config.EMBEDDING_MODEL}'...")
        _EMBEDDING_MODEL_INSTANCE = SentenceTransformer(config.EMBEDDING_MODEL)
    return _EMBEDDING_MODEL_INSTANCE


def entity_embeddings_node(state: PipelineState) -> PipelineState:
    """
    LangGraph Node: Entity Embeddings
    Encodes resolved entities using BAAI/bge-base-en-v1.5 transformer embeddings.
    """
    logger.info("--- [Stage 7/8] Entity Embeddings (BAAI/bge-base-en-v1.5) ---")
    resolved_entities = state.get("resolved_entities", [])
    sentences = state.get("sentences", [])
    errors: List[str] = list(state.get("errors", []))

    if not resolved_entities:
        return {**state, "embedded_entities": []}

    # Index sentences by sentence_id
    sentence_map = {s["sentence_id"]: s.get("text", "") for s in sentences}

    try:
        model = get_embedding_model()
    except Exception as err:
        err_msg = f"Failed to load embedding model '{config.EMBEDDING_MODEL}': {str(err)}"
        logger.error(err_msg)
        errors.append(err_msg)
        return {**state, "embedded_entities": resolved_entities, "errors": errors}

    # Construct entity context texts
    texts_to_embed: List[str] = []
    for ent in resolved_entities:
        name = ent.get("name", "")
        e_type = ent.get("entity_type", "")
        desc = ent.get("description", "")
        sent_id = ent.get("sentence_id")
        sent_context = sentence_map.get(sent_id, "")
        embed_input = f"Entity: {name} | Type: {e_type} | Description: {desc} | Context: {sent_context}"
        texts_to_embed.append(embed_input)

    try:
        logger.info(f"Generating dense vector embeddings for {len(texts_to_embed)} entities...")
        embeddings = model.encode(texts_to_embed, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

        embedded_entities: List[Dict[str, Any]] = []
        for ent, vec in zip(resolved_entities, embeddings):
            ent_copy = dict(ent)
            ent_copy["embedding"] = vec.tolist()
            embedded_entities.append(ent_copy)

        logger.info("Dense vector embedding generation completed.")
        return {
            **state,
            "resolved_entities": embedded_entities,
            "embedded_entities": embedded_entities,
            "errors": errors,
        }

    except Exception as err:
        err_msg = f"Error during vector embedding generation: {str(err)}"
        logger.error(err_msg)
        errors.append(err_msg)
        return {**state, "resolved_entities": resolved_entities, "embedded_entities": resolved_entities, "errors": errors}
