"""
LangGraph Node: embed_entities
Generates vector embeddings for each entity's description using FastEmbed locally.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding
        logger.info("Loading FastEmbed model (BAAI/bge-small-en-v1.5)...")
        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedding_model

def embed_entities_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("=== Starting Node: embed_entities (Local FastEmbed) ===")

    extracted_knowledge: List[Dict[str, Any]] = state.get("extracted_knowledge", [])
    errors: List[str] = state.get("errors", [])

    if not extracted_knowledge:
        logger.info("No extracted knowledge to embed. Skipping.")
        return {"extracted_knowledge": extracted_knowledge, "errors": errors}

    index_map: List[tuple] = []
    texts_to_embed: List[str] = []

    for art_idx, article in enumerate(extracted_knowledge):
        for ent_idx, entity in enumerate(article.get("entities", [])):
            desc = entity.get("description", "").strip()
            if not desc:
                desc = f"{entity.get('name', '')} ({entity.get('type', 'Entity')})"
                logger.debug(f"Entity '{entity.get('name')}' missing description — using fallback.")
            index_map.append((art_idx, ent_idx))
            texts_to_embed.append(desc)

    if not texts_to_embed:
        logger.info("No entity descriptions found to embed.")
        return {"extracted_knowledge": extracted_knowledge, "errors": errors}

    logger.info(f"Embedding {len(texts_to_embed)} entity descriptions locally...")

    try:
        model = get_embedding_model()
        # model.embed returns an iterable of numpy arrays
        embeddings_gen = model.embed(texts_to_embed)
        # Convert numpy arrays to list of floats for JSON/Neo4j serialization
        embeddings = [list(map(float, emb)) for emb in embeddings_gen]
    except Exception as e:
        err = f"Fatal embedding error: {e}"
        logger.error(err, exc_info=True)
        errors.append(err)
        return {"extracted_knowledge": extracted_knowledge, "errors": errors}

    embedded_count = 0
    for (art_idx, ent_idx), embedding in zip(index_map, embeddings):
        extracted_knowledge[art_idx]["entities"][ent_idx]["embedding"] = embedding
        if embedding:
            embedded_count += 1

    logger.info(f"=== Node embed_entities Complete: {embedded_count}/{len(texts_to_embed)} entities embedded. ===")

    return {
        "extracted_knowledge": extracted_knowledge,
        "errors": errors
    }
