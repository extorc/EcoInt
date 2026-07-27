"""
LangGraph Node: embed_entities
Generates vector embeddings for each entity's description using Google text-embedding-004.
Embeddings are attached to each entity dict so the store_neo4j node can perform
semantic similarity search before deciding to CREATE or MERGE a node.
"""

import logging
import time
from typing import Dict, Any, List

import config

logger = logging.getLogger(__name__)

# Embedding model candidates — tried in order until one succeeds.
# text-embedding-004 requires the v1 endpoint; if the installed library defaults
# to v1beta it returns a 404. embedding-001 is the stable v1beta fallback and
# also produces 768-dim vectors, so the Neo4j vector index needs no changes.
CANDIDATE_EMBEDDING_MODELS = [
    "models/gemini-embedding-2",
    "models/text-embedding-004",
    "models/gemini-embedding-2-preview",
    "models/embedding-001",
    "models/gemini-embedding-001"
]

# Dimensions produced by both candidates above
EMBEDDING_DIMENSIONS = 768

# Retry settings for quota/rate-limit errors
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def _embed_one(text: str, api_key: str) -> List[float]:
    """
    Embed a single text string, trying each candidate model in order.
    Returns a 768-dim float list, or [] if all models fail.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    for model_name in CANDIDATE_EMBEDDING_MODELS:
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                result = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type="SEMANTIC_SIMILARITY"
                )
                embedding = result["embedding"]
                
                # Enforce exactly 768 dimensions (Neo4j index size).
                # Gemini models use Matryoshka representation learning, so we can
                # safely slice the first N dimensions without losing much accuracy.
                if len(embedding) > EMBEDDING_DIMENSIONS:
                    embedding = embedding[:EMBEDDING_DIMENSIONS]
                    
                logger.debug(f"Embedded via {model_name} → {len(embedding)} dims")
                return embedding

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "quota" in err_msg.lower():
                    attempt += 1
                    logger.warning(
                        f"[{model_name}] Quota hit (attempt {attempt}/{MAX_RETRIES}). "
                        f"Retrying in {RETRY_DELAY_SECONDS}s..."
                    )
                    time.sleep(RETRY_DELAY_SECONDS)
                elif "404" in err_msg or "not found" in err_msg.lower():
                    # Model not available on this API version — try next candidate
                    logger.warning(
                        f"[{model_name}] Not available (404). "
                        f"Falling back to next embedding model candidate..."
                    )
                    break  # exit retry loop, move to next model
                else:
                    logger.error(f"[{model_name}] Embedding error: {e}")
                    break  # non-retryable error — try next model

    logger.error(f"All embedding model candidates failed for text: '{text[:60]}'")
    return []


def _embed_texts(texts: List[str], api_key: str) -> List[List[float]]:
    """
    Embed a list of text strings, one at a time, using _embed_one.
    Returns a list of 768-dim float vectors (or [] for any that fail).
    """
    all_embeddings: List[List[float]] = []

    for i, text in enumerate(texts):
        embedding = _embed_one(text, api_key)
        all_embeddings.append(embedding)
        if embedding:
            logger.debug(f"[{i+1}/{len(texts)}] OK — '{text[:60]}' → {len(embedding)} dims")
        else:
            logger.error(f"[{i+1}/{len(texts)}] FAILED — '{text[:60]}'")

    return all_embeddings


def embed_entities_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node: Generates a 768-dim vector embedding for each entity's description.

    For every entity across all articles in extracted_knowledge, this node:
      1. Reads the entity's 'description' field (set by extract_knowledge_node).
      2. Calls Gemini text-embedding-004 with task_type=SEMANTIC_SIMILARITY.
      3. Attaches the resulting float list as entity['embedding'].

    Entities with missing or empty descriptions get an empty embedding ([]) and
    will be stored as new nodes without vector search (safe fallback).

    State Input:
        - extracted_knowledge: List of article dicts, each with an 'entities' list.
        - errors: Existing pipeline error list.

    State Output:
        - extracted_knowledge: Same structure, but each entity now has an 'embedding' key.
        - errors: Updated error list.
    """
    logger.info("=== Starting Node: embed_entities ===")

    extracted_knowledge: List[Dict[str, Any]] = state.get("extracted_knowledge", [])
    errors: List[str] = state.get("errors", [])

    gemini_key = getattr(config, "GEMINI_API_KEY", "") or config.get_gemini_api_key()

    if not gemini_key:
        msg = "GEMINI_API_KEY missing — skipping embedding. Nodes will be stored without vector embeddings."
        logger.warning(msg)
        errors.append(msg)
        return {"extracted_knowledge": extracted_knowledge, "errors": errors}

    if not extracted_knowledge:
        logger.info("No extracted knowledge to embed. Skipping.")
        return {"extracted_knowledge": extracted_knowledge, "errors": errors}

    # ------------------------------------------------------------------ #
    # Flatten all entity descriptions into a single list, keeping         #
    # (article_idx, entity_idx) pointers so we can write embeddings back  #
    # ------------------------------------------------------------------ #
    index_map: List[tuple] = []   # [(article_idx, entity_idx), ...]
    texts_to_embed: List[str] = []

    for art_idx, article in enumerate(extracted_knowledge):
        for ent_idx, entity in enumerate(article.get("entities", [])):
            desc = entity.get("description", "").strip()
            if not desc:
                # Fallback: embed the entity name + type if description is absent
                desc = f"{entity.get('name', '')} ({entity.get('type', 'Entity')})"
                logger.debug(
                    f"Entity '{entity.get('name')}' missing description — using name+type as fallback."
                )
            index_map.append((art_idx, ent_idx))
            texts_to_embed.append(desc)

    if not texts_to_embed:
        logger.info("No entity descriptions found to embed.")
        return {"extracted_knowledge": extracted_knowledge, "errors": errors}

    logger.info(f"Embedding {len(texts_to_embed)} entity descriptions via {CANDIDATE_EMBEDDING_MODELS[0]} (with fallback)...")

    try:
        embeddings = _embed_texts(texts_to_embed, gemini_key)
    except Exception as e:
        err = f"Fatal embedding error: {e}"
        logger.error(err, exc_info=True)
        errors.append(err)
        return {"extracted_knowledge": extracted_knowledge, "errors": errors}

    # Write embeddings back into the entity dicts
    embedded_count = 0
    for (art_idx, ent_idx), embedding in zip(index_map, embeddings):
        extracted_knowledge[art_idx]["entities"][ent_idx]["embedding"] = embedding
        if embedding:
            embedded_count += 1

    logger.info(
        f"=== Node embed_entities Complete: {embedded_count}/{len(texts_to_embed)} entities successfully embedded. ==="
    )

    return {
        "extracted_knowledge": extracted_knowledge,
        "errors": errors
    }
