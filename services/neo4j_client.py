import logging
import re
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase, exceptions

logger = logging.getLogger(__name__)

# Cosine similarity threshold for entity resolution
MERGE_THRESHOLD = 0.92

# Neo4j vector index settings
VECTOR_INDEX_NAME = "entity_embedding_index_384"
VECTOR_DIMENSIONS = 384
VECTOR_SIMILARITY_FN = "cosine"


def sanitize_cypher_label(label: str) -> str:
    """Sanitize string to valid Cypher label / relationship type."""
    if not label:
        return "Entity"
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label.strip())
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "E_" + sanitized
    return sanitized


def ensure_vector_index(session) -> None:
    """
    Create the vector index on Entity.embedding if it does not already exist.
    Uses the Neo4j 5.x vector index syntax.
    """
    try:
        session.run(
            f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {VECTOR_DIMENSIONS},
                    `vector.similarity_function`: '{VECTOR_SIMILARITY_FN}'
                }}
            }}
            """
        )
        # Wait for the index to come online before we try to query it
        session.run(f"CALL db.awaitIndex('{VECTOR_INDEX_NAME}', 300)")
        logger.info(f"Vector index '{VECTOR_INDEX_NAME}' ensured and ONLINE.")
    except Exception as e:
        logger.warning(f"Could not create or await vector index (continuing without it): {e}")


def find_similar_entity(session, embedding: List[float], top_k: int = 1) -> Optional[Dict[str, Any]]:
    """
    Query the Neo4j vector index for the most similar existing entity.

    Returns a dict with keys: name, type, description, score
    or None if no sufficiently similar entity exists.
    """
    if not embedding:
        return None

    try:
        result = session.run(
            f"""
            CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding)
            YIELD node, score
            RETURN node.name AS name, node.type AS type, node.description AS description, score
            ORDER BY score DESC
            LIMIT 1
            """,
            top_k=top_k,
            embedding=embedding
        )
        record = result.single()
        if record:
            return {
                "name": record["name"],
                "type": record["type"],
                "description": record["description"],
                "score": record["score"]
            }
    except Exception as e:
        logger.warning(f"Vector similarity query failed: {e}")

    return None


def upsert_entity(session, ent_name: str, ent_type: str, description: str,
                   embedding: List[float]) -> str:
    """
    Resolve and upsert a single entity into Neo4j.

    Returns the canonical node name that was ultimately used
    (may differ from ent_name if merged with an existing node).
    """
    valid_label = sanitize_cypher_label(ent_type)

    # --- Semantic resolution path (embedding available) ---
    if embedding:
        match = find_similar_entity(session, embedding)

        if match and match["score"] >= MERGE_THRESHOLD:
            # MERGE path: entity already exists — keep original, only add the new type label
            canonical_name = match["name"]
            logger.debug(
                f"MERGE  '{ent_name}' → existing node '{canonical_name}' "
                f"(similarity={match['score']:.4f} ≥ {MERGE_THRESHOLD})"
            )
            # Add the incoming type as an additional label if not already present
            session.run(
                f"""
                MATCH (e:Entity {{name: $canonical_name}})
                SET e:{valid_label}
                """,
                canonical_name=canonical_name
            )
            return canonical_name

        else:
            score_info = f"similarity={match['score']:.4f}" if match else "no match in index"
            logger.debug(
                f"CREATE '{ent_name}' as new node ({score_info} < {MERGE_THRESHOLD})"
            )

    else:
        # --- Fallback: no embedding — exact name merge ---
        logger.debug(f"FALLBACK exact-name merge for '{ent_name}' (no embedding available).")

    # CREATE / exact-name-MERGE path
    # Note: SET e:Label is written inline — no CALL subquery needed (avoids Neo4j deprecation warning).
    session.run(
        f"""
        MERGE (e:Entity {{name: $name}})
        ON CREATE SET
            e.type        = $type,
            e.description = $description,
            e.embedding   = $embedding
        SET e:{valid_label}
        """,
        name=ent_name,
        type=ent_type,
        description=description,
        embedding=embedding if embedding else None
    )
    return ent_name
