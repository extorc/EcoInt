"""
LangGraph Node: store_neo4j
Ingests entities and relationships into Neo4j with semantic entity resolution.

Entity Resolution Strategy (per entity):
  1. If the entity has a valid embedding → query the Neo4j vector index.
     - Similarity >= MERGE_THRESHOLD (0.92): entity already exists semantically → MERGE
       (preserve original name/type/description/embedding; only add new relationships)
     - Similarity <  MERGE_THRESHOLD       : genuinely new entity → CREATE
  2. If no embedding available (empty list) → fall back to exact name-based MERGE.

Vector index is auto-created on first run (IF NOT EXISTS) using Neo4j's vector index API.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from neo4j import GraphDatabase, exceptions

import config

logger = logging.getLogger(__name__)

# Silence Neo4j driver's internal notifications (prevents IF NOT EXISTS and DEPRECATION spam)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# Cosine similarity threshold for entity resolution
MERGE_THRESHOLD = 0.92

# Neo4j vector index settings
VECTOR_INDEX_NAME = "entity_embedding_index"
VECTOR_DIMENSIONS = 768
VECTOR_SIMILARITY_FN = "cosine"


def _sanitize_cypher_label(label: str) -> str:
    """Sanitize string to valid Cypher label / relationship type."""
    if not label:
        return "Entity"
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label.strip())
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "E_" + sanitized
    return sanitized


def _ensure_vector_index(session) -> None:
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
        logger.info(f"Vector index '{VECTOR_INDEX_NAME}' ensured (created or already exists).")
    except Exception as e:
        logger.warning(f"Could not create vector index (continuing without it): {e}")


def _find_similar_entity(session, embedding: List[float], top_k: int = 1) -> Optional[Dict[str, Any]]:
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


def _upsert_entity(session, ent_name: str, ent_type: str, description: str,
                   embedding: List[float]) -> str:
    """
    Resolve and upsert a single entity into Neo4j.

    Returns the canonical node name that was ultimately used
    (may differ from ent_name if merged with an existing node).
    """
    valid_label = _sanitize_cypher_label(ent_type)

    # --- Semantic resolution path (embedding available) ---
    if embedding:
        match = _find_similar_entity(session, embedding)

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


def store_neo4j_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node: Ingests extracted entities and relationships into Neo4j.

    Entity Resolution:
        Before creating any entity, a cosine similarity search is performed against
        the vector index. If an existing node has similarity >= MERGE_THRESHOLD (0.92),
        the incoming entity is considered the same and is merged (original node preserved).
        Otherwise a new node is created.

    State Input:
        - extracted_knowledge: Articles with entities (including description & embedding) and relationships.
        - neo4j_uri / neo4j_username / neo4j_password / neo4j_database (optional, falls back to config).
        - errors: Existing pipeline error list.

    State Output:
        - neo4j_nodes_created: Count of node operations processed.
        - neo4j_relationships_created: Count of relationship operations.
        - neo4j_status: 'SUCCESS' | 'FAILED_CONNECTION' | 'ERROR' | 'NO_DATA'.
        - errors: Updated error list.
    """
    logger.info("=== Starting Node: store_neo4j (with semantic entity resolution) ===")

    extracted_knowledge: List[Dict[str, Any]] = state.get("extracted_knowledge", [])
    errors: List[str] = state.get("errors", [])

    uri = state.get("neo4j_uri") or getattr(config, "NEO4J_URI", "bolt://localhost:7687")
    username = state.get("neo4j_username") or getattr(config, "NEO4J_USERNAME", "neo4j")
    password = state.get("neo4j_password") or getattr(config, "NEO4J_PASSWORD", "password")
    database = state.get("neo4j_database") or getattr(config, "NEO4J_DATABASE", "neo4j")

    if not extracted_knowledge:
        msg = "No extracted knowledge available to store in Neo4j."
        logger.warning(msg)
        errors.append(msg)
        return {
            "neo4j_nodes_created": 0,
            "neo4j_relationships_created": 0,
            "neo4j_status": "NO_DATA",
            "errors": errors
        }

    driver = None
    nodes_count = 0
    relationships_count = 0
    neo4j_status = "UNKNOWN"

    try:
        logger.info(f"Connecting to Neo4j at '{uri}' as '{username}'...")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j.")

        with driver.session(database=database) as session:

            # Step 1: Ensure vector index exists
            _ensure_vector_index(session)

            # Step 2: Process each article
            for item in extracted_knowledge:
                # --- Build a canonical-name lookup for this article's entities ---
                # Maps incoming entity name → the resolved canonical Neo4j node name
                # This is needed so relationships reference the right merged node.
                canonical_map: Dict[str, str] = {}

                # --- Upsert entities ---
                for ent in item.get("entities", []):
                    ent_name = ent.get("name", "").strip()
                    ent_type = ent.get("type", "Entity").strip()
                    description = ent.get("description", "").strip()
                    embedding = ent.get("embedding", [])

                    if not ent_name:
                        continue

                    try:
                        canonical_name = _upsert_entity(
                            session, ent_name, ent_type, description, embedding
                        )
                        canonical_map[ent_name] = canonical_name
                        nodes_count += 1
                    except Exception as e:
                        err = f"Failed to upsert entity '{ent_name}': {e}"
                        logger.error(err)
                        errors.append(err)
                        canonical_map[ent_name] = ent_name  # best-effort fallback

                # --- Upsert entity-to-entity relationships ---
                for rel in item.get("relationships", []):
                    src_raw = rel.get("source", "").strip()
                    tgt_raw = rel.get("target", "").strip()
                    rel_type = rel.get("relationship", "ASSOCIATED_WITH").strip()

                    if not src_raw or not tgt_raw or src_raw == tgt_raw:
                        continue

                    # Resolve to canonical names (handles merged entities)
                    src_name = canonical_map.get(src_raw, src_raw)
                    tgt_name = canonical_map.get(tgt_raw, tgt_raw)

                    clean_rel = _sanitize_cypher_label(rel_type).upper()

                    try:
                        session.run(
                            f"""
                            MERGE (s:Entity {{name: $src_name}})
                            MERGE (t:Entity {{name: $tgt_name}})
                            MERGE (s)-[r:`{clean_rel}`]->(t)
                            """,
                            src_name=src_name,
                            tgt_name=tgt_name
                        )
                        relationships_count += 1
                    except Exception as e:
                        err = f"Failed to create relationship ({src_name})-[{clean_rel}]->({tgt_name}): {e}"
                        logger.error(err)
                        errors.append(err)

        neo4j_status = "SUCCESS"
        logger.info(
            f"Neo4j Ingestion Complete: {nodes_count} node operations, "
            f"{relationships_count} relationship operations."
        )

    except (exceptions.ServiceUnavailable, exceptions.AuthError, ConnectionRefusedError, OSError) as conn_err:
        neo4j_status = "FAILED_CONNECTION"
        error_msg = (
            f"Neo4j Connection Error ({uri}): {conn_err}. "
            "Ensure Neo4j is running and credentials are correct "
            "(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)."
        )
        logger.error(error_msg)
        errors.append(error_msg)

    except Exception as ex:
        neo4j_status = "ERROR"
        error_msg = f"Unexpected error storing data in Neo4j: {ex}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)

    finally:
        if driver:
            try:
                driver.close()
            except Exception:
                pass

    logger.info("=== Node store_neo4j Complete ===")

    return {
        "neo4j_nodes_created": nodes_count,
        "neo4j_relationships_created": relationships_count,
        "neo4j_status": neo4j_status,
        "errors": errors
    }
