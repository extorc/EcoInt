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

from services.neo4j_client import ensure_vector_index, upsert_entity


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
            ensure_vector_index(session)

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
                        canonical_name = upsert_entity(
                            session, ent_name, ent_type, description, embedding
                        )
                        canonical_map[ent_name] = canonical_name
                        nodes_count += 1
                    except Exception as e:
                        err = f"Failed to upsert entity '{ent_name}': {e}"
                        logger.error(err)
                        errors.append(err)
                        canonical_map[ent_name] = ent_name  # best-effort fallback

                article_url = item.get("link", "").strip()
                article_title = item.get("title", "").strip()
                article_source = item.get("source", "").strip()
                published_rss = item.get("published_rss", "").strip()
                category = item.get("category", "").strip()

                if not article_url:
                    logger.warning(f"No link found for article '{article_title}', skipping Article Node creation.")
                    continue

                try:
                    session.run(
                        """
                        MERGE (a:Article {url: $url})
                        ON CREATE SET a.title = $title, a.source = $source, a.published_rss = $published_rss, a.category = $category
                        """,
                        url=article_url, title=article_title, source=article_source, published_rss=published_rss, category=category
                    )
                    nodes_count += 1
                except Exception as e:
                    err = f"Failed to upsert Article '{article_title}': {e}"
                    logger.error(err)
                    errors.append(err)
                    continue

                for ent_name, canonical_name in canonical_map.items():
                    try:
                        session.run(
                            """
                            MATCH (a:Article {url: $url})
                            MATCH (e:Entity {name: $canonical_name})
                            MERGE (e)-[r:IN]->(a)
                            """,
                            url=article_url,
                            canonical_name=canonical_name
                        )
                        relationships_count += 1
                    except Exception as e:
                        err = f"Failed to create relationship ({canonical_name})-[IN]->(Article): {e}"
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
