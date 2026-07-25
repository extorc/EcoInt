import logging
import re
from typing import Dict, Any, List
from neo4j import GraphDatabase, exceptions

import config

logger = logging.getLogger(__name__)


def _sanitize_cypher_label(label: str) -> str:
    """Sanitize string to valid Cypher label / relationship type."""
    if not label:
        return "Entity"
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label.strip())
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "E_" + sanitized
    return sanitized


def store_neo4j_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node: Ingests extracted entities and relationships into Neo4j Knowledge Graph.
    Articles are ephemeral — used only for LLM extraction. Only entities and
    entity-to-entity relationships are persisted in the graph.

    State Input:
        - extracted_knowledge: List of articles with extracted entities and relationships.
        - neo4j_uri (optional): Neo4j connection URI (e.g. 'bolt://localhost:7687').
        - neo4j_username (optional): Neo4j username (e.g. 'neo4j').
        - neo4j_password (optional): Neo4j password.
        - neo4j_database (optional): Neo4j target database name.
        - errors: List of error messages.

    State Output:
        - neo4j_nodes_created: Total count of entity nodes merged/created in Neo4j.
        - neo4j_relationships_created: Total count of entity-to-entity relationships created.
        - neo4j_status: 'SUCCESS' or 'FAILED_CONNECTION'.
        - errors: List of error messages.
    """
    logger.info("=== Starting Node: store_neo4j ===")

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
        logger.info(f"Connecting to Neo4j database at '{uri}' as user '{username}'...")
        driver = GraphDatabase.driver(uri, auth=(username, password))

        # Verify connectivity
        driver.verify_connectivity()
        logger.info(" Successfully connected to Neo4j instance!")

        with driver.session(database=database) as session:
            # Step 1: Ensure Entity uniqueness constraint
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            except Exception as ce:
                logger.warning(f"Could not create constraints (continuing anyway): {ce}")

            # Step 2: Ingest entities and entity-to-entity relationships
            # Articles are not persisted — they are ephemeral extraction sources only.
            for item in extracted_knowledge:
                # Merge Entities only
                for ent in item.get("entities", []):
                    ent_name = ent.get("name", "").strip()
                    ent_type = ent.get("type", "Entity").strip()

                    if not ent_name:
                        continue

                    valid_label = _sanitize_cypher_label(ent_type)

                    session.run(
                        f"""
                        MERGE (e:Entity {{name: $name}})
                        SET e.type = $type
                        WITH e
                        CALL {{
                            WITH e
                            SET e:{valid_label}
                        }}
                        """,
                        name=ent_name,
                        type=ent_type
                    )
                    nodes_count += 1

                # Merge Entity-to-Entity Relationships
                for rel in item.get("relationships", []):
                    src_name = rel.get("source", "").strip()
                    tgt_name = rel.get("target", "").strip()
                    rel_type = rel.get("relationship", "ASSOCIATED_WITH").strip()

                    if not src_name or not tgt_name or src_name == tgt_name:
                        continue

                    clean_rel = _sanitize_cypher_label(rel_type).upper()

                    query = f"""
                    MERGE (s:Entity {{name: $src_name}})
                    MERGE (t:Entity {{name: $tgt_name}})
                    MERGE (s)-[r:`{clean_rel}`]->(t)
                    """
                    session.run(query, src_name=src_name, tgt_name=tgt_name)
                    relationships_count += 1

        neo4j_status = "SUCCESS"
        logger.info(f" Neo4j Ingestion Complete: Processed {nodes_count} node operations & {relationships_count} relationship links.")

    except (exceptions.ServiceUnavailable, exceptions.AuthError, ConnectionRefusedError, OSError) as conn_err:
        neo4j_status = "FAILED_CONNECTION"
        error_msg = (
            f"Neo4j Connection Error ({uri}): {conn_err}. "
            "Ensure Neo4j desktop/Docker container is running and credentials (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) are correct."
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
