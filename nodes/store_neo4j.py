"""
Node 8: Neo4j Knowledge Graph Storage
Stores Documents, Entities, and typed Relationship edges into Neo4j.
Configures Neo4j Vector Index on Entity(embedding) for hybrid semantic search.
"""

import logging
from typing import List, Dict, Any
from data_models import PipelineState
import config

logger = logging.getLogger("store_neo4j_node")


def store_neo4j_node(state: PipelineState) -> PipelineState:
    """
    LangGraph Node: Store Neo4j
    Upserts graph nodes and edges into Neo4j database using Cypher driver transactions.
    """
    logger.info("--- [Stage 8/8] Neo4j Knowledge Graph Storage ---")
    documents = state.get("documents", [])
    embedded_entities = state.get("embedded_entities", [])
    relationships = state.get("relationships", [])
    errors: List[str] = list(state.get("errors", []))

    uri = state.get("neo4j_uri", config.NEO4J_URI)
    user = state.get("neo4j_username", config.NEO4J_USERNAME)
    password = state.get("neo4j_password", config.NEO4J_PASSWORD)
    database = state.get("neo4j_database", config.NEO4J_DATABASE)

    nodes_created = 0
    rels_created = 0
    neo4j_status = "SUCCESS"

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session(database=database) as session:
            # 1. Ensure Constraints & Vector Index exist
            logger.info("Setting up Neo4j constraints and vector index...")

            # Unique Constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.uuid IS UNIQUE;")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.uuid IS UNIQUE;")

            # Vector Index on Entity(embedding)
            vector_index_query = """
            CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}
            """
            try:
                session.run(vector_index_query)
                logger.info("Neo4j Vector Index 'entity_embedding_index' configured.")
            except Exception as v_err:
                logger.warning(f"Vector Index query warning (may require Neo4j 5.11+): {v_err}")

            # 2. Batch Ingest Documents
            if documents:
                logger.info(f"Ingesting {len(documents)} Document nodes...")
                doc_query = """
                UNWIND $docs AS doc
                MERGE (d:Document {url: doc.url})
                SET d.uuid = doc.uuid,
                    d.title = doc.title,
                    d.source = doc.source,
                    d.published_date = doc.published_date,
                    d.created_at = doc.created_at
                """
                session.run(doc_query, docs=documents)
                nodes_created += len(documents)

            # 3. Batch Ingest Entities
            if embedded_entities:
                logger.info(f"Ingesting {len(embedded_entities)} Entity nodes...")
                ent_query = """
                UNWIND $ents AS ent
                MERGE (e:Entity {uuid: ent.uuid})
                SET e.name = ent.name,
                    e.normalized_name = ent.normalized_name,
                    e.entity_type = ent.entity_type,
                    e.description = ent.description,
                    e.confidence = ent.confidence
                WITH e, ent
                WHERE ent.embedding IS NOT NULL
                SET e.embedding = ent.embedding
                """
                session.run(ent_query, ents=embedded_entities)
                nodes_created += len(embedded_entities)

                # Batch Ingest MENTIONED_IN edges connecting Entities to Documents
                mention_query = """
                UNWIND $ents AS ent
                MATCH (e:Entity {uuid: ent.uuid})
                MATCH (d:Document {uuid: ent.document_uuid})
                MERGE (e)-[r:MENTIONED_IN]->(d)
                SET r.sentence_id = ent.sentence_id
                """
                try:
                    session.run(mention_query, ents=embedded_entities)
                except Exception as m_err:
                    logger.debug(f"MENTIONED_IN edge query notice: {m_err}")

            # 4. Batch Ingest Relationships
            if relationships:
                logger.info(f"Ingesting {len(relationships)} Relationship edges...")
                for rel in relationships:
                    rel_type = rel.get("relationship_type", "RELATED_TO").upper()
                    # Sanitize rel_type for Cypher syntax
                    rel_type = "".join([c if c.isalnum() else "_" for c in rel_type])

                    cypher_rel_query = f"""
                    MATCH (src:Entity {{uuid: $source_entity_uuid}})
                    MATCH (tgt:Entity {{uuid: $target_entity_uuid}})
                    MERGE (src)-[r:{rel_type}]->(tgt)
                    SET r.uuid = $uuid,
                        r.description = $description,
                        r.confidence = $confidence,
                        r.supporting_sentence = $supporting_sentence,
                        r.source_document = $document_uuid,
                        r.timestamp = $timestamp
                    """
                    session.run(cypher_rel_query, **rel)
                    rels_created += 1

        driver.close()
        logger.info(f"Neo4j ingestion complete. Nodes: {nodes_created}, Rels: {rels_created}.")

    except Exception as err:
        err_msg = f"Neo4j storage connection or query execution failed: {str(err)}"
        logger.warning(err_msg)
        errors.append(err_msg)
        neo4j_status = "FAILED_OR_UNAVAILABLE"

    return {
        **state,
        "neo4j_nodes_created": nodes_created,
        "neo4j_relationships_created": rels_created,
        "neo4j_status": neo4j_status,
        "errors": errors,
    }
