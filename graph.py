"""
LangGraph Ingestion Pipeline Graph Definition.
Defines state schema, nodes, edges, and compiles the executable 8-stage graph.
"""

import logging
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END

from data_models import PipelineState
from nodes import (
    fetch_rss_preprocess_node,
    sentence_segmentation_node,
    named_entity_recognition_node,
    entity_normalization_node,
    relationship_extraction_node,
    entity_resolution_node,
    entity_embeddings_node,
    store_neo4j_node,
)
import config

logger = logging.getLogger("graph_pipeline")


def build_pipeline_graph():
    """
    Constructs and compiles the 8-stage LangGraph workflow for Neo4j Knowledge Graph Ingestion.
    """
    workflow = StateGraph(PipelineState)

    # 1. Register Nodes
    workflow.add_node("preprocessing", fetch_rss_preprocess_node)
    workflow.add_node("sentence_segmentation", sentence_segmentation_node)
    workflow.add_node("named_entity_recognition", named_entity_recognition_node)
    workflow.add_node("entity_normalization", entity_normalization_node)
    workflow.add_node("relationship_extraction", relationship_extraction_node)
    workflow.add_node("entity_resolution", entity_resolution_node)
    workflow.add_node("entity_embeddings", entity_embeddings_node)
    workflow.add_node("store_neo4j", store_neo4j_node)

    # 2. Register Linear Sequential Edges
    workflow.add_edge(START, "preprocessing")
    workflow.add_edge("preprocessing", "sentence_segmentation")
    workflow.add_edge("sentence_segmentation", "named_entity_recognition")
    workflow.add_edge("named_entity_recognition", "entity_normalization")
    workflow.add_edge("entity_normalization", "relationship_extraction")
    workflow.add_edge("relationship_extraction", "entity_resolution")
    workflow.add_edge("entity_resolution", "entity_embeddings")
    workflow.add_edge("entity_embeddings", "store_neo4j")
    workflow.add_edge("store_neo4j", END)

    app = workflow.compile()
    return app


# Pre-compiled executable graph instance
app = build_pipeline_graph()


def run_pipeline(
    rss_feeds: Optional[List[Dict[str, str]]] = None,
    max_articles_per_feed: int = config.MAX_ARTICLES_PER_FEED,
    total_max_articles: int = config.TOTAL_MAX_ARTICLES,
    neo4j_uri: str = config.NEO4J_URI,
    neo4j_username: str = config.NEO4J_USERNAME,
    neo4j_password: str = config.NEO4J_PASSWORD,
    neo4j_database: str = config.NEO4J_DATABASE,
) -> PipelineState:
    """
    Helper function to initialize state and execute the LangGraph pipeline.
    """
    initial_state: PipelineState = {
        "rss_feeds": rss_feeds or config.RSS_FEEDS,
        "max_articles_per_feed": max_articles_per_feed,
        "total_max_articles": total_max_articles,
        "documents": [],
        "sentences": [],
        "raw_entities": [],
        "normalized_entities": [],
        "relationships": [],
        "resolved_entities": [],
        "embedded_entities": [],
        "total_entities_extracted": 0,
        "total_relationships_extracted": 0,
        "neo4j_uri": neo4j_uri,
        "neo4j_username": neo4j_username,
        "neo4j_password": neo4j_password,
        "neo4j_database": neo4j_database,
        "neo4j_nodes_created": 0,
        "neo4j_relationships_created": 0,
        "neo4j_status": "PENDING",
        "errors": [],
    }

    logger.info("Starting Economic Intelligence 8-stage LangGraph Pipeline...")
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    result = run_pipeline()
    print("\n--- Pipeline Execution Summary ---")
    print(f"Documents Ingested:      {len(result.get('documents', []))}")
    print(f"Sentences Segmented:     {len(result.get('sentences', []))}")
    print(f"Entities Extracted:      {result.get('total_entities_extracted')}")
    print(f"Canonical Resolved Ents: {len(result.get('resolved_entities', []))}")
    print(f"Relationships Extracted: {result.get('total_relationships_extracted')}")
    print(f"Neo4j Status:            {result.get('neo4j_status')}")
    print(f"Neo4j Nodes Created:     {result.get('neo4j_nodes_created')}")
    print(f"Neo4j Links Created:     {result.get('neo4j_relationships_created')}")
    print(f"Warnings/Errors:         {len(result.get('errors', []))}")
