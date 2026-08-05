"""
LangGraph Ingestion Pipeline Graph Definition.
Defines state schema, nodes, edges, and compiles the executable graph.

Pipeline:
  RSS Fetch
    → LLM Entity & Relationship Extraction  (now includes 'description' per entity)
    → Gemini Embedding Generation            (attaches 768-dim vector to each entity)
    → Neo4j Semantic Ingestion               (cosine similarity >= 0.92 → MERGE, else CREATE)
"""

import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from nodes import fetch_rss_node, extract_knowledge_node, embed_entities_node, store_neo4j_node
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("graph_pipeline")


# Define Pipeline State Schema using TypedDict
class PipelineState(TypedDict, total=False):
    rss_feeds: List[Dict[str, str]]
    max_articles_per_feed: int
    total_max_articles: int
    raw_articles: List[Dict[str, Any]]
    extracted_knowledge: List[Dict[str, Any]]   # entities now include 'description' + 'embedding'
    total_entities_extracted: int
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    neo4j_nodes_created: int
    neo4j_relationships_created: int
    neo4j_status: str
    errors: List[str]


def build_pipeline_graph():
    """
    Constructs and compiles the ETL LangGraph pipeline for Neo4j Knowledge Graph.

    Graph Topology (Mini-Batch processing):
    [START] -> extract_knowledge -> embed_entities -> store_neo4j -> [END]
    """
    workflow = StateGraph(PipelineState)

    # Register Nodes
    workflow.add_node("extract_knowledge", extract_knowledge_node)
    workflow.add_node("embed_entities", embed_entities_node)
    workflow.add_node("store_neo4j", store_neo4j_node)

    # Register Edges
    workflow.add_edge(START, "extract_knowledge")
    workflow.add_edge("extract_knowledge", "embed_entities")
    workflow.add_edge("embed_entities", "store_neo4j")
    workflow.add_edge("store_neo4j", END)

    app = workflow.compile()
    return app


# Pre-compiled executable graph instance for chunk processing
app = build_pipeline_graph()


def run_pipeline(
    rss_feeds: Optional[List[Dict[str, str]]] = None,
    max_articles_per_feed: int = config.MAX_ARTICLES_PER_FEED,
    total_max_articles: int = config.TOTAL_MAX_ARTICLES,
    neo4j_uri: str = config.NEO4J_URI,
    neo4j_username: str = config.NEO4J_USERNAME,
    neo4j_password: str = config.NEO4J_PASSWORD,
    neo4j_database: str = config.NEO4J_DATABASE
) -> Dict[str, Any]:
    """
    Helper function to initialize state, fetch RSS, and invoke the LangGraph pipeline in 10 mini-batches.
    """
    initial_state: PipelineState = {
        "rss_feeds": rss_feeds or config.RSS_FEEDS,
        "max_articles_per_feed": max_articles_per_feed,
        "total_max_articles": total_max_articles,
        "raw_articles": [],
        "extracted_knowledge": [],
        "total_entities_extracted": 0,
        "neo4j_uri": neo4j_uri,
        "neo4j_username": neo4j_username,
        "neo4j_password": neo4j_password,
        "neo4j_database": neo4j_database,
        "neo4j_nodes_created": 0,
        "neo4j_relationships_created": 0,
        "neo4j_status": "PENDING",
        "errors": []
    }

    logger.info("Step 1: Fetching all RSS articles...")
    fetch_result = fetch_rss_node(initial_state)
    all_raw_articles = fetch_result.get("raw_articles", [])
    
    if not all_raw_articles:
        logger.warning("No articles fetched. Aborting pipeline.")
        return initial_state

    # Chunk the articles into batches of 10
    chunk_size = 10
    chunks = [all_raw_articles[i:i + chunk_size] for i in range(0, len(all_raw_articles), chunk_size)]

    total_nodes = 0
    total_rels = 0
    total_entities = 0
    all_errors = fetch_result.get("errors", [])
    
    logger.info(f"Step 2: Processing {len(all_raw_articles)} articles across {len(chunks)} mini-batches (size {chunk_size})...")

    for i, chunk in enumerate(chunks, 1):
        if not chunk:
            continue
            
        logger.info(f"--- Starting Mini-Batch {i}/{len(chunks)} ({len(chunk)} articles) ---")
        
        batch_state = initial_state.copy()
        batch_state["raw_articles"] = chunk
        batch_state["extracted_knowledge"] = []
        batch_state["errors"] = []
        
        batch_result = app.invoke(batch_state)
        
        total_nodes += batch_result.get("neo4j_nodes_created", 0)
        total_rels += batch_result.get("neo4j_relationships_created", 0)
        total_entities += batch_result.get("total_entities_extracted", 0)
        all_errors.extend(batch_result.get("errors", []))
        
        logger.info(f"--- Finished Mini-Batch {i}/{len(chunks)} ---")

    final_state = initial_state.copy()
    final_state["raw_articles"] = all_raw_articles
    final_state["total_entities_extracted"] = total_entities
    final_state["neo4j_nodes_created"] = total_nodes
    final_state["neo4j_relationships_created"] = total_rels
    final_state["neo4j_status"] = "SUCCESS" if not all_errors else "COMPLETED_WITH_ERRORS"
    final_state["errors"] = all_errors

    return final_state


if __name__ == "__main__":
    result = run_pipeline()
    print("\n--- Pipeline Run Summary ---")
    print(f"Articles Fetched:           {len(result.get('raw_articles', []))}")
    print(f"Extracted Entities:         {result.get('total_entities_extracted')}")
    print(f"Neo4j Status:               {result.get('neo4j_status')}")
    print(f"Neo4j Nodes Created:        {result.get('neo4j_nodes_created')}")
    print(f"Neo4j Relationships Created:{result.get('neo4j_relationships_created')}")
    print(f"Errors Encountered:         {len(result.get('errors', []))}")
