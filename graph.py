"""
LangGraph Ingestion Pipeline Graph Definition.
Defines state schema, nodes, edges, and compiles the executable graph.
Replaces temporary local file storage with Qdrant Vector DB storage & similarity search.
"""

import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from nodes import fetch_rss_node, store_qdrant_node
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
    qdrant_path: str
    qdrant_collection_name: str
    search_query: str
    search_results: List[Dict[str, Any]]
    indexed_count: int
    errors: List[str]


def build_pipeline_graph():
    """
    Constructs and compiles the ETL LangGraph pipeline.
    
    Graph Topology:
    [START] -> fetch_rss -> store_qdrant -> [END]
    """
    workflow = StateGraph(PipelineState)

    # Register Nodes
    workflow.add_node("fetch_rss", fetch_rss_node)
    workflow.add_node("store_qdrant", store_qdrant_node)

    # Register Edges
    workflow.add_edge(START, "fetch_rss")
    workflow.add_edge("fetch_rss", "store_qdrant")
    workflow.add_edge("store_qdrant", END)

    app = workflow.compile()
    return app


# Pre-compiled executable graph instance
app = build_pipeline_graph()


def run_pipeline(
    rss_feeds: Optional[List[Dict[str, str]]] = None,
    max_articles_per_feed: int = config.MAX_ARTICLES_PER_FEED,
    total_max_articles: int = config.TOTAL_MAX_ARTICLES,
    search_query: str = config.DEFAULT_SEARCH_QUERY,
    qdrant_path: str = config.QDRANT_STORAGE_PATH
) -> Dict[str, Any]:
    """
    Helper function to initialize state and invoke the LangGraph pipeline.
    """
    initial_state: PipelineState = {
        "rss_feeds": rss_feeds or config.RSS_FEEDS,
        "max_articles_per_feed": max_articles_per_feed,
        "total_max_articles": total_max_articles,
        "raw_articles": [],
        "qdrant_path": qdrant_path,
        "qdrant_collection_name": config.QDRANT_COLLECTION_NAME,
        "search_query": search_query,
        "search_results": [],
        "indexed_count": 0,
        "errors": []
    }
    
    logger.info("Initializing and invoking LangGraph RSS -> Qdrant Vector DB pipeline...")
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    result = run_pipeline()
    print("\n--- Pipeline Run Summary ---")
    print(f"Collection Name: {result.get('qdrant_collection_name')}")
    print(f"Indexed Count: {result.get('indexed_count')}")
    print(f"Search Query: '{result.get('search_query')}'")
    print(f"Search Results Count: {len(result.get('search_results', []))}")
    print(f"Errors Encountered: {len(result.get('errors', []))}")
