"""
ETL Ingestion Pipeline Nodes for LangGraph Knowledge Graph.
Contains modular state processing nodes:
- fetch_rss: Fetches and parses RSS feeds.
- extract_knowledge: Uses LLM to extract entities (with descriptions) and relationships.
- embed_entities: Generates 768-dim vector embeddings for each entity description via Gemini text-embedding-004.
- store_neo4j: Resolves entities semantically (cosine similarity >= 0.92 = MERGE, else CREATE),
               then stores entities and relationships into the Neo4j Knowledge Graph.
"""

from nodes.fetch_rss import fetch_rss_node
from nodes.extract_knowledge import extract_knowledge_node
from nodes.embed_entities import embed_entities_node
from nodes.store_neo4j import store_neo4j_node

__all__ = [
    "fetch_rss_node",
    "extract_knowledge_node",
    "embed_entities_node",
    "store_neo4j_node"
]
