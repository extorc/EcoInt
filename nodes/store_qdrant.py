import logging
import uuid
from typing import Dict, Any, List
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import config

logger = logging.getLogger(__name__)

# Global singleton or cached embedding model to avoid re-loading on each node call
_EMBEDDING_MODEL = None

def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info(f"Loading FastEmbed model '{config.EMBEDDING_MODEL_NAME}'...")
        _EMBEDDING_MODEL = TextEmbedding(model_name=config.EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def store_qdrant_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Embeds raw articles, upserts vectors into Qdrant Vector DB, 
    and executes an immediate similarity search.
    
    State Input:
        - raw_articles: List of articles fetched from RSS feeds.
        - search_query (optional): Custom search query text.
        - qdrant_path (optional): Qdrant storage path or ':memory:'.
        - errors: List of previous error messages.
        
    State Output:
        - indexed_count: Number of vectors indexed into Qdrant.
        - qdrant_collection_name: Name of Qdrant collection used.
        - search_query: Query text evaluated.
        - search_results: List of top similarity search hits with scores.
        - errors: List of error messages.
    """
    logger.info("=== Starting Node: store_qdrant ===")
    
    raw_articles: List[Dict[str, Any]] = state.get("raw_articles", [])
    search_query: str = state.get("search_query", config.DEFAULT_SEARCH_QUERY)
    qdrant_path: str = state.get("qdrant_path", config.QDRANT_STORAGE_PATH)
    collection_name: str = state.get("qdrant_collection_name", config.QDRANT_COLLECTION_NAME)
    errors: List[str] = state.get("errors", [])
    
    if not raw_articles:
        msg = "No raw articles available to embed and store in Qdrant."
        logger.warning(msg)
        errors.append(msg)
        return {
            "indexed_count": 0,
            "qdrant_collection_name": collection_name,
            "search_query": search_query,
            "search_results": [],
            "errors": errors
        }
        
    try:
        # Load embedding model
        model = get_embedding_model()
        
        # Prepare text content for embedding: Title + RSS Summary
        texts_to_embed = [
            f"Title: {item.get('title', '')}. Summary: {item.get('rss_summary', '')}"
            for item in raw_articles
        ]
        
        logger.info(f"Generating dense vector embeddings for {len(texts_to_embed)} articles...")
        embeddings = list(model.embed(texts_to_embed))
        vector_dim = len(embeddings[0])
        
        # Initialize Qdrant local client (persistent or in-memory)
        logger.info(f"Connecting to Qdrant Vector DB at path '{qdrant_path}'...")
        client = QdrantClient(path=qdrant_path)
        
        # Re-create or ensure collection exists
        logger.info(f"Creating/updating collection '{collection_name}' (vector dimension: {vector_dim})...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        
        # Build Qdrant PointStruct items
        points = []
        for idx, (emb, item) in enumerate(zip(embeddings, raw_articles)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, item.get("link", str(idx))))
            payload = {
                "article_id": item.get("article_id", ""),
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "source": item.get("source", ""),
                "category": item.get("category", ""),
                "published_rss": item.get("published_rss", ""),
                "rss_summary": item.get("rss_summary", ""),
                "fetched_at": item.get("fetched_at", "")
            }
            points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))
            
        # Upsert vectors into Qdrant
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f" Successfully indexed {len(points)} article vectors into Qdrant collection '{collection_name}'.")
        
        # Perform similarity search query
        logger.info(f"Running similarity search query: '{search_query}'...")
        query_embedding = list(model.embed([search_query]))[0].tolist()
        
        search_res = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=5
        )
        
        search_hits = []
        for hit in search_res.points:
            search_hits.append({
                "score": float(round(hit.score, 4)),
                "title": hit.payload.get("title"),
                "source": hit.payload.get("source"),
                "category": hit.payload.get("category"),
                "link": hit.payload.get("link"),
                "summary": hit.payload.get("rss_summary")
            })
            
        logger.info(f" Found {len(search_hits)} relevant matches for query '{search_query}'. Top score: {search_hits[0]['score'] if search_hits else 'N/A'}")
        
    except Exception as e:
        error_msg = f"Failed in Qdrant store & search node: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
        search_hits = []
        points = []
        
    logger.info("=== Node store_qdrant Complete ===")
    
    return {
        "indexed_count": len(raw_articles),
        "qdrant_collection_name": collection_name,
        "search_query": search_query,
        "search_results": search_hits,
        "errors": errors
    }
