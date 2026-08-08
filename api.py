import sys
import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
import urllib.parse
import config
from graph import run_pipeline

app = FastAPI(title="EcoInt API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_driver():
    uri = getattr(config, "NEO4J_URI", "bolt://localhost:7687")
    username = getattr(config, "NEO4J_USERNAME", "neo4j")
    password = getattr(config, "NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(username, password))

@app.get("/api/nodes")
def get_graph():
    nodes_query = """
    MATCH (n)
    WHERE 'Entity' IN labels(n) OR 'Article' IN labels(n)
    RETURN toString(id(n)) AS id, 
           COALESCE(n.name, n.title, 'Unknown') AS label, 
           labels(n)[0] AS group, 
           COALESCE(n.description, n.url, '') AS description
    """
    edges_query = """
    MATCH (s)-[r:IN]->(t)
    RETURN toString(id(s)) AS source, toString(id(t)) AS target
    """
    try:
        driver = get_driver()
        with driver.session(database=getattr(config, "NEO4J_DATABASE", "neo4j")) as session:
            nodes_result = session.run(nodes_query)
            nodes = [record.data() for record in nodes_result]
            
            edges_result = session.run(edges_query)
            edges = [record.data() for record in edges_result]
        driver.close()
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/nodes/delete_bulk")
def delete_nodes_bulk(payload: dict):
    node_ids = payload.get("nodes", [])
    if not node_ids:
        return {"message": "No nodes provided."}
        
    query = """
    MATCH (n)
    WHERE toString(id(n)) IN $node_ids
    DETACH DELETE n
    """
    try:
        driver = get_driver()
        with driver.session(database=getattr(config, "NEO4J_DATABASE", "neo4j")) as session:
            result = session.run(query, node_ids=node_ids)
            summary = result.consume()
        driver.close()
        return {"message": f"Successfully deleted {summary.counters.nodes_deleted} nodes."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/nodes/merge_bulk")
def merge_nodes_bulk(payload: dict):
    node_ids = payload.get("nodes", [])
    merged_name = payload.get("merged_name", "Merged Entity")
    
    if len(node_ids) < 2:
        return {"message": "Need at least 2 nodes to merge."}
        
    query = """
    MATCH (n)
    WHERE toString(id(n)) IN $node_ids
    WITH collect(n) AS nodes
    WITH head(nodes) AS primary, tail(nodes) AS others
    SET primary.name = $merged_name
    WITH primary, others
    UNWIND others AS other
    OPTIONAL MATCH (other)-[:IN]->(a:Article)
    WITH primary, other, collect(a) AS articles
    FOREACH (art IN articles | 
        MERGE (primary)-[:IN]->(art)
    )
    DETACH DELETE other
    """
    try:
        driver = get_driver()
        with driver.session(database=getattr(config, "NEO4J_DATABASE", "neo4j")) as session:
            result = session.run(query, node_ids=node_ids, merged_name=merged_name)
            summary = result.consume()
        driver.close()
        return {"message": f"Successfully merged {len(node_ids)} nodes into '{merged_name}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/{entity_name}")
def ingest_entity(entity_name: str, background_tasks: BackgroundTasks):
    query = urllib.parse.quote(entity_name)
    search_rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    custom_feed = [{"name": f"Google News Search: {entity_name}", "url": search_rss_url, "category": "Targeted"}]
    
    def run_ingestion():
        try:
            print(f"Starting background ingestion for {entity_name}")
            run_pipeline(
                rss_feeds=custom_feed,
                max_articles_per_feed=10,
                total_max_articles=10
            )
            print(f"Finished background ingestion for {entity_name}")
        except Exception as e:
            print(f"Ingestion failed for {entity_name}: {e}")

    background_tasks.add_task(run_ingestion)
    return {"message": f"Ingestion started for '{entity_name}'. Check server logs for progress."}

@app.delete("/api/nodes/{entity_name}")
def delete_node(entity_name: str):
    query = """
    MATCH (e:Entity {name: $name})
    DETACH DELETE e
    """
    try:
        driver = get_driver()
        with driver.session(database=getattr(config, "NEO4J_DATABASE", "neo4j")) as session:
            result = session.run(query, name=entity_name)
            summary = result.consume()
        driver.close()
        
        if summary.counters.nodes_deleted == 0:
            raise HTTPException(status_code=404, detail="Node not found.")
            
        return {"message": f"Successfully deleted '{entity_name}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
def get_neo4j_config():
    return {
        "uri": getattr(config, "NEO4J_URI", "bolt://localhost:7687"),
        "user": getattr(config, "NEO4J_USERNAME", "neo4j"),
        "password": getattr(config, "NEO4J_PASSWORD", "password")
    }

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
