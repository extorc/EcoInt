import sys
import argparse
import json
import logging
from openai import OpenAI
from neo4j import GraphDatabase
import config
from nodes.embed_entities import get_embedding_model

from services.semantic_search import generate_description, get_embedding, search_node
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def classify_query(query: str, client: OpenAI) -> dict:
    prompt = f"""
You are an intent classifier for a Knowledge Graph Retrieval System.
Analyze the user's question and determine the query type and extract the entities mentioned.

Query Types:
1. TWO_ENTITIES: The user is asking about the relation, effect, or connection between exactly two distinct entities.
2. ONE_ENTITY_EFFECTS: The user is asking about the downstream effects of one entity.
3. ONE_ENTITY_CAUSES: The user is asking what caused one entity.
4. ONE_ENTITY_CONTEXT: The user is asking for general information about one entity.

User Query: "{query}"

Output ONLY a raw JSON object with this schema:
{{
  "type": "TWO_ENTITIES" | "ONE_ENTITY_EFFECTS" | "ONE_ENTITY_CAUSES" | "ONE_ENTITY_CONTEXT",
  "entities": ["Entity 1", "Entity 2"] // List of 1 or 2 entities depending on the type
}}
"""
    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200
    )
    raw_text = response.choices[0].message.content.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    return json.loads(raw_text.strip())

def get_connecting_articles(tx, node1_name: str, node2_name: str):
    # Try finding the shortest path up to 6 hops (which means up to 3 intermediate articles)
    query = """
    MATCH (e1:Entity {name: $n1}), (e2:Entity {name: $n2})
    MATCH p = shortestPath((e1)-[*1..6]-(e2))
    WITH nodes(p) AS path_nodes
    UNWIND path_nodes AS n
    WITH n WHERE 'Article' IN labels(n)
    RETURN DISTINCT n.title AS title, n.url AS url, n.source AS source, n.published_rss AS date
    LIMIT 15
    """
    result = tx.run(query, n1=node1_name, n2=node2_name)
    return [record.data() for record in result]

def scrape_article(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        # Limit to first 4000 chars to avoid blowing up context window
        return text[:4000] + ("..." if len(text) > 4000 else "")
    except Exception as e:
        return f"Failed to fetch content: {e}"

def generate_final_answer(query: str, articles: list, client: OpenAI) -> str:
    context = ""
    for i, a in enumerate(articles, 1):
        logging.info(f"Scraping content for article {i}...")
        content = scrape_article(a.get('url'))
        context += f"Article {i}:\nTitle: {a.get('title')}\nSource: {a.get('source')}\nDate: {a.get('date')}\nURL: {a.get('url')}\nContent Snippet: {content}\n\n"
        
    prompt = f"""
You are a financial and economic analysis assistant.
Based ONLY on the following news article contents connecting these entities, answer the user's query about their relationship. 
If they do not appear in the same article, explain their indirect connection based on the provided bridging articles.
If the article titles do not contain enough information to deduce a confident answer, summarize what the articles are generally about regarding these entities.

Articles Context:
{context}

User Query: {query}
"""
    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Query Router")
    parser.add_argument("query", type=str, help="Natural language query (e.g. 'How does X affect Y?')")
    args = parser.parse_args()

    if not getattr(config, "NVIDIA_API_KEY", None):
        logging.error("NVIDIA_API_KEY is not set. Please set it in your environment or .env file.")
        sys.exit(1)
        
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=config.NVIDIA_API_KEY
    )

    logging.info("Analyzing query intent and extracting entities...")
    try:
        intent = classify_query(args.query, client)
        q_type = intent.get("type")
        entities = intent.get("entities", [])
        logging.info(f"Intent classified as: {q_type}")
        logging.info(f"Entities extracted: {entities}")
    except Exception as e:
        logging.error(f"Failed to parse query intent: {e}")
        sys.exit(1)

    if q_type == "TWO_ENTITIES":
        if len(entities) != 2:
            logging.error("TWO_ENTITIES query requires exactly 2 entities to be extracted.")
            sys.exit(1)
            
        uri = getattr(config, "NEO4J_URI", "bolt://localhost:7687")
        username = getattr(config, "NEO4J_USERNAME", "neo4j")
        password = getattr(config, "NEO4J_PASSWORD", "password")
        database = getattr(config, "NEO4J_DATABASE", "neo4j")

        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            driver.verify_connectivity()
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            sys.exit(1)

        with driver.session(database=database) as session:
            resolved_nodes = []
            for ent in entities:
                logging.info(f"Resolving entity: '{ent}'...")
                desc = generate_description(ent)
                emb = get_embedding(desc)
                results = session.execute_read(search_node, emb)
                if not results:
                    logging.error(f"Could not find any match in DB for entity: {ent}")
                    sys.exit(1)
                best_match = results[0]['props']['name']
                logging.info(f"Resolved '{ent}' to Graph Node: '{best_match}' (score: {results[0]['score']:.4f})")
                resolved_nodes.append(best_match)

            logging.info(f"Finding path and connecting articles between '{resolved_nodes[0]}' and '{resolved_nodes[1]}'...")
            connecting_articles = session.execute_read(get_connecting_articles, resolved_nodes[0], resolved_nodes[1])
            
            if not connecting_articles:
                logging.warning("No connecting path found between these two entities in the graph (up to 6 hops away).")
                sys.exit(0)
                
            logging.info(f"Found {len(connecting_articles)} connecting articles. Generating answer...")
            answer = generate_final_answer(args.query, connecting_articles, client)
            
            print("\n" + "="*70)
            print("FINAL ANSWER")
            print("="*70)
            print(answer)
            print("\n" + "-"*70)
            print("Sources used (Path articles):")
            for a in connecting_articles:
                print(f" - {a.get('title')} ({a.get('url')})")
            print("="*70 + "\n")
                
        driver.close()

    elif q_type in ["ONE_ENTITY_EFFECTS", "ONE_ENTITY_CAUSES", "ONE_ENTITY_CONTEXT"]:
        logging.info(f"Endpoint for query type '{q_type}' is correctly routed but currently unimplemented. Stopping execution.")
    else:
        logging.error(f"Unknown query type: {q_type}")

if __name__ == "__main__":
    main()
