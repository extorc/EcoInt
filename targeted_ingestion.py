import sys
from neo4j import GraphDatabase
import urllib.parse
from graph import run_pipeline
import config

def get_all_entities(tx):
    query = """
    MATCH (e:Entity)
    RETURN e.name AS name, e.type AS type, e.description AS description
    ORDER BY e.name ASC
    """
    result = tx.run(query)
    return [record.data() for record in result]

def main():
    uri = getattr(config, "NEO4J_URI", "bolt://localhost:7687")
    username = getattr(config, "NEO4J_USERNAME", "neo4j")
    password = getattr(config, "NEO4J_PASSWORD", "password")
    database = getattr(config, "NEO4J_DATABASE", "neo4j")
    
    print("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)
        
    with driver.session(database=database) as session:
        entities = session.execute_read(get_all_entities)
        
    if not entities:
        print("No entities found in the graph.")
        driver.close()
        sys.exit(0)
        
    print(f"Found {len(entities)} entities in the graph.")
    
    for i, ent in enumerate(entities, 1):
        name = ent.get("name")
        ent_type = ent.get("type")
        desc = ent.get("description")
        
        print("\n" + "="*70)
        print(f"ENTITY {i}/{len(entities)}")
        print(f"Name:        {name}")
        print(f"Type:        {ent_type}")
        print(f"Description: {desc}")
        print("="*70)
        
        while True:
            choice = input(f"Fetch up to 10 new articles for '{name}'? (y/n/q to quit): ").strip().lower()
            if choice in ['y', 'n', 'q']:
                break
            print("Please enter 'y', 'n', or 'q'.")
            
        if choice == 'q':
            print("Exiting targeted ingestion...")
            break
            
        if choice == 'y':
            # Create a Google News RSS search URL to target this specific entity
            query = urllib.parse.quote(name)
            search_rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            custom_feed = [{"name": f"Google News Search: {name}", "url": search_rss_url, "category": ent_type}]
            
            print(f"\n[+] Running ingestion pipeline for '{name}'...")
            
            try:
                # We reuse the existing graph.py run_pipeline but override the feeds and limits
                result = run_pipeline(
                    rss_feeds=custom_feed,
                    max_articles_per_feed=10,
                    total_max_articles=10
                )
                
                print(f"[-] Finished ingestion for '{name}'.")
                print(f"    Articles Fetched: {len(result.get('raw_articles', []))}")
                print(f"    Neo4j Nodes Created: {result.get('neo4j_nodes_created')}")
                print(f"    Neo4j Relationships Created: {result.get('neo4j_relationships_created')}")
                
            except Exception as e:
                print(f"[!] Error during pipeline execution for '{name}': {e}")
        else:
            print(f"Skipping '{name}'.")

    driver.close()
    print("\nTargeted ingestion complete.")

if __name__ == "__main__":
    main()
