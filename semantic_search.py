import sys
import argparse
from openai import OpenAI
from neo4j import GraphDatabase
import config
from nodes.embed_entities import get_embedding_model

def generate_description(prompt: str) -> str:
    print(f"Generating description for: '{prompt}' using Llama on NVIDIA...")
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=config.NVIDIA_API_KEY
        )
        full_prompt = f"Provide a single-sentence, absolute, objective, encyclopedia-style definition for this entity/concept:\n\n{prompt}"
        response = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        desc = response.choices[0].message.content.strip()
        print(f"Generated Description: {desc}")
        return desc
    except Exception as e:
        print(f"Failed to generate description: {e}")
        sys.exit(1)

def get_embedding(text: str) -> list[float]:
    print("Generating embedding using FastEmbed...")
    model = get_embedding_model()
    embeddings_gen = model.embed([text])
    embeddings = [list(map(float, emb)) for emb in embeddings_gen]
    return embeddings[0]

def search_node(tx, embedding: list[float]):
    query = """
    CALL db.index.vector.queryNodes('entity_embedding_index', 10, $embedding)
    YIELD node, score
    RETURN properties(node) AS props, score, labels(node) AS labels
    ORDER BY score DESC
    """
    result = tx.run(query, embedding=embedding)
    return [record.data() for record in result]

def main():
    parser = argparse.ArgumentParser(description="Find a node in Neo4j via semantic embedding search.")
    parser.add_argument("prompt", type=str, help="Prompt to define the start node")
    
    args = parser.parse_args()
    
    if not getattr(config, "NVIDIA_API_KEY", None):
        print("NVIDIA_API_KEY is not set. Please set it in your environment or .env file.")
        sys.exit(1)
        
    uri = getattr(config, "NEO4J_URI", "bolt://localhost:7687")
    username = getattr(config, "NEO4J_USERNAME", "neo4j")
    password = getattr(config, "NEO4J_PASSWORD", "password")
    database = getattr(config, "NEO4J_DATABASE", "neo4j")
    
    print(f"Connecting to Neo4j at {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)
        
    with driver.session(database=database) as session:
        print(f"\n======================================")
        print(f" SEARCHING FOR: '{args.prompt}'")
        print(f"======================================")
        
        desc = generate_description(args.prompt)
        emb = get_embedding(desc)
        
        results = session.execute_read(search_node, emb)
        if not results:
            print("❌ No similar nodes found in Neo4j.")
        else:
            for i, r in enumerate(results, 1):
                print(f"\nMatch #{i} (Similarity Score: {r.get('score', 0):.4f}):")
                labels = r.get("labels", [])
                print(f"  • Labels: {labels}")
                props = r.get("props", {})
                
                # Exclude the embedding from output
                if "embedding" in props:
                    del props["embedding"]
                    
                for k, v in props.items():
                    print(f"  • {k.capitalize()}: {v}")

    driver.close()

if __name__ == "__main__":
    main()
