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


