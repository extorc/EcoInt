"""
Main Entry Point for running the LangGraph RSS to Qdrant Ingestion & Vector Search Pipeline.
"""

import sys
import logging
from graph import run_pipeline
import config

def main():
    print("==================================================")
    print("  LangGraph RSS -> Qdrant Vector DB Pipeline      ")
    print("==================================================")
    print(f"Configured Sources ({len(config.RSS_FEEDS)} feeds):")
    for feed in config.RSS_FEEDS:
        print(f"  - [{feed['category']}] {feed['name']}")
    print(f"Embedding Model: {config.EMBEDDING_MODEL_NAME}")
    print(f"Qdrant Storage Path: {config.QDRANT_STORAGE_PATH}")
    print("--------------------------------------------------\n")

    # Allow custom search query via command-line arguments if provided
    query = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_SEARCH_QUERY

    # Execute graph pipeline
    result_state = run_pipeline(search_query=query)

    # Extract state variables
    indexed_count = result_state.get("indexed_count", 0)
    collection_name = result_state.get("qdrant_collection_name", "")
    search_query = result_state.get("search_query", "")
    search_results = result_state.get("search_results", [])
    errors = result_state.get("errors", [])

    print("\n==================================================")
    print("           Vector Ingestion & Search              ")
    print("==================================================")
    print(f" Status: {' SUCCESS' if indexed_count > 0 else ' FAILED'}")
    print(f" Qdrant Collection: {collection_name}")
    print(f" Articles Indexed into Vectors: {indexed_count}")
    print(f" Total Pipeline Errors: {len(errors)}")

    print(f"\n--- Similarity Search Query: '{search_query}' ---")
    if search_results:
        for idx, res in enumerate(search_results, 1):
            score = res.get("score")
            title = res.get("title")
            source = res.get("source")
            link = res.get("link")
            summary = res.get("summary", "")[:120]
            print(f"\n[{idx}] Match Score: {score:.4f}")
            print(f"    Source:  {source}")
            print(f"    Title:   {title}")
            print(f"    Link:    {link}")
            print(f"    Snippet: {summary}...")
    else:
        print("No matching articles found.")

    if errors:
        print("\n--- Logged Warnings/Errors ---")
        for err in errors:
            print(f" - {err}")

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()