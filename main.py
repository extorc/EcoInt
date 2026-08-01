"""
Main Entry Point for running the LangGraph RSS -> LLM Entity Extraction -> Neo4j Knowledge Graph Pipeline.
"""

import sys
import logging
from graph import run_pipeline
import config

RED = "\033[91m"
BOLD_RED = "\033[1;91m"
RESET = "\033[0m"


def main():
    print("==================================================")
    print("  LangGraph RSS -> LLM -> Neo4j Knowledge Graph  ")
    print("==================================================")
    print(f"Configured Sources ({len(config.RSS_FEEDS)} feeds):")
    for feed in config.RSS_FEEDS:
        print(f"  - [{feed['category']}] {feed['name']}")
    print(f"Neo4j URI: {config.NEO4J_URI}")
    api_key_found = bool(config.GEMINI_API_KEY)
    print(f"Gemini API Key Detected: {'Yes' if api_key_found else 'No'}")
    print("--------------------------------------------------\n")

    # Execute graph pipeline
    result_state = run_pipeline()

    # Extract state variables
    raw_articles = result_state.get("raw_articles", [])
    extracted_knowledge = result_state.get("extracted_knowledge", [])
    total_entities = result_state.get("total_entities_extracted", 0)
    neo4j_status = result_state.get("neo4j_status", "UNKNOWN")
    nodes_created = result_state.get("neo4j_nodes_created", 0)
    rels_created = result_state.get("neo4j_relationships_created", 0)
    errors = result_state.get("errors", [])

    print("\n==================================================")
    print("          Knowledge Extraction & Neo4j            ")
    print("==================================================")
    print(f" Articles Processed:      {len(raw_articles)}")
    print(f" Total Entities Extracted: {total_entities}")
    print(f" Neo4j Database Status:    {neo4j_status}")
    print(f" Neo4j Nodes Created:      {nodes_created}")
    print(f" Neo4j Links Created:      {rels_created}")
    print(f" Total Pipeline Warnings:  {len(errors)}")

    if extracted_knowledge:
        print("\n--- Extracted Knowledge Graph Sample (Top Articles) ---")
        for idx, item in enumerate(extracted_knowledge[:3], 1):
            print(f"\n[{idx}] Title: {item.get('title')}")
            print(f"    Source: {item.get('source')} | Category: {item.get('category')}")

            entities = item.get("entities", [])
            ent_str = ", ".join([f"{e.get('name')} ({e.get('type')})" for e in entities[:5]])
            print(f"    Entities ({len(entities)}): {ent_str}...")

    if errors:
        print(f"\n{BOLD_RED}--- Logged Warnings/Errors ---{RESET}")
        for err in errors:
            print(f"{RED} - {err}{RESET}")

    print("\nPipeline execution completed.")


if __name__ == "__main__":
    main()