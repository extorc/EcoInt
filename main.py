"""
Main Entry Point for running the Economic Intelligence Knowledge Extraction Pipeline using LangGraph, GLiNER, spaCy, RapidFuzz, BGE Embeddings, and Neo4j.
"""

import sys
import logging
from graph import run_pipeline
import config

RED = "\033[91m"
BOLD_RED = "\033[1;91m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"\n{CYAN}========================================================================={RESET}")
    print(f"{CYAN}  Economic Intelligence Platform - NLP Knowledge Graph Extraction Pipeline {RESET}")
    print(f"{CYAN}========================================================================={RESET}")
    print(f"Configured RSS Feeds ({len(config.RSS_FEEDS)} sources):")
    for feed in config.RSS_FEEDS:
        print(f"  - [{feed['category']}] {feed['name']}")
    print(f"\nModel Configuration:")
    print(f"  - NER Model:        {config.GLINER_MODEL} (GLiNER Zero-Shot)")
    print(f"  - Sentence Segmenter: {config.SPACY_MODEL} (spaCy)")
    print(f"  - Embedding Model:  {config.EMBEDDING_MODEL} (768-dim dense vectors)")
    print(f"  - Reranker Model:   {config.RERANKER_MODEL} (CrossEncoder Reranker)")
    print(f"  - Neo4j Instance:   {config.NEO4J_URI}")
    print("-------------------------------------------------------------------------\n")

    # Execute LangGraph Pipeline
    result_state = run_pipeline()

    documents = result_state.get("documents", [])
    sentences = result_state.get("sentences", [])
    raw_entities = result_state.get("raw_entities", [])
    resolved_entities = result_state.get("resolved_entities", [])
    embedded_entities = result_state.get("embedded_entities", [])
    relationships = result_state.get("relationships", [])
    neo4j_status = result_state.get("neo4j_status", "UNKNOWN")
    nodes_created = result_state.get("neo4j_nodes_created", 0)
    rels_created = result_state.get("neo4j_relationships_created", 0)
    errors = result_state.get("errors", [])

    print(f"\n{GREEN}========================================================================={RESET}")
    print(f"{GREEN}                     Pipeline Execution Results Summary                   {RESET}")
    print(f"{GREEN}========================================================================={RESET}")
    print(f" Documents Processed:      {len(documents)}")
    print(f" Sentences Segmented:     {len(sentences)}")
    print(f" Entities Extracted (NER): {len(raw_entities)}")
    print(f" Resolved Entities:       {len(resolved_entities)}")
    print(f" Entities Vectorized:     {len(embedded_entities)}")
    print(f" Relationships Extracted: {len(relationships)}")
    print(f" Neo4j Database Status:   {neo4j_status}")
    print(f" Neo4j Nodes Ingested:    {nodes_created}")
    print(f" Neo4j Edges Ingested:    {rels_created}")
    print(f" Pipeline Warnings/Errors:{len(errors)}")

    if resolved_entities:
        print(f"\n{CYAN}--- Sample Resolved Entities ---{RESET}")
        for idx, ent in enumerate(resolved_entities[:5], 1):
            has_vec = "Yes (768d)" if ent.get("embedding") else "No"
            desc_snip = ent.get("description", "")[:60]
            print(f"  [{idx}] {ent.get('name')} | Type: {ent.get('entity_type')} | Vector: {has_vec}")
            print(f"      Description: \"{desc_snip}...\"")

    if relationships:
        print(f"\n{CYAN}--- Sample Extracted Relationships ---{RESET}")
        for idx, rel in enumerate(relationships[:5], 1):
            print(f"  [{idx}] ({rel.get('source_entity_name')}) -[{rel.get('relationship_type')}]-> ({rel.get('target_entity_name')})")
            why_desc = rel.get('description', '')[:100]
            sentence_text = rel.get('supporting_sentence', '')[:80]
            try:
                print(f"      Why Connected: \"{why_desc}...\"")
                print(f"      Confidence: {rel.get('confidence')} | Context: \"{sentence_text}...\"")
            except UnicodeEncodeError:
                safe_desc = why_desc.encode('ascii', 'ignore').decode('ascii')
                safe_text = sentence_text.encode('ascii', 'ignore').decode('ascii')
                print(f"      Why Connected: \"{safe_desc}...\"")
                print(f"      Confidence: {rel.get('confidence')} | Context: \"{safe_text}...\"")

    if errors:
        print(f"\n{BOLD_RED}--- Pipeline Warnings & Errors ---{RESET}")
        for err in errors:
            print(f"{RED} - {err}{RESET}")

    print(f"\n{GREEN}Pipeline execution finished.{RESET}")


if __name__ == "__main__":
    main()