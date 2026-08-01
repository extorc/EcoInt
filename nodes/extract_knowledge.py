import os
import json
import logging
import re
import time
from typing import Dict, Any, List

import config

logger = logging.getLogger(__name__)

from services.llm_client import extract_with_nemotron


def extract_knowledge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node: Extracts entities and relationships from raw RSS articles strictly using LLM.

    State Input:
        - raw_articles: List of parsed RSS articles.
        - errors: Existing errors list.

    State Output:
        - extracted_knowledge: List of extracted graph structures per article.
        - total_entities_extracted: Total count of entities extracted across all articles.
        - errors: Updated errors list.
    """
    logger.info("=== Starting Node: extract_knowledge (LLM Only) ===")

    raw_articles: List[Dict[str, Any]] = state.get("raw_articles", [])
    errors: List[str] = state.get("errors", [])

    nvidia_api_key = getattr(config, "NVIDIA_API_KEY", "")

    extracted_knowledge = []
    total_entities = 0

    if not raw_articles:
        msg = "No raw articles available for knowledge extraction."
        logger.warning(msg)
        errors.append(msg)
        return {
            "extracted_knowledge": [],
            "total_entities_extracted": 0,
            "errors": errors
        }

    if not nvidia_api_key:
        error_msg = "NVIDIA API key is missing!"
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            "extracted_knowledge": [],
            "total_entities_extracted": 0,
            "errors": errors
        }

    for idx, article in enumerate(raw_articles, 1):
        title = article.get("title", "")
        summary = article.get("rss_summary", "")
        full_text = f"Title: {title}\nSummary: {summary}"

        logger.info(f"[{idx}/{len(raw_articles)}] Extracting knowledge via LLM for article: '{title[:60]}...'")

        start_time = time.time()

        try:
            extraction_res = extract_with_nemotron(full_text, nvidia_api_key)
            entities = extraction_res.get("entities", [])
            model_used = extraction_res.get("model_used", "")

            total_entities += len(entities)

            extracted_knowledge.append({
                "article_id": article.get("article_id"),
                "title": title,
                "link": article.get("link"),
                "source": article.get("source"),
                "category": article.get("category"),
                "published_rss": article.get("published_rss"),
                "entities": entities
            })
            logger.info(f" LLM Extraction successful ({model_used}) for article '{article.get('article_id')}': {len(entities)} entities.")

        except Exception as e:
            err = f"LLM extraction failed for article {article.get('article_id')}: {e}"
            logger.error(err, exc_info=True)
            errors.append(err)

        # Rate limiting: 40 requests per minute (1.5 seconds per request)
        elapsed = time.time() - start_time
        sleep_time = 1.5 - elapsed
        if sleep_time > 0 and idx < len(raw_articles):
            time.sleep(sleep_time)

    logger.info(f"=== Node extract_knowledge Complete: Processed {len(extracted_knowledge)} articles via LLM | Extracted {total_entities} entities. ===")

    return {
        "extracted_knowledge": extracted_knowledge,
        "total_entities_extracted": total_entities,
        "errors": errors
    }
