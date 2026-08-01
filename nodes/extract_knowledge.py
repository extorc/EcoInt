import os
import json
import logging
import re
from typing import Dict, Any, List

import config

logger = logging.getLogger(__name__)

# NVIDIA model candidates (fallbacks for when specific endpoints are down or restricted)
NVIDIA_MODELS = [
    "meta/llama-3.1-8b-instruct"
]

def _extract_with_nemotron(text: str, api_key: str) -> Dict[str, Any]:
    """Extract entities and relationships using NVIDIA's Nemotron API (via OpenAI client)."""
    from openai import OpenAI
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    prompt = f"""You are an expert financial and economic Knowledge Graph extraction system.
Analyze the following news text and extract key entities and direct relationships between them.

Text:
"{text}"

Return ONLY a raw JSON object (no markdown formatting, no code blocks) matching this schema:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "COMPANY|PERSON|GOVERNMENT|REGULATOR|SECTOR",
      "description": "A single sentence that provides an absolute, objective, encyclopedia-style definition of what this entity fundamentally is. Do NOT describe its role or actions in the context of the article. For example, if the entity is 'US', the description MUST be 'The United States of America is a country in North America.', regardless of what the US did in the news story."
    }}
  ]
}}
Rules:
- ENTITIES MUST BE CONCRETE ACTORS: Do NOT extract abstract concepts (e.g., 'Trade', 'Inflation', 'AI', 'Economy', 'Law'). Only extract tangible companies, people, governments, regulators, or sectors.
- The 'description' must define the entity in a universal, standalone way. Do NOT include what the entity is doing in this specific news story.
- Every entity MUST have a non-empty description.
"""

    last_exception = None
    for model_name in NVIDIA_MODELS:
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "stream": False
            }

            response = client.chat.completions.create(**kwargs)
            raw_text = response.choices[0].message.content.strip()

            # Clean markdown backticks if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)

            data = json.loads(raw_text)
            return {
                "entities": data.get("entities", []),
                "model_used": model_name
            }
        except Exception as e:
            logger.warning(f"NVIDIA model '{model_name}' failed ({e}). Trying next candidate...")
            last_exception = e

    raise last_exception if last_exception else RuntimeError("All NVIDIA model candidates failed.")


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

        try:
            extraction_res = _extract_with_nemotron(full_text, nvidia_api_key)
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

    logger.info(f"=== Node extract_knowledge Complete: Processed {len(extracted_knowledge)} articles via LLM | Extracted {total_entities} entities. ===")

    return {
        "extracted_knowledge": extracted_knowledge,
        "total_entities_extracted": total_entities,
        "errors": errors
    }
