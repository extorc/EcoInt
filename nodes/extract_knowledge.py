import os
import json
import logging
import re
from typing import Dict, Any, List

import config

logger = logging.getLogger(__name__)

# List of models with confirmed active quota and relaxed token limits
CANDIDATE_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it"
]


def _extract_with_gemini(text: str, api_key: str) -> Dict[str, Any]:
    """Extract entities and relationships using Google Gemini API with relaxed quota model retries."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    primary_model = getattr(config, "LLM_MODEL_NAME", "gemini-flash-lite-latest")
    models_to_try = [primary_model] + [m for m in CANDIDATE_MODELS if m != primary_model]

    prompt = f"""You are an expert financial and economic Knowledge Graph extraction system.
Analyze the following news text and extract key entities and direct relationships between them.

Text:
"{text}"

Return ONLY a raw JSON object (no markdown formatting, no code blocks) matching this schema:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "Organization|Country|Concept|Person|Location|Sector",
      "description": "A single sentence that provides an absolute, objective, encyclopedia-style definition of what this entity fundamentally is. Do NOT describe its role or actions in the context of the article. For example, if the entity is 'US', the description MUST be 'The United States of America is a country in North America.', regardless of what the US did in the news story."
    }}
  ],
  "relationships": [
    {{"source": "Entity A Name", "target": "Entity B Name", "relationship": "RELATION_TYPE"}}
  ]
}}
Rules:
- The 'description' must define the entity in a universal, standalone way. Do NOT include what the entity is doing in this specific news story. This is critical for vector similarity matching across different articles.
- Ensure relationship types are uppercase with underscores (e.g. IMPACTS, OPERATES_IN, REGULATES, ASSOCIATED_WITH).
- Every entity MUST have a non-empty description.
"""

    last_exception = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            # Clean markdown backticks if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)

            data = json.loads(raw_text)
            return {
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
                "model_used": model_name
            }
        except Exception as e:
            err_msg = str(e)
            if "Quota" in err_msg or "429" in err_msg:
                logger.warning(f"Model '{model_name}' hit quota limit (429). Automatically switching to next high-quota candidate model...")
            else:
                logger.warning(f"Model '{model_name}' failed ({e}). Trying next model candidate...")
            last_exception = e

    raise last_exception if last_exception else RuntimeError("All Gemini model candidates failed.")


def extract_knowledge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node: Extracts entities and relationships from raw RSS articles strictly using LLM.

    State Input:
        - raw_articles: List of parsed RSS articles.
        - errors: Existing errors list.

    State Output:
        - extracted_knowledge: List of extracted graph structures per article.
        - total_entities_extracted: Total count of entities extracted across all articles.
        - total_relationships_extracted: Total count of relationships extracted across all articles.
        - errors: Updated errors list.
    """
    logger.info("=== Starting Node: extract_knowledge (LLM Only) ===")

    raw_articles: List[Dict[str, Any]] = state.get("raw_articles", [])
    errors: List[str] = state.get("errors", [])

    gemini_key = getattr(config, "GEMINI_API_KEY", "") or config.get_gemini_api_key()

    extracted_knowledge = []
    total_entities = 0
    total_relationships = 0

    if not raw_articles:
        msg = "No raw articles available for knowledge extraction."
        logger.warning(msg)
        errors.append(msg)
        return {
            "extracted_knowledge": [],
            "total_entities_extracted": 0,
            "total_relationships_extracted": 0,
            "errors": errors
        }

    if not gemini_key:
        error_msg = "GEMINI_API_KEY is missing! LLM extraction requires a valid API key configured in environment variables or config.py."
        logger.error(error_msg)
        errors.append(error_msg)
        return {
            "extracted_knowledge": [],
            "total_entities_extracted": 0,
            "total_relationships_extracted": 0,
            "errors": errors
        }

    for idx, article in enumerate(raw_articles, 1):
        title = article.get("title", "")
        summary = article.get("rss_summary", "")
        full_text = f"Title: {title}\nSummary: {summary}"

        logger.info(f"[{idx}/{len(raw_articles)}] Extracting knowledge via LLM for article: '{title[:60]}...'")

        try:
            extraction_res = _extract_with_gemini(full_text, gemini_key)
            entities = extraction_res.get("entities", [])
            relationships = extraction_res.get("relationships", [])
            model_used = extraction_res.get("model_used", "")

            total_entities += len(entities)
            total_relationships += len(relationships)

            extracted_knowledge.append({
                "article_id": article.get("article_id"),
                "title": title,
                "link": article.get("link"),
                "source": article.get("source"),
                "category": article.get("category"),
                "published_rss": article.get("published_rss"),
                "entities": entities,
                "relationships": relationships
            })
            logger.info(f" LLM Extraction successful ({model_used}) for article '{article.get('article_id')}': {len(entities)} entities, {len(relationships)} relationships.")

        except Exception as e:
            err = f"LLM extraction failed for article {article.get('article_id')}: {e}"
            logger.error(err, exc_info=True)
            errors.append(err)

    logger.info(f"=== Node extract_knowledge Complete: Processed {len(extracted_knowledge)} articles via LLM | Extracted {total_entities} entities & {total_relationships} relationships. ===")

    return {
        "extracted_knowledge": extracted_knowledge,
        "total_entities_extracted": total_entities,
        "total_relationships_extracted": total_relationships,
        "errors": errors
    }
