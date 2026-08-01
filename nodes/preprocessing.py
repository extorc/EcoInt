"""
Node 1: Preprocessing
Fetches RSS feeds, cleans HTML and boilerplate, normalizes Unicode and whitespace,
deduplicates articles, and extracts document metadata.
"""

import logging
import feedparser
from typing import Dict, Any, List
from data_models import PipelineState, DocumentModel
from utils.text_processing import (
    clean_html_content,
    normalize_unicode_and_whitespace,
    compute_article_hash,
)
import config

logger = logging.getLogger("preprocessing_node")


def fetch_rss_preprocess_node(state: PipelineState) -> PipelineState:
    """
    LangGraph Node: Preprocessing
    1. Fetches RSS feeds specified in state.
    2. Cleans HTML tags, boilerplate, and normalizes Unicode/whitespace.
    3. Deduplicates articles by URL/title fingerprint.
    4. Extracts document metadata into structured Document records.
    """
    logger.info("--- [Stage 1/8] Preprocessing & Article Ingestion ---")
    feeds = state.get("rss_feeds", config.RSS_FEEDS)
    max_per_feed = state.get("max_articles_per_feed", config.MAX_ARTICLES_PER_FEED)
    total_max = state.get("total_max_articles", config.TOTAL_MAX_ARTICLES)

    seen_hashes = set()
    documents: List[Dict[str, Any]] = []
    errors: List[str] = list(state.get("errors", []))

    for feed_info in feeds:
        name = feed_info.get("name", "Unknown Source")
        url = feed_info.get("url")
        if not url:
            continue

        try:
            logger.info(f"Fetching RSS feed: {name} ({url})...")
            parsed_feed = feedparser.parse(url, agent=config.USER_AGENT)

            count = 0
            for entry in parsed_feed.entries:
                if count >= max_per_feed or len(documents) >= total_max:
                    break

                article_title = normalize_unicode_and_whitespace(getattr(entry, "title", ""))
                article_url = getattr(entry, "link", "")
                pub_date = getattr(entry, "published", getattr(entry, "updated", ""))

                if not article_title or not article_url:
                    continue

                # Deduplication check
                article_hash = compute_article_hash(article_url, article_title)
                if article_hash in seen_hashes:
                    logger.debug(f"Skipping duplicate article: {article_title}")
                    continue
                seen_hashes.add(article_hash)

                # Raw content extraction
                raw_body = ""
                if "content" in entry and len(entry.content) > 0:
                    raw_body = entry.content[0].value
                elif "summary" in entry:
                    raw_body = entry.summary
                elif "description" in entry:
                    raw_body = entry.description

                clean_body = clean_html_content(raw_body)
                if not clean_body or len(clean_body) < 30:
                    clean_body = article_title  # Fallback to title if body is too short

                doc_obj = DocumentModel(
                    title=article_title,
                    source=name,
                    url=article_url,
                    published_date=pub_date,
                    raw_content=raw_body,
                    clean_content=clean_body,
                )

                documents.append(doc_obj.model_dump())
                count += 1

        except Exception as err:
            err_msg = f"Error fetching RSS feed '{name}': {str(err)}"
            logger.error(err_msg)
            errors.append(err_msg)

    logger.info(f"Preprocessing completed. Ingested {len(documents)} unique documents.")
    return {
        **state,
        "documents": documents,
        "errors": errors,
    }
