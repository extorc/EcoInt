import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import feedparser
import config

logger = logging.getLogger(__name__)

def fetch_rss_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Fetches RSS feeds from configured news sources and extracts candidate articles.
    
    State Input:
        - rss_feeds (optional): List of dicts with 'name' and 'url'.
        - max_articles_per_feed (optional): Max articles per feed.
        - total_max_articles (optional): Overall limit on articles collected.
    
    State Output:
        - raw_articles: List of parsed RSS feed article items.
        - errors: List of non-fatal error messages encountered during execution.
    """
    logger.info("=== Starting Node: fetch_rss ===")
    
    feeds = state.get("rss_feeds", config.RSS_FEEDS)
    max_per_feed = state.get("max_articles_per_feed", config.MAX_ARTICLES_PER_FEED)
    total_max = state.get("total_max_articles", config.TOTAL_MAX_ARTICLES)
    
    raw_articles: List[Dict[str, Any]] = state.get("raw_articles", [])
    errors: List[str] = state.get("errors", [])
    
    count_before = len(raw_articles)
    
    for feed_info in feeds:
        source_name = feed_info.get("name", "Unknown Source")
        url = feed_info.get("url")
        category = feed_info.get("category", "General")
        
        if not url:
            msg = f"Skipping feed '{source_name}': Missing URL."
            logger.warning(msg)
            errors.append(msg)
            continue
            
        logger.info(f"Fetching RSS feed from '{source_name}' ({url})...")
        
        try:
            # Parse feed with custom user-agent
            parsed_feed = feedparser.parse(
                url, 
                agent=config.USER_AGENT,
                request_headers={"User-Agent": config.USER_AGENT}
            )
            
            # Check for bozo exception (well-formedness flag in feedparser)
            if getattr(parsed_feed, "bozo", False) and not parsed_feed.entries:
                bozo_exc = getattr(parsed_feed, "bozo_exception", "Unknown parse error")
                msg = f"Feed parsing warning/error for '{source_name}': {bozo_exc}"
                logger.warning(msg)
                errors.append(msg)
                
            entries = parsed_feed.entries
            if not entries:
                logger.warning(f"No entries found in feed '{source_name}'.")
                continue
                
            logger.info(f"Retrieved {len(entries)} entries from '{source_name}'. Processing top {max_per_feed}...")
            
            feed_count = 0
            for entry in entries:
                if feed_count >= max_per_feed:
                    break
                if len(raw_articles) >= total_max:
                    logger.info(f"Reached total article limit ({total_max}). Halting RSS fetch.")
                    break
                    
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                
                if not title or not link:
                    continue
                    
                # Generate unique ID based on URL
                article_id = hashlib.md5(link.encode("utf-8")).hexdigest()[:12]
                
                published = entry.get("published") or entry.get("updated") or entry.get("pubDate") or ""
                summary = entry.get("summary") or entry.get("description") or ""
                
                article_item = {
                    "article_id": article_id,
                    "title": title,
                    "link": link,
                    "source": source_name,
                    "category": category,
                    "published_rss": published,
                    "rss_summary": summary,
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                }
                
                raw_articles.append(article_item)
                feed_count += 1
                
        except Exception as e:
            error_msg = f"Failed to fetch RSS for '{source_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            
        if len(raw_articles) >= total_max:
            break
            
    total_fetched = len(raw_articles) - count_before
    logger.info(f"=== Node fetch_rss Complete: Fetched {total_fetched} raw articles across {len(feeds)} sources. ===")
    
    return {
        "raw_articles": raw_articles,
        "errors": errors
    }
