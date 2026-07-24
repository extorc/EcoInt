import os

# RSS News Sources (Business & Economic News)
RSS_FEEDS = [
    {
        "name": "Economic Times",
        "url": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
        "category": "Economy"
    },
    {
        "name": "Moneycontrol",
        "url": "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "category": "Markets & Economy"
    },
    {
        "name": "Mint",
        "url": "https://www.livemint.com/rss/economy",
        "category": "Economy"
    },
    {
        "name": "CNBC",
        "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "category": "Global Economy"
    },
    {
        "name": "BBC Business",
        "url": "http://feeds.bbci.co.uk/news/business/rss.xml",
        "category": "Global Business"
    }
]

# Pipeline parameters
MAX_ARTICLES_PER_FEED = 2  # 5 feeds * 2 = 10 articles total default
TOTAL_MAX_ARTICLES = 10

# Network configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Qdrant Vector DB configuration
QDRANT_COLLECTION_NAME = "news_articles"
QDRANT_STORAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Default similarity search query after ingestion
DEFAULT_SEARCH_QUERY = "tariffs and export trade opportunities"

# Logging configuration
LOG_LEVEL = "INFO"
