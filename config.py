"""
Global Configuration Settings for the Economic Intelligence Knowledge Extraction Pipeline.
"""

import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()


# Colored Logging Formatter for Console
class RedErrorFormatter(logging.Formatter):
    RED = "\033[91m"
    BOLD_RED = "\033[1;91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelno >= logging.ERROR:
            prefix = self.BOLD_RED
            suffix = self.RESET
        elif record.levelno == logging.WARNING:
            prefix = self.YELLOW
            suffix = self.RESET
        else:
            prefix = ""
            suffix = ""

        formatted_msg = super().format(record)
        return f"{prefix}{formatted_msg}{suffix}"


def setup_colored_logging(log_level: str = "INFO"):
    """Configures global logging with RED error highlights and ANSI color support."""
    if sys.platform == "win32":
        try:
            import colorama
            colorama.init()
        except ImportError:
            os.system("")  # Enables VT100 ANSI escape sequences in Windows terminal

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = RedErrorFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


# Enable colored logging by default
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_colored_logging(LOG_LEVEL)

# RSS News Sources (Business & Economic News)
RSS_FEEDS = [
    {
        "name": "Economic Times",
        "url": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
        "category": "Economy"
    },
    {
        "name": "Moneycontrol",
        "url": "https://www.moneycontrol.com/rss/economy.xml",
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

# Ingestion Pipeline Parameters
MAX_ARTICLES_PER_FEED = 5
TOTAL_MAX_ARTICLES = 25
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Machine Learning & NLP Model Configurations
GLINER_MODEL = "urchade/gliner_medium-v2.1"
SPACY_MODEL = "en_core_web_trf"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-large"

# Entity Resolution Thresholds
FUZZY_MATCH_THRESHOLD = 85.0
RERANKER_SCORE_THRESHOLD = 0.65

# API Keys Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Neo4j Knowledge Graph Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
