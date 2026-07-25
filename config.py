import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()


def get_gemini_api_key() -> str:
    """
    Resolves Gemini API Key from:
    1. os.getenv / os.environ (GEMINI_API_KEY, GOOGLE_API_KEY, etc.)
    2. .env file
    3. Windows Registry (User & System environment variables) for newly set system env vars.
    """
    for key_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY", "GOOGLE_GEMINI_API_KEY"]:
        val = os.getenv(key_name)
        if val and val.strip():
            return val.strip()

    # Case-insensitive search in os.environ
    for k, v in os.environ.items():
        if k.upper() in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"] and v.strip():
            return v.strip()

    # On Windows, check Windows Registry directly in case terminal was opened before system env var was set
    if sys.platform == "win32":
        try:
            import winreg
            for hk in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
                path = r"Environment" if hk == winreg.HKEY_CURRENT_USER else r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
                try:
                    with winreg.OpenKey(hk, path, 0, winreg.KEY_READ) as key:
                        for target in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"]:
                            try:
                                val, _ = winreg.QueryValueEx(key, target)
                                if val and str(val).strip():
                                    val_str = str(val).strip()
                                    os.environ[target] = val_str
                                    return val_str
                            except FileNotFoundError:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

    return ""


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
LOG_LEVEL = "INFO"
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
MAX_ARTICLES_PER_FEED = 6  # 5 feeds * 6 = 30 articles total
TOTAL_MAX_ARTICLES = 30

# Network configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# LLM Extraction configuration
GEMINI_API_KEY = get_gemini_api_key()
# High-quota, relaxed-limit model defaults
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-flash-lite-latest")

# Neo4j Knowledge Graph configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
