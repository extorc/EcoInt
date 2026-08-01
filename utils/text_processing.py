"""
Text Preprocessing Utilities for Cleaning, Unicode Normalization, Ticker Resolution, and Corporate Suffix Normalization.
"""

import re
import hashlib
import unicodedata
from typing import List, Dict, Any, Optional, Set
from bs4 import BeautifulSoup

# Standard stock ticker mapping to full corporate names
TICKER_MAP = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "BRK.A": "Berkshire Hathaway Inc.",
    "BRK.B": "Berkshire Hathaway Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "WMT": "Walmart Inc.",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble Co.",
    "MA": "Mastercard Inc.",
    "UNH": "UnitedHealth Group Inc.",
    "HD": "Home Depot Inc.",
    "BAC": "Bank of America Corp.",
    "XOM": "Exxon Mobil Corp.",
    "PFE": "Pfizer Inc.",
    "DIS": "Walt Disney Co.",
    "NFLX": "Netflix Inc.",
    "CSCO": "Cisco Systems Inc.",
    "INTC": "Intel Corp.",
    "AMD": "Advanced Micro Devices Inc.",
    "BABA": "Alibaba Group Holding Ltd.",
    "TCS": "Tata Consultancy Services Ltd.",
    "RELIANCE": "Reliance Industries Ltd.",
    "INFY": "Infosys Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.",
}

# Regex pattern matching standard corporate legal suffixes
COMPANY_SUFFIX_REGEX = re.compile(
    r"\b(Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|Co\.?|Company|LLC|PLC|Group|Holdings|N\.V\.|S\.A\.|AG|SE|GmbH)\b",
    re.IGNORECASE,
)


def clean_html_content(raw_html: str) -> str:
    """Strips HTML tags, scripts, styles, and extracts clean text."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    text = soup.get_text(separator=" ")
    return normalize_unicode_and_whitespace(text)


def normalize_unicode_and_whitespace(text: str) -> str:
    """Normalizes Unicode characters via NFKC and collapses multiple whitespace characters."""
    if not text:
        return ""
    # Normalize NFKC (compatibility decomposition followed by canonical composition)
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters except newline
    text = re.sub(r"[\r\t\f\v]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ ]+", " ", text)
    # Collapse multiple newlines
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def compute_article_hash(url: str, title: str) -> str:
    """Generates a unique SHA-256 fingerprint for deduplicating articles."""
    key = f"{url.strip().lower()}|{title.strip().lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def normalize_company_name(name: str) -> str:
    """
    Normalizes company/entity name by resolving tickers, removing corporate suffixes,
    and cleaning punctuation to form a canonical comparison key.
    """
    if not name:
        return ""
    clean_name = name.strip()

    # 1. Resolve exact Ticker match if applicable
    upper_name = clean_name.upper()
    if upper_name in TICKER_MAP:
        clean_name = TICKER_MAP[upper_name]

    # 2. Case normalization (Title Case representation)
    title_name = clean_name.title()

    # 3. Strip corporate suffixes to produce normalized matching key
    base_key = COMPANY_SUFFIX_REGEX.sub("", title_name)
    base_key = re.sub(r"[^\w\s]", "", base_key)  # Remove punctuation
    base_key = re.sub(r"\s+", " ", base_key).strip()

    return base_key if base_key else title_name


def clean_entity_text(name: str) -> str:
    """Strips outer quotes, punctuation, and extraneous spaces from entity mentions."""
    if not name:
        return ""
    cleaned = re.sub(r"^[\"\']|[\"\']$", "", name.strip())
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned)
    return cleaned.strip()


# Blacklisted generic phrases that should never be extracted as concrete entities
GENERIC_ENTITY_BLACKLIST = {
    "various products",
    "price increases",
    "financial conditions",
    "market to focus",
    "policy guidance",
    "economic conditions",
    "several items",
    "some products",
    "recent months",
    "next quarter",
    "last year",
    "this week",
}

# Regular expressions for pure numbers, monetary values, percentages, and currencies
MONETARY_NUMERIC_REGEX = re.compile(
    r"^[\$€£₹]?\s*\d+[\d,.]*\s*(percent|crore|lakh|billion|million|trillion|k|m|b)?$",
    re.IGNORECASE,
)


def is_valid_entity_mention(name: str, entity_type: str = "") -> bool:
    """
    Validates whether an extracted entity string is a valid concrete actor
    and not a pure number, monetary amount, percentage, or generic phrase.
    """
    if not name or len(name.strip()) < 2:
        return False

    cleaned = name.strip()
    cleaned_lower = cleaned.lower()

    # 1. Check Blacklist
    if cleaned_lower in GENERIC_ENTITY_BLACKLIST:
        return False

    # 2. Check if string is purely numbers or monetary quantity (e.g. "84,008-crore", "5.25 percent")
    if MONETARY_NUMERIC_REGEX.match(cleaned_lower):
        return False

    # 3. Reject strings that are only numbers or punctuation
    stripped_alpha = re.sub(r"[^\w]", "", cleaned)
    if not stripped_alpha or stripped_alpha.isdigit():
        return False

    return True


def generate_entity_description(name: str, entity_type: str, context_sentence: str = "") -> str:
    """
    Synthesizes an absolute, objective, encyclopedia-style description for an entity
    using its canonical name, entity type, and local context evidence.
    """
    clean_name = name.strip()
    e_type = entity_type.strip() if entity_type else "Entity"

    # Base definition by entity type
    if e_type == "Company":
        base_desc = f"{clean_name} is a business enterprise operating in the economic and commercial sector."
    elif e_type == "Regulator":
        base_desc = f"{clean_name} is an official regulatory authority overseeing policy, governance, or compliance."
    elif e_type == "Government":
        base_desc = f"{clean_name} is a sovereign government body or executive authority."
    elif e_type == "Person":
        base_desc = f"{clean_name} is an individual figure referenced in financial or economic discourse."
    elif e_type == "Country":
        base_desc = f"{clean_name} is a sovereign nation state."
    elif e_type == "Industry":
        base_desc = f"{clean_name} is a specialized sector of economic activity and industrial production."
    elif e_type == "Market":
        base_desc = f"{clean_name} is a financial or commercial trading market."
    elif e_type == "Technology":
        base_desc = f"{clean_name} is a technological system, platform, or innovation."
    elif e_type == "Product":
        base_desc = f"{clean_name} is a commercial product or service commodity."
    elif e_type == "Commodity":
        base_desc = f"{clean_name} is a raw material or primary agricultural/mineral good traded in markets."
    elif e_type == "Currency":
        base_desc = f"{clean_name} is a medium of exchange or monetary unit."
    elif e_type == "Financial Instrument":
        base_desc = f"{clean_name} is a monetary asset or tradeable financial contract."
    else:
        base_desc = f"{clean_name} is an economic entity classified as {e_type}."

    if context_sentence:
        snip = context_sentence.strip()
        if len(snip) > 120:
            snip = snip[:117] + "..."
        base_desc += f" Mentioned in context: \"{snip}\""

    return base_desc

