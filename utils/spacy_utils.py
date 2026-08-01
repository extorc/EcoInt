"""
spaCy Helper Utilities for Sentence Segmentation and Dependency Pattern Matching.
"""

import logging
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from typing import Tuple, List, Dict, Any

logger = logging.getLogger("spacy_utils")

_SPACY_NLP_INSTANCE = None


def get_spacy_nlp(model_name: str = "en_core_web_trf"):
    """
    Loads and caches the requested spaCy pipeline.
    Falls back gracefully to 'en_core_web_sm' if the transformer model is unavailable.
    """
    global _SPACY_NLP_INSTANCE
    if _SPACY_NLP_INSTANCE is not None:
        return _SPACY_NLP_INSTANCE

    try:
        logger.info(f"Loading preferred spaCy pipeline: '{model_name}'...")
        nlp = spacy.load(model_name)
    except Exception as err:
        logger.warning(f"Failed to load '{model_name}' ({err}). Attempting fallback to 'en_core_web_sm'...")
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.info("Downloading spaCy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

    _SPACY_NLP_INSTANCE = nlp
    return _SPACY_NLP_INSTANCE


def create_relationship_matchers(nlp) -> Tuple[Matcher, Dict[str, str]]:
    """
    Constructs a spaCy Matcher configured with patterns for financial/economic relationships:
    - ACQUIRED
    - INVESTED_IN
    - PARTNERED_WITH
    - SUPPLIES
    - DEPENDS_ON
    - LOCATED_IN
    - PRODUCES
    - REGULATES
    - COMPETES_WITH
    - OWNS
    """
    matcher = Matcher(nlp.vocab)

    # 1. ACQUIRED
    acquired_patterns = [
        [{"LEMMA": "acquire"}],
        [{"LEMMA": "buy"}, {"OP": "?"}, {"LEMMA": "out"}],
        [{"LEMMA": "purchase"}],
        [{"LEMMA": "take"}, {"LEMMA": "over"}],
        [{"LEMMA": "merge"}, {"LEMMA": "with"}],
    ]
    for idx, pat in enumerate(acquired_patterns):
        matcher.add(f"ACQUIRED_{idx}", [pat])

    # 2. INVESTED_IN
    invested_patterns = [
        [{"LEMMA": "invest"}, {"LEMMA": "in"}],
        [{"LEMMA": "fund"}],
        [{"LEMMA": "inject"}, {"LEMMA": "capital"}],
        [{"LEMMA": "finance"}],
        [{"LEMMA": "back"}],
    ]
    for idx, pat in enumerate(invested_patterns):
        matcher.add(f"INVESTED_IN_{idx}", [pat])

    # 3. PARTNERED_WITH
    partner_patterns = [
        [{"LEMMA": "partner"}, {"LEMMA": "with"}],
        [{"LEMMA": "collaborate"}, {"LEMMA": "with"}],
        [{"LEMMA": "team"}, {"LEMMA": "up"}],
        [{"LEMMA": "joint"}, {"LEMMA": "venture"}],
        [{"LEMMA": "tie"}, {"LEMMA": "up"}],
    ]
    for idx, pat in enumerate(partner_patterns):
        matcher.add(f"PARTNERED_WITH_{idx}", [pat])

    # 4. SUPPLIES
    supplies_patterns = [
        [{"LEMMA": "supply"}],
        [{"LEMMA": "provide"}, {"LEMMA": "to"}],
        [{"LEMMA": "deliver"}, {"LEMMA": "to"}],
        [{"LEMMA": "vendor"}],
    ]
    for idx, pat in enumerate(supplies_patterns):
        matcher.add(f"SUPPLIES_{idx}", [pat])

    # 5. DEPENDS_ON
    depends_patterns = [
        [{"LEMMA": "depend"}, {"LEMMA": "on"}],
        [{"LEMMA": "rely"}, {"LEMMA": "on"}],
        [{"LEMMA": "contingent"}, {"LEMMA": "on"}],
    ]
    for idx, pat in enumerate(depends_patterns):
        matcher.add(f"DEPENDS_ON_{idx}", [pat])

    # 6. LOCATED_IN
    located_patterns = [
        [{"LEMMA": "base"}, {"LEMMA": "in"}],
        [{"LEMMA": "headquarter"}, {"LEMMA": "in"}],
        [{"LEMMA": "locate"}, {"LEMMA": "in"}],
        [{"LEMMA": "operate"}, {"LEMMA": "in"}],
    ]
    for idx, pat in enumerate(located_patterns):
        matcher.add(f"LOCATED_IN_{idx}", [pat])

    # 7. PRODUCES
    produces_patterns = [
        [{"LEMMA": "produce"}],
        [{"LEMMA": "manufacture"}],
        [{"LEMMA": "develop"}],
        [{"LEMMA": "launch"}],
        [{"LEMMA": "create"}],
    ]
    for idx, pat in enumerate(produces_patterns):
        matcher.add(f"PRODUCES_{idx}", [pat])

    # 8. REGULATES
    regulates_patterns = [
        [{"LEMMA": "regulate"}],
        [{"LEMMA": "oversee"}],
        [{"LEMMA": "penalize"}],
        [{"LEMMA": "fine"}],
        [{"LEMMA": "approve"}],
        [{"LEMMA": "probe"}],
        [{"LEMMA": "investigate"}],
    ]
    for idx, pat in enumerate(regulates_patterns):
        matcher.add(f"REGULATES_{idx}", [pat])

    # 9. COMPETES_WITH
    competes_patterns = [
        [{"LEMMA": "compete"}, {"LEMMA": "with"}],
        [{"LEMMA": "rival"}],
        [{"LEMMA": "challenge"}],
        [{"LEMMA": "vie"}, {"LEMMA": "against"}],
    ]
    for idx, pat in enumerate(competes_patterns):
        matcher.add(f"COMPETES_WITH_{idx}", [pat])

    # 10. OWNS
    owns_patterns = [
        [{"LEMMA": "own"}],
        [{"LEMMA": "possess"}],
        [{"LEMMA": "hold"}, {"LEMMA": "stake"}],
        [{"LEMMA": "parent"}, {"LEMMA": "company"}],
    ]
    for idx, pat in enumerate(owns_patterns):
        matcher.add(f"OWNS_{idx}", [pat])

    return matcher
