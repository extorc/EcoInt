"""
Node 2: Sentence Segmentation
Segments cleaned document texts into distinct sentence records using spaCy (en_core_web_trf).
"""

import logging
from typing import List, Dict, Any
from data_models import PipelineState, SentenceModel
from utils.spacy_utils import get_spacy_nlp
import config

logger = logging.getLogger("sentence_segmentation_node")


def sentence_segmentation_node(state: PipelineState) -> PipelineState:
    """
    LangGraph Node: Sentence Segmentation
    Processes documents using spaCy to break text into sentences with sentence_ids.
    """
    logger.info("--- [Stage 2/8] Sentence Segmentation (spaCy) ---")
    documents = state.get("documents", [])
    errors: List[str] = list(state.get("errors", []))

    nlp = get_spacy_nlp(config.SPACY_MODEL)
    segmented_sentences: List[Dict[str, Any]] = []

    for doc_dict in documents:
        doc_uuid = doc_dict.get("uuid")
        clean_text = doc_dict.get("clean_content", "")
        title = doc_dict.get("title", "")

        if not clean_text:
            continue

        try:
            # Run text through spaCy
            spacy_doc = nlp(clean_text)

            sents = [sent for sent in spacy_doc.sents if len(sent.text.strip()) > 5]

            # If no sentences found (or short body), add title as single sentence
            if not sents:
                title_sent = SentenceModel(
                    sentence_id=f"{doc_uuid}_s0",
                    document_uuid=doc_uuid,
                    index=0,
                    text=title,
                    start_char=0,
                    end_char=len(title),
                )
                segmented_sentences.append(title_sent.model_dump())
                continue

            for idx, sent in enumerate(sents):
                sent_obj = SentenceModel(
                    sentence_id=f"{doc_uuid}_s{idx}",
                    document_uuid=doc_uuid,
                    index=idx,
                    text=sent.text.strip(),
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                )
                segmented_sentences.append(sent_obj.model_dump())

        except Exception as err:
            err_msg = f"Error in sentence segmentation for doc '{doc_uuid}': {str(err)}"
            logger.error(err_msg)
            errors.append(err_msg)

    logger.info(f"Sentence segmentation completed. Extracted {len(segmented_sentences)} sentences.")
    return {
        **state,
        "sentences": segmented_sentences,
        "errors": errors,
    }
