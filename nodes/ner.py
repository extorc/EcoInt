"""
Node 3: Named Entity Recognition
Extracts entities using GLiNER (urchade/gliner_medium-v2.1) without LLM calls.
Target labels (14): Company, Person, Organization, Government, Regulator, Country,
Industry, Sector, Technology, Product, Commodity, Currency, Financial Instrument, Market.
"""

import logging
from typing import List, Dict, Any
from data_models import PipelineState, EntityModel, ENTITY_TYPES
from utils.text_processing import is_valid_entity_mention, generate_entity_description
import config

logger = logging.getLogger("ner_node")

_GLINER_MODEL_INSTANCE = None


def get_gliner_model():
    """Lazy initialization of GLiNER model."""
    global _GLINER_MODEL_INSTANCE
    if _GLINER_MODEL_INSTANCE is None:
        logger.info(f"Loading GLiNER model: '{config.GLINER_MODEL}'...")
        # pyrefly: ignore [missing-import]
        from gliner import GLiNER
        _GLINER_MODEL_INSTANCE = GLiNER.from_pretrained(config.GLINER_MODEL)
    return _GLINER_MODEL_INSTANCE


def named_entity_recognition_node(state: PipelineState) -> PipelineState:
    """
    LangGraph Node: Named Entity Recognition (GLiNER)
    Predicts 14 custom business/financial entity labels per sentence.
    """
    logger.info("--- [Stage 3/8] Named Entity Recognition (GLiNER v2.1) ---")
    sentences = state.get("sentences", [])
    errors: List[str] = list(state.get("errors", []))

    raw_entities: List[Dict[str, Any]] = []

    try:
        model = get_gliner_model()
    except Exception as err:
        err_msg = f"Failed to load GLiNER model: {str(err)}"
        logger.error(err_msg)
        errors.append(err_msg)
        return {**state, "raw_entities": [], "errors": errors}

    for sent_dict in sentences:
        sent_id = sent_dict.get("sentence_id")
        doc_uuid = sent_dict.get("document_uuid")
        sent_text = sent_dict.get("text", "")

        if not sent_text or len(sent_text) < 5:
            continue

        try:
            preds = model.predict_entities(sent_text, ENTITY_TYPES, threshold=0.35)

            for p in preds:
                entity_text = p.get("text", "").strip()
                label = p.get("label", "")
                score = float(p.get("score", 0.0))

                if not is_valid_entity_mention(entity_text, label):
                    continue

                desc = generate_entity_description(entity_text, label, sent_text)

                entity_obj = EntityModel(
                    name=entity_text,
                    entity_type=label,
                    description=desc,
                    confidence=round(score, 4),
                    sentence_id=sent_id,
                    document_uuid=doc_uuid,
                    start_char=int(p.get("start", 0)),
                    end_char=int(p.get("end", 0)),
                )
                raw_entities.append(entity_obj.model_dump())

        except Exception as err:
            err_msg = f"Error performing NER on sentence '{sent_id}': {str(err)}"
            logger.error(err_msg)
            errors.append(err_msg)

    logger.info(f"NER completed. Extracted {len(raw_entities)} entity mentions.")
    return {
        **state,
        "raw_entities": raw_entities,
        "total_entities_extracted": len(raw_entities),
        "errors": errors,
    }
