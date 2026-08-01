"""
Node 5: Relationship Extraction & "Why Connected" Explanation Node
Leverages LLM (Gemini / NVIDIA) with spaCy fallback to extract dynamic semantic relationships,
generate explicit "Why Connected" explanations, and merge duplicate graph edges.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from data_models import PipelineState, RelationshipModel
from utils.spacy_utils import get_spacy_nlp, create_relationship_matchers
from utils.text_processing import is_valid_entity_mention
import config

logger = logging.getLogger("relationship_extraction_node")


def _extract_llm_relationships(
    sentence_text: str,
    entities: List[Dict[str, Any]],
    gemini_key: str,
    nvidia_key: str,
) -> List[Dict[str, Any]]:
    """
    Invokes LLM (Gemini or NVIDIA) to dynamically extract semantic relationships
    and generate 'Why Connected' explanations.
    """
    ent_names = [f"'{e.get('name')}' ({e.get('entity_type')})" for e in entities]
    ent_list_str = ", ".join(ent_names)

    prompt = f"""You are an expert economic Knowledge Graph analyst.
Analyze the following sentence and extract semantic relationships between the specified entities.

Sentence: "{sentence_text}"
Entities present: [{ent_list_str}]

Extract direct relationships between any pairs of entities.
For each relationship:
- "source_entity": Name of Entity A (must match one of the entities above)
- "target_entity": Name of Entity B (must match one of the entities above)
- "relationship_type": A dynamic, uppercase semantic action/relation string reflecting the article (e.g. APPROVED_FUNDING_FOR, SUBSIDIZED, INCREASED_PRICES_OF, INVESTED_IN, ACQUIRED, POLICY_GUIDANCE_ON, COMPETES_WITH, REGULATES, SUPPLIES_TO, PARTNERED_WITH, PROBED, LAUNCHED).
- "description": A clear 1-2 sentence explanation of WHY these two nodes are connected according to the text.
- "confidence": Float between 0.5 and 1.0.

Return ONLY a raw JSON array for example (no markdown code blocks, no text before or after):
[
  {{
    "source_entity": "Entity A",
    "target_entity": "Entity B",
    "relationship_type": "APPROVED_FUNDING_FOR",
    "description": "The Union Cabinet is connected to PM-Kisan scheme because it approved an ₹84,008-crore programme to accelerate agricultural investment.",
    "confidence": 0.95
  }}
]
"""

    raw_text = ""

    # Attempt 1: Gemini API via google.genai SDK
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            raw_text = resp.text.strip()
        except Exception:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=gemini_key)
                model = legacy_genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(prompt)
                raw_text = resp.text.strip()
            except Exception as g_err:
                logger.debug(f"Gemini API relationship extraction notice: {g_err}")

    # Attempt 2: NVIDIA API fallback
    if not raw_text and nvidia_key:
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key)
            resp = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=768,
            )
            raw_text = resp.choices[0].message.content.strip()
        except Exception as n_err:
            logger.debug(f"NVIDIA API relationship extraction notice: {n_err}")

    if not raw_text:
        return []

    # Clean markdown formatting if present
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text)

    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
    except Exception as parse_err:
        logger.debug(f"Failed to parse LLM relationship JSON: {parse_err}")

    return []


def relationship_extraction_node(state: PipelineState) -> PipelineState:
    """
    LangGraph Node: Relationship Extraction
    Extracts semantic relationships with explicit "Why Connected" explanations,
    supporting dynamic relation types and strict edge deduplication.
    """
    logger.info("--- [Stage 5/8] LLM Semantic Relationship & Explanation Extraction ---")
    normalized_entities = state.get("normalized_entities", [])
    sentences = state.get("sentences", [])
    errors: List[str] = list(state.get("errors", []))

    gemini_key = getattr(config, "GEMINI_API_KEY", "")
    nvidia_key = getattr(config, "NVIDIA_API_KEY", "")

    nlp = get_spacy_nlp(config.SPACY_MODEL)
    matcher = create_relationship_matchers(nlp)

    # Index sentences by sentence_id
    sentence_map = {s["sentence_id"]: s for s in sentences}

    # Group valid entities by sentence_id
    entities_by_sentence = defaultdict(list)
    for ent in normalized_entities:
        e_name = ent.get("name", "")
        e_type = ent.get("entity_type", "")
        if is_valid_entity_mention(e_name, e_type):
            entities_by_sentence[ent["sentence_id"]].append(ent)

    raw_extracted_relationships: List[Dict[str, Any]] = []

    for sent_id, ent_list in entities_by_sentence.items():
        if len(ent_list) < 2:
            continue

        sent_info = sentence_map.get(sent_id)
        if not sent_info:
            continue

        sent_text = sent_info.get("text", "")
        doc_uuid = sent_info.get("document_uuid")
        if not sent_text:
            continue

        # Fast lookup mapping from entity name/normalized name to entity object
        ent_lookup = {}
        for e in ent_list:
            ent_lookup[e["name"].lower()] = e
            ent_lookup[e["normalized_name"].lower()] = e

        # 1. Try LLM Extraction for dynamic relation types & "why connected" explanation
        llm_rels = _extract_llm_relationships(sent_text, ent_list, gemini_key, nvidia_key)

        found_llm = False
        if llm_rels:
            for item in llm_rels:
                src_name_raw = str(item.get("source_entity", "")).strip()
                tgt_name_raw = str(item.get("target_entity", "")).strip()
                rel_type_raw = str(item.get("relationship_type", "RELATED_TO")).strip().upper()
                desc = str(item.get("description", "")).strip()
                conf = float(item.get("confidence", 0.85))

                # Sanitize rel_type_raw for Cypher compatibility
                rel_type = re.sub(r"[^\w]", "_", rel_type_raw).strip("_")
                if not rel_type:
                    rel_type = "RELATED_TO"

                # Match names to entity dicts
                src_ent = ent_lookup.get(src_name_raw.lower())
                tgt_ent = ent_lookup.get(tgt_name_raw.lower())

                if not src_ent or not tgt_ent:
                    # Fuzzy match fallback within sentence entities
                    for e in ent_list:
                        if not src_ent and (src_name_raw.lower() in e["name"].lower() or e["name"].lower() in src_name_raw.lower()):
                            src_ent = e
                        if not tgt_ent and (tgt_name_raw.lower() in e["name"].lower() or e["name"].lower() in tgt_name_raw.lower()):
                            tgt_ent = e

                if src_ent and tgt_ent:
                    src_uuid = src_ent.get("canonical_uuid") or src_ent.get("uuid")
                    tgt_uuid = tgt_ent.get("canonical_uuid") or tgt_ent.get("uuid")

                    if src_uuid == tgt_uuid:
                        continue

                    if not desc:
                        desc = f"{src_ent['name']} is connected to {tgt_ent['name']} via {rel_type} in context: \"{sent_text[:100]}\"."

                    rel_obj = RelationshipModel(
                        source_entity_uuid=src_uuid,
                        source_entity_name=src_ent["name"],
                        target_entity_uuid=tgt_uuid,
                        target_entity_name=tgt_ent["name"],
                        relationship_type=rel_type,
                        description=desc,
                        confidence=conf,
                        supporting_sentence=sent_text,
                        sentence_id=sent_id,
                        document_uuid=doc_uuid,
                    )
                    raw_extracted_relationships.append(rel_obj.model_dump())
                    found_llm = True

        # 2. Rule-Based Fallback if LLM produced no output for this sentence
        if not found_llm:
            try:
                spacy_doc = nlp(sent_text)
                matches = matcher(spacy_doc)

                detected_relations = []
                for match_id, start, end in matches:
                    rule_name = nlp.vocab.strings[match_id]
                    rel_type = rule_name.rsplit("_", 1)[0]
                    detected_relations.append(rel_type)

                for i in range(len(ent_list)):
                    for j in range(len(ent_list)):
                        if i == j:
                            continue
                        e_src = ent_list[i]
                        e_tgt = ent_list[j]

                        src_uuid = e_src.get("canonical_uuid") or e_src.get("uuid")
                        tgt_uuid = e_tgt.get("canonical_uuid") or e_tgt.get("uuid")
                        if src_uuid == tgt_uuid:
                            continue

                        assigned_rel = detected_relations[0] if detected_relations else None
                        conf = 0.85 if detected_relations else 0.70

                        if not assigned_rel:
                            s_t = e_src.get("entity_type")
                            t_t = e_tgt.get("entity_type")
                            if s_t in ["Company", "Organization"] and t_t == "Country":
                                assigned_rel = "LOCATED_IN"
                            elif s_t in ["Company", "Organization"] and t_t == "Product":
                                assigned_rel = "PRODUCES"
                            elif s_t == "Regulator" and t_t in ["Company", "Market", "Industry", "Sector"]:
                                assigned_rel = "REGULATES"
                            elif s_t == "Company" and t_t == "Company":
                                assigned_rel = "COMPETES_WITH"

                        if assigned_rel:
                            fallback_desc = f"{e_src['name']} is connected to {e_tgt['name']} ({assigned_rel}) according to article text: \"{sent_text[:100]}\"."
                            rel_obj = RelationshipModel(
                                source_entity_uuid=src_uuid,
                                source_entity_name=e_src["name"],
                                target_entity_uuid=tgt_uuid,
                                target_entity_name=e_tgt["name"],
                                relationship_type=assigned_rel,
                                description=fallback_desc,
                                confidence=conf,
                                supporting_sentence=sent_text,
                                sentence_id=sent_id,
                                document_uuid=doc_uuid,
                            )
                            raw_extracted_relationships.append(rel_obj.model_dump())
            except Exception as spacy_err:
                logger.debug(f"spaCy fallback notice: {spacy_err}")

    # --- STRICT RELATIONSHIP EDGE DEDUPLICATION & MERGING ---
    # Merge duplicate relationship triples by (source_uuid, relationship_type, target_uuid)
    merged_map: Dict[tuple, Dict[str, Any]] = {}

    for rel in raw_extracted_relationships:
        key = (rel["source_entity_uuid"], rel["relationship_type"], rel["target_entity_uuid"])

        if key not in merged_map:
            merged_map[key] = dict(rel)
        else:
            existing = merged_map[key]
            existing["confidence"] = max(existing["confidence"], rel["confidence"])

            # Merge why-connected descriptions if distinct
            if rel["description"] and rel["description"] not in existing["description"]:
                existing["description"] += f" | {rel['description']}"

            # Merge supporting sentences
            if rel["supporting_sentence"] not in existing["supporting_sentence"]:
                existing["supporting_sentence"] += f" | {rel['supporting_sentence']}"

    merged_relationships = list(merged_map.values())

    logger.info(
        f"Relationship extraction completed. Extracted {len(raw_extracted_relationships)} raw relations, "
        f"merged down to {len(merged_relationships)} unique deduplicated graph edges."
    )

    return {
        **state,
        "relationships": merged_relationships,
        "total_relationships_extracted": len(merged_relationships),
        "errors": errors,
    }
