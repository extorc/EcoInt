import json
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# NVIDIA model candidates (fallbacks for when specific endpoints are down or restricted)
NVIDIA_MODELS = [
    "meta/llama-3.1-8b-instruct"
]

def extract_with_nemotron(text: str, api_key: str) -> Dict[str, Any]:
    """Extract entities and relationships using NVIDIA's Nemotron API (via OpenAI client)."""
    from openai import OpenAI
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    prompt = f"""You are an expert financial and economic Knowledge Graph extraction system.
Analyze the following news text and extract key entities and direct relationships between them.

Text:
"{text}"

Return ONLY a raw JSON object (no markdown formatting, no code blocks) matching this schema:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "COMPANY|PERSON|ORGANIZATION|GOVERNMENT|COUNTRY|REGULATOR|INDUSTRY|SECTOR|TECHNOLOGY|PRODUCT|COMMODITY|FINANCIAL_INSTRUMENT|MARKET|CURRENCY",
      "description": "A single sentence that provides an absolute, objective, encyclopedia-style definition of what this entity fundamentally is. Do NOT describe its role or actions in the context of the article. For example, if the entity is 'US', the description MUST be 'The United States of America is a country in North America.', regardless of what the US did in the news story."
    }}
  ]
}}
Rules:
- STRICT ENTITY TYPES: You must categorize entities strictly into one of the types provided in the schema above.
- The 'description' must define the entity in a universal, standalone way. Do NOT include what the entity is doing in this specific news story.
- Every entity MUST have a non-empty description.
"""

    last_exception = None
    for model_name in NVIDIA_MODELS:
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "stream": False
            }

            response = client.chat.completions.create(**kwargs)
            raw_text = response.choices[0].message.content.strip()

            # Clean markdown backticks if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)

            data = json.loads(raw_text)
            return {
                "entities": data.get("entities", []),
                "model_used": model_name
            }
        except Exception as e:
            logger.warning(f"NVIDIA model '{model_name}' failed ({e}). Trying next candidate...")
            last_exception = e

    raise last_exception if last_exception else RuntimeError("All NVIDIA model candidates failed.")
