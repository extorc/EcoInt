"""
Data Models and TypedDict State for the Economic Intelligence Knowledge Extraction Pipeline.
"""

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
import uuid
import datetime

# Target GLiNER Entity Types
ENTITY_TYPES = [
    "Company",
    "Person",
    "Organization",
    "Government",
    "Regulator",
    "Country",
    "Industry",
    "Sector",
    "Technology",
    "Product",
    "Commodity",
    "Currency",
    "Financial Instrument",
    "Market",
]

# Target spaCy Pattern Relationship Types
RELATIONSHIP_TYPES = [
    "ACQUIRED",
    "INVESTED_IN",
    "PARTNERED_WITH",
    "SUPPLIES",
    "DEPENDS_ON",
    "LOCATED_IN",
    "PRODUCES",
    "REGULATES",
    "COMPETES_WITH",
    "OWNS",
]


class DocumentModel(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    source: str
    url: str
    published_date: str = ""
    raw_content: str = ""
    clean_content: str = ""
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class SentenceModel(BaseModel):
    sentence_id: str
    document_uuid: str
    index: int
    text: str
    start_char: int = 0
    end_char: int = 0


class EntityModel(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    normalized_name: str = ""
    entity_type: str
    description: str = ""
    confidence: float = 1.0
    sentence_id: str
    document_uuid: str
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[List[float]] = None
    canonical_uuid: Optional[str] = None


class RelationshipModel(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_uuid: str
    source_entity_name: str
    target_entity_uuid: str
    target_entity_name: str
    relationship_type: str
    description: str = ""
    confidence: float = 1.0
    supporting_sentence: str
    sentence_id: str
    document_uuid: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class PipelineState(TypedDict, total=False):
    """
    LangGraph Pipeline State passing data between pipeline stages.
    """
    rss_feeds: List[Dict[str, str]]
    max_articles_per_feed: int
    total_max_articles: int
    documents: List[Dict[str, Any]]
    sentences: List[Dict[str, Any]]
    raw_entities: List[Dict[str, Any]]
    normalized_entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    resolved_entities: List[Dict[str, Any]]
    embedded_entities: List[Dict[str, Any]]
    total_entities_extracted: int
    total_relationships_extracted: int
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    neo4j_nodes_created: int
    neo4j_relationships_created: int
    neo4j_status: str
    errors: List[str]
