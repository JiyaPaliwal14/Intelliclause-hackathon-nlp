"""
Defines Pydantic models for request/response & metadata.
"""

from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class ChunkMetadata(BaseModel):
    document_id: str
    file_name: str
    chunk_id: int
    page_number: int
    section_title: Optional[str]
    clause_id: Optional[str]
    doc_type: Optional[str]