from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str
    documents: List[str] = []
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float

class UploadResponse(BaseModel):
    document_id: str
    chunks: int
    status: str

class SourceChunk(BaseModel):
    content: str
    source: str
    score: float
