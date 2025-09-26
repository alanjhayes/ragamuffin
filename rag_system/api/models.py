"""Pydantic models for API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    question: str = Field(..., description="The question to ask")
    max_context_length: int = Field(2000, description="Maximum context length")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="LLM temperature")
    use_diverse_retrieval: bool = Field(False, description="Use diverse retrieval")
    include_context: bool = Field(True, description="Include surrounding context")

class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    """Request model for chat interface."""
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    max_context_length: int = Field(2000, description="Maximum context length")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="LLM temperature")

class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    clear_existing: bool = Field(False, description="Clear existing documents")

class RetrievedDocument(BaseModel):
    """Model for retrieved document."""
    content: str
    score: float
    source: str
    chunk_id: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    query: str
    retrieved_documents: List[RetrievedDocument]
    retrieval_time: float
    generation_time: float
    total_time: float
    confidence_score: Optional[float] = None

class ChatResponse(BaseModel):
    """Response model for chat."""
    response: str
    timestamp: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    components: Dict[str, Any]
    timestamp: float

class CollectionStatsResponse(BaseModel):
    """Collection statistics response."""
    collection_info: Dict[str, Any]
    embedding_model: Dict[str, Any]
    retrieval_config: Dict[str, Any]

class IngestResponse(BaseModel):
    """Document ingestion response."""
    success: bool
    message: str
    documents_processed: int
    processing_time: float