"""FastAPI application for RAG system."""

import logging
import time
import os
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import aiofiles

from ..rag_pipeline import RAGPipeline
from ..core.config import config
from .models import (
    QueryRequest, QueryResponse, ChatRequest, ChatResponse,
    IngestRequest, IngestResponse, HealthResponse, CollectionStatsResponse,
    RetrievedDocument
)

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="RAG System API",
        description="Retrieval Augmented Generation API with Ollama and Qdrant",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize RAG pipeline on startup."""
        logger.info("Starting RAG API...")
        success = rag_pipeline.initialize()
        if not success:
            logger.error("Failed to initialize RAG pipeline")
            raise RuntimeError("RAG pipeline initialization failed")
        logger.info("RAG API started successfully")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "RAG System API", "version": "1.0.0"}
    
    @app.post("/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Query the RAG system with a question."""
        try:
            response = rag_pipeline.query(
                question=request.question,
                max_context_length=request.max_context_length,
                temperature=request.temperature,
                use_diverse_retrieval=request.use_diverse_retrieval,
                include_context=request.include_context
            )
            
            # Convert to API response format
            retrieved_docs = [
                RetrievedDocument(
                    content=doc.content,
                    score=doc.score,
                    source=doc.source,
                    chunk_id=doc.chunk_id,
                    metadata=doc.metadata
                )
                for doc in response.retrieved_documents
            ]
            
            return QueryResponse(
                answer=response.answer,
                query=response.query,
                retrieved_documents=retrieved_docs,
                retrieval_time=response.retrieval_time,
                generation_time=response.generation_time,
                total_time=response.total_time,
                confidence_score=response.confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Chat with the RAG system."""
        try:
            messages = [msg.dict() for msg in request.messages]
            response = rag_pipeline.chat(
                messages=messages,
                max_context_length=request.max_context_length,
                temperature=request.temperature
            )
            
            return ChatResponse(
                response=response,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_documents(
        background_tasks: BackgroundTasks,
        request: IngestRequest,
        files: Optional[List[UploadFile]] = File(None)
    ):
        """Ingest documents into the RAG system."""
        try:
            start_time = time.time()
            documents_processed = 0
            
            if files:
                # Save uploaded files temporarily
                temp_paths = []
                for file in files:
                    if file.filename:
                        temp_path = Path("./data/documents") / file.filename
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        async with aiofiles.open(temp_path, 'wb') as f:
                            content = await file.read()
                            await f.write(content)
                        
                        temp_paths.append(temp_path)
                        documents_processed += 1
                
                if temp_paths:
                    # Process documents in background
                    background_tasks.add_task(
                        _process_documents_background,
                        rag_pipeline,
                        temp_paths,
                        request.clear_existing
                    )
            else:
                # Process documents from data/documents directory
                docs_dir = Path("./data/documents")
                if docs_dir.exists():
                    success = rag_pipeline.ingest_documents(
                        docs_dir,
                        clear_existing=request.clear_existing
                    )
                    if not success:
                        raise HTTPException(status_code=500, detail="Failed to ingest documents")
                    
                    # Count processed files
                    for file_path in docs_dir.rglob('*'):
                        if file_path.is_file():
                            documents_processed += 1
                else:
                    raise HTTPException(status_code=400, detail="No documents directory found")
            
            processing_time = time.time() - start_time
            
            return IngestResponse(
                success=True,
                message=f"Successfully processed {documents_processed} documents",
                documents_processed=documents_processed,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        try:
            health_status = rag_pipeline.health_check()
            
            return HealthResponse(
                status=health_status.get('overall_status', 'unknown'),
                components=health_status,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return HealthResponse(
                status="unhealthy",
                components={"error": str(e)},
                timestamp=time.time()
            )
    
    @app.get("/stats", response_model=CollectionStatsResponse)
    async def get_collection_stats():
        """Get collection statistics."""
        try:
            stats = rag_pipeline.get_collection_stats()
            
            return CollectionStatsResponse(
                collection_info=stats.get('collection_info', {}),
                embedding_model=stats.get('embedding_model', {}),
                retrieval_config=stats.get('retrieval_config', {})
            )
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/clear")
    async def clear_collection():
        """Clear all documents from the collection."""
        try:
            success = rag_pipeline.vector_db.clear_collection()
            if success:
                return {"message": "Collection cleared successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to clear collection")
                
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

async def _process_documents_background(
    rag_pipeline: RAGPipeline,
    document_paths: List[Path],
    clear_existing: bool
):
    """Process documents in background task."""
    try:
        logger.info(f"Background processing of {len(document_paths)} documents")
        success = rag_pipeline.ingest_documents(document_paths, clear_existing)
        
        if success:
            logger.info("Background document processing completed successfully")
        else:
            logger.error("Background document processing failed")
            
        # Clean up temporary files
        for path in document_paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file {path}: {e}")
                
    except Exception as e:
        logger.error(f"Error in background document processing: {e}")

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    app = create_app()
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level="info"
    )