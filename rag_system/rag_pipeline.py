"""Main RAG pipeline integrating retrieval and generation."""

import logging
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from .core.config import config
from .core.ollama_client import OllamaClient
from .embeddings.embedding_model import EmbeddingModel
from .database.qdrant_client import QdrantVectorDB
from .retrieval.retriever import DocumentRetriever, RetrievalResult
from .utils.document_processor import DocumentProcessor, DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    retrieved_documents: List[RetrievalResult]
    query: str
    retrieval_time: float
    generation_time: float
    total_time: float
    confidence_score: Optional[float] = None

class RAGPipeline:
    """Complete RAG pipeline for document question answering."""
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_db: Optional[QdrantVectorDB] = None,
        document_processor: Optional[DocumentProcessor] = None,
        retriever: Optional[DocumentRetriever] = None
    ):
        """Initialize RAG pipeline.
        
        Args:
            ollama_client: Client for LLM generation
            embedding_model: Model for embeddings
            vector_db: Vector database
            document_processor: Document processing utilities
            retriever: Document retrieval system
        """
        self.ollama_client = ollama_client or OllamaClient()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_db = vector_db or QdrantVectorDB()
        self.document_processor = document_processor or DocumentProcessor()
        self.retriever = retriever or DocumentRetriever(
            embedding_model=self.embedding_model,
            vector_db=self.vector_db
        )
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the RAG pipeline components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing RAG pipeline...")
            
            # Check Ollama model availability
            if not self.ollama_client.check_model_availability():
                logger.info("Model not available, attempting to pull...")
                if not self.ollama_client.pull_model():
                    logger.error("Failed to pull model")
                    return False
            
            # Initialize vector database collection
            if not self.vector_db.create_collection():
                logger.error("Failed to create vector database collection")
                return False
            
            self._initialized = True
            logger.info("RAG pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            return False
    
    def ingest_documents(
        self, 
        document_paths: Union[str, Path, List[Union[str, Path]]], 
        clear_existing: bool = False
    ) -> bool:
        """Ingest documents into the RAG system.
        
        Args:
            document_paths: Path(s) to documents or directories
            clear_existing: Whether to clear existing documents
            
        Returns:
            True if ingestion successful, False otherwise
        """
        if not self._initialized:
            logger.error("RAG pipeline not initialized")
            return False
        
        try:
            start_time = time.time()
            
            if clear_existing:
                logger.info("Clearing existing documents...")
                self.vector_db.clear_collection()
            
            # Convert to list if single path
            if isinstance(document_paths, (str, Path)):
                document_paths = [document_paths]
            
            all_chunks = []
            
            # Process each path
            for path in document_paths:
                path = Path(path)
                
                if path.is_file():
                    logger.info(f"Processing file: {path}")
                    chunks = self.document_processor.process_document(path)
                    all_chunks.extend(chunks)
                elif path.is_dir():
                    logger.info(f"Processing directory: {path}")
                    chunks = self.document_processor.process_directory(path)
                    all_chunks.extend(chunks)
                else:
                    logger.warning(f"Path not found: {path}")
                    continue
            
            if not all_chunks:
                logger.warning("No documents to ingest")
                return True
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            encoded_chunks = self.embedding_model.encode_chunks(all_chunks)
            
            # Store in vector database
            logger.info("Storing documents in vector database...")
            success = self.vector_db.add_documents(encoded_chunks)
            
            if success:
                ingestion_time = time.time() - start_time
                logger.info(f"Successfully ingested {len(all_chunks)} chunks in {ingestion_time:.2f}s")
                return True
            else:
                logger.error("Failed to store documents in vector database")
                return False
                
        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            return False
    
    def query(
        self,
        question: str,
        max_context_length: int = 2000,
        temperature: float = 0.7,
        use_diverse_retrieval: bool = False,
        include_context: bool = True
    ) -> RAGResponse:
        """Answer a question using RAG.
        
        Args:
            question: The question to answer
            max_context_length: Maximum length of context to include
            temperature: LLM temperature for generation
            use_diverse_retrieval: Whether to use diverse retrieval
            include_context: Whether to include surrounding context
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        if not self._initialized:
            return RAGResponse(
                answer="Error: RAG pipeline not initialized",
                retrieved_documents=[],
                query=question,
                retrieval_time=0.0,
                generation_time=0.0,
                total_time=0.0
            )
        
        try:
            # Retrieval phase
            retrieval_start = time.time()
            
            if use_diverse_retrieval:
                retrieved_docs = self.retriever.retrieve_diverse(question)
            elif include_context:
                retrieved_docs = self.retriever.retrieve_with_context(question)
            else:
                retrieved_docs = self.retriever.retrieve(question)
            
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_docs:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    retrieved_documents=[],
                    query=question,
                    retrieval_time=retrieval_time,
                    generation_time=0.0,
                    total_time=time.time() - start_time
                )
            
            # Format context
            context = self.retriever.format_context(retrieved_docs, max_context_length)
            
            # Generation phase
            generation_start = time.time()
            
            answer = self.ollama_client.generate_response(
                prompt=question,
                context=context,
                temperature=temperature
            )
            
            generation_time = time.time() - generation_start
            total_time = time.time() - start_time
            
            # Calculate confidence score based on retrieval scores
            avg_score = sum(doc.score for doc in retrieved_docs) / len(retrieved_docs)
            confidence_score = min(avg_score * 1.2, 1.0)  # Scale and cap at 1.0
            
            return RAGResponse(
                answer=answer,
                retrieved_documents=retrieved_docs,
                query=question,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                retrieved_documents=[],
                query=question,
                retrieval_time=0.0,
                generation_time=0.0,
                total_time=time.time() - start_time
            )
    
    def batch_query(
        self, 
        questions: List[str], 
        **kwargs
    ) -> List[RAGResponse]:
        """Process multiple questions in batch.
        
        Args:
            questions: List of questions to answer
            **kwargs: Additional arguments for query method
            
        Returns:
            List of RAG responses
        """
        responses = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            response = self.query(question, **kwargs)
            responses.append(response)
        
        return responses
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_context_length: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """Chat interface with conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_context_length: Maximum context length
            temperature: LLM temperature
            
        Returns:
            Generated response
        """
        if not self._initialized:
            return "Error: RAG pipeline not initialized"
        
        try:
            # Get the latest user message
            latest_message = None
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    latest_message = msg.get('content', '')
                    break
            
            if not latest_message:
                return "No user message found"
            
            # Retrieve relevant documents for the latest message
            retrieved_docs = self.retriever.retrieve(latest_message)
            context = self.retriever.format_context(retrieved_docs, max_context_length)
            
            # Add context to the conversation
            if context:
                system_message = {
                    'role': 'system',
                    'content': f"Use the following context to help answer questions:\n\n{context}"
                }
                messages = [system_message] + messages
            
            # Generate response using chat interface
            response = self.ollama_client.chat(messages, temperature)
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error: {str(e)}"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection.
        
        Returns:
            Collection statistics
        """
        try:
            collection_info = self.vector_db.get_collection_info()
            embedding_info = self.embedding_model.get_model_info()
            
            return {
                'collection_info': collection_info,
                'embedding_model': embedding_info,
                'retrieval_config': {
                    'similarity_threshold': self.retriever.similarity_threshold,
                    'max_results': self.retriever.max_results
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all RAG components.
        
        Returns:
            Health status of all components
        """
        health_status = {
            'pipeline_initialized': self._initialized,
            'ollama': self.ollama_client.health_check(),
            'retriever': self.retriever.health_check(),
            'timestamp': time.time()
        }
        
        # Determine overall status
        component_statuses = [
            health_status['ollama'].get('status') == 'healthy',
            health_status['retriever'].get('status') == 'healthy',
            self._initialized
        ]
        
        health_status['overall_status'] = 'healthy' if all(component_statuses) else 'unhealthy'
        
        return health_status