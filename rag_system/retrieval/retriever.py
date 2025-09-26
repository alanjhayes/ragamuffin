"""Document retrieval system for RAG."""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from ..embeddings.embedding_model import EmbeddingModel
from ..database.qdrant_client import QdrantVectorDB, SearchResult
from ..core.config import config

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Represents a retrieved document with context."""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    chunk_id: str

class DocumentRetriever:
    """Handles document retrieval for RAG system."""
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_db: Optional[QdrantVectorDB] = None,
        similarity_threshold: Optional[float] = None,
        max_results: Optional[int] = None
    ):
        """Initialize document retriever.
        
        Args:
            embedding_model: Embedding model for query encoding
            vector_db: Vector database for similarity search
            similarity_threshold: Minimum similarity score for results
            max_results: Maximum number of results to return
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_db = vector_db or QdrantVectorDB()
        self.similarity_threshold = similarity_threshold or config.similarity_threshold
        self.max_results = max_results or config.max_results
    
    def retrieve(
        self,
        query: str,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            limit: Maximum number of results (overrides default)
            score_threshold: Minimum similarity score (overrides default)
            filter_conditions: Additional metadata filters
            
        Returns:
            List of retrieved documents
        """
        try:
            # Use provided values or defaults
            limit = limit or self.max_results
            score_threshold = score_threshold or self.similarity_threshold
            
            logger.info(f"Retrieving documents for query: '{query[:50]}...'")
            
            # Encode query
            query_embedding = self.embedding_model.encode_text(query)
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding[0]  # Take first embedding if batch
            
            # Search vector database
            search_results = self.vector_db.search(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    content=result.content,
                    score=result.score,
                    metadata=result.metadata,
                    source=result.metadata.get('source', 'unknown'),
                    chunk_id=result.metadata.get('chunk_id', result.id)
                )
                retrieval_results.append(retrieval_result)
            
            logger.info(f"Retrieved {len(retrieval_results)} relevant documents")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def retrieve_with_reranking(
        self,
        query: str,
        initial_limit: Optional[int] = None,
        final_limit: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """Retrieve documents with optional reranking.
        
        Args:
            query: The search query
            initial_limit: Initial number of candidates to retrieve
            final_limit: Final number of results after reranking
            score_threshold: Minimum similarity score
            
        Returns:
            List of reranked documents
        """
        # Get more candidates initially
        initial_limit = initial_limit or (self.max_results * 2)
        final_limit = final_limit or self.max_results
        
        # Retrieve initial candidates
        candidates = self.retrieve(
            query=query,
            limit=initial_limit,
            score_threshold=score_threshold
        )
        
        if not candidates:
            return []
        
        # Simple reranking based on content length and score
        # More sophisticated reranking could use cross-encoders
        for result in candidates:
            # Adjust score based on content quality heuristics
            content_length = len(result.content)
            
            # Prefer medium-length chunks (not too short, not too long)
            if 200 <= content_length <= 800:
                result.score *= 1.1
            elif content_length < 100:
                result.score *= 0.9
            elif content_length > 1200:
                result.score *= 0.95
        
        # Sort by adjusted score and return top results
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:final_limit]
    
    def retrieve_diverse(
        self,
        query: str,
        limit: Optional[int] = None,
        diversity_threshold: float = 0.8
    ) -> List[RetrievalResult]:
        """Retrieve diverse documents to avoid redundancy.
        
        Args:
            query: The search query
            limit: Maximum number of results
            diversity_threshold: Minimum diversity score between results
            
        Returns:
            List of diverse documents
        """
        limit = limit or self.max_results
        
        # Get more candidates than needed
        candidates = self.retrieve(
            query=query,
            limit=limit * 2
        )
        
        if not candidates:
            return []
        
        # Select diverse results
        diverse_results = [candidates[0]]  # Always include best match
        
        for candidate in candidates[1:]:
            if len(diverse_results) >= limit:
                break
            
            # Check diversity against selected results
            is_diverse = True
            candidate_embedding = self.embedding_model.encode_text(candidate.content)[0]
            
            for selected in diverse_results:
                selected_embedding = self.embedding_model.encode_text(selected.content)[0]
                similarity = self.embedding_model.compute_similarity(
                    candidate_embedding, selected_embedding
                )
                
                if similarity > diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(candidate)
        
        logger.info(f"Selected {len(diverse_results)} diverse documents")
        return diverse_results
    
    def retrieve_with_context(
        self,
        query: str,
        include_surrounding: bool = True,
        context_window: int = 1
    ) -> List[RetrievalResult]:
        """Retrieve documents with surrounding context chunks.
        
        Args:
            query: The search query
            include_surrounding: Whether to include surrounding chunks
            context_window: Number of surrounding chunks to include
            
        Returns:
            List of documents with context
        """
        results = self.retrieve(query)
        
        if not include_surrounding or not results:
            return results
        
        # Group results by source document
        by_source: Dict[str, List[RetrievalResult]] = {}
        for result in results:
            source = result.source
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result)
        
        # For each source, try to get surrounding chunks
        enhanced_results = []
        for source, source_results in by_source.items():
            for result in source_results:
                # Add the main result
                enhanced_results.append(result)
                
                # Try to get surrounding chunks
                chunk_index = result.metadata.get('chunk_index')
                if chunk_index is not None:
                    for offset in range(-context_window, context_window + 1):
                        if offset == 0:  # Skip the main chunk
                            continue
                        
                        # Search for surrounding chunks
                        surrounding_results = self.vector_db.search(
                            query_embedding=np.zeros(self.embedding_model.embedding_dimension),
                            limit=1,
                            filter_conditions={
                                'source': source,
                                'chunk_index': chunk_index + offset
                            }
                        )
                        
                        for surrounding in surrounding_results:
                            context_result = RetrievalResult(
                                content=surrounding.content,
                                score=result.score * 0.8,  # Lower score for context
                                metadata={**surrounding.metadata, 'is_context': True},
                                source=source,
                                chunk_id=surrounding.metadata.get('chunk_id', surrounding.id)
                            )
                            enhanced_results.append(context_result)
        
        return enhanced_results
    
    def format_context(self, results: List[RetrievalResult], max_length: int = 2000) -> str:
        """Format retrieved results into context string.
        
        Args:
            results: List of retrieved results
            max_length: Maximum length of formatted context
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            # Format result
            content = result.content.strip()
            source_info = f"[Source: {result.metadata.get('filename', result.source)}]"
            
            part = f"{source_info}\n{content}\n"
            
            # Check if adding this part would exceed max_length
            if current_length + len(part) > max_length and context_parts:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "\n---\n".join(context_parts)
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of retrieval components.
        
        Returns:
            Health status information
        """
        try:
            embedding_info = self.embedding_model.get_model_info()
            vector_db_info = self.vector_db.health_check()
            
            return {
                'status': 'healthy',
                'embedding_model': embedding_info,
                'vector_database': vector_db_info,
                'similarity_threshold': self.similarity_threshold,
                'max_results': self.max_results
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }