"""Qdrant vector database client for RAG system."""

import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, SearchRequest,
    CreateCollection, UpdateCollection
)
from ..core.config import config

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result from Qdrant."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class QdrantVectorDB:
    """Qdrant vector database client for storing and retrieving embeddings."""
    
    def __init__(
        self, 
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None
    ):
        """Initialize Qdrant client.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection
            vector_size: Size of embedding vectors
        """
        self.host = host or config.qdrant_host
        self.port = port or config.qdrant_port
        self.collection_name = collection_name or config.qdrant_collection_name
        self.vector_size = vector_size or config.qdrant_vector_size
        
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Qdrant."""
        try:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)
            
            # Test connection
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant at {self.host}:{self.port}")
    
    def create_collection(self, overwrite: bool = False) -> bool:
        """Create collection if it doesn't exist.
        
        Args:
            overwrite: Whether to recreate collection if it exists
            
        Returns:
            True if collection was created/exists, False otherwise
        """
        try:
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if collection_exists:
                if overwrite:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return True
            
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Collection {self.collection_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents with embeddings to the collection.
        
        Args:
            documents: List of documents with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents to add")
            return True
        
        try:
            points = []
            for doc in documents:
                # Generate unique ID if not provided
                doc_id = doc.get('id', str(uuid.uuid4()))
                
                # Ensure embedding is numpy array
                embedding = doc['embedding']
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                # Create point
                point = PointStruct(
                    id=doc_id,
                    vector=embedding.tolist(),
                    payload={
                        'content': doc['content'],
                        'metadata': doc.get('metadata', {})
                    }
                )
                points.append(point)
            
            # Batch upsert points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully added {len(points)} documents to collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search(
        self, 
        query_embedding: Union[List[float], np.ndarray],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Additional filter conditions
            
        Returns:
            List of search results
        """
        try:
            # Convert embedding to list if numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                search_result = SearchResult(
                    id=str(result.id),
                    content=result.payload.get('content', ''),
                    score=result.score,
                    metadata=result.payload.get('metadata', {})
                )
                results.append(search_result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=document_ids
            )
            
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'name': collection_info.config.params.vectors.size,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance,
                'points_count': collection_info.points_count,
                'status': collection_info.status
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of Qdrant connection.
        
        Returns:
            Health status information
        """
        try:
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            return {
                'status': 'healthy',
                'host': self.host,
                'port': self.port,
                'collection_name': self.collection_name,
                'collection_exists': collection_exists,
                'available_collections': [c.name for c in collections.collections]
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'host': self.host,
                'port': self.port
            }
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all points and delete them
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count > 0:
                # Delete collection and recreate it
                self.client.delete_collection(self.collection_name)
                self.create_collection()
                logger.info(f"Cleared collection {self.collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False