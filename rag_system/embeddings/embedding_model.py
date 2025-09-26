"""Vector embedding model for RAG system."""

import logging
import numpy as np
from typing import List, Union, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from ..core.config import config
from ..utils.document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Handles text to vector embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name or config.embedding_model
        self.device = device or 'cpu'
        self.model = None
        self.embedding_dimension = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.embedding_dimension = test_embedding.shape[1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Encode text into vector embeddings.
        
        Args:
            text: Single text or list of texts to encode
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            if isinstance(text, str):
                text = [text]
            
            # Generate embeddings
            embeddings = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=len(text) > 10
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def encode_chunks(self, chunks: List[DocumentChunk], normalize: bool = True) -> List[Dict[str, Any]]:
        """Encode document chunks into embeddings.
        
        Args:
            chunks: List of document chunks
            normalize: Whether to normalize embeddings
            
        Returns:
            List of dictionaries containing chunk data and embeddings
        """
        if not chunks:
            return []
        
        try:
            logger.info(f"Encoding {len(chunks)} document chunks")
            
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.encode_text(texts, normalize=normalize)
            
            # Combine chunks with embeddings
            encoded_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                encoded_chunk = {
                    'id': f"{chunk.source}_{chunk.chunk_id}",
                    'content': chunk.content,
                    'embedding': embedding.tolist(),  # Convert to list for JSON serialization
                    'metadata': {
                        **chunk.metadata,
                        'embedding_model': self.model_name,
                        'embedding_dimension': self.embedding_dimension
                    }
                }
                encoded_chunks.append(encoded_chunk)
            
            logger.info(f"Successfully encoded {len(encoded_chunks)} chunks")
            return encoded_chunks
            
        except Exception as e:
            logger.error(f"Error encoding chunks: {e}")
            raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Ensure embeddings are numpy arrays
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
            
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of similarity results with indices and scores
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                similarities.append({
                    'index': i,
                    'similarity': similarity
                })
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top k results
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar embeddings: {e}")
            return []
    
    def batch_encode(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        normalize: bool = True
    ) -> np.ndarray:
        """Encode texts in batches for better performance.
        
        Args:
            texts: List of texts to encode
            batch_size: Size of each batch
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.encode_text(batch, normalize=normalize)
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            return np.vstack(all_embeddings)
            
        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'device': self.device,
            'max_sequence_length': getattr(self.model, 'max_seq_length', None) if self.model else None
        }