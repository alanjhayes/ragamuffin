"""Document processing utilities for RAG system."""

import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import pypdf
from ..core.config import config

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    start_index: int
    end_index: int

class DocumentProcessor:
    """Handles document loading, cleaning, and chunking."""
    
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        """Initialize document processor.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def load_document(self, file_path: Union[str, Path]) -> str:
        """Load document from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document content as string
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._load_pdf(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                return self._load_text(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return self._load_text(file_path)  # Try as text anyway
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load PDF document."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
        
        return text
    
    def _load_text(self, file_path: Path) -> str:
        """Load text document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove multiple consecutive periods
        text = re.sub(r'\.{3,}', '...', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            metadata: Additional metadata for chunks
            
        Returns:
            List of document chunks
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                
                if sentence_end > start:
                    end = sentence_end
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata={
                        **metadata,
                        'chunk_index': chunk_id,
                        'total_chunks': None,  # Will be set after all chunks are created
                        'char_count': len(chunk_content)
                    },
                    chunk_id=f"chunk_{chunk_id}",
                    source=metadata.get('source', 'unknown'),
                    start_index=start,
                    end_index=end
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Ensure we don't go backwards
            if start <= chunks[-1].start_index if chunks else 0:
                start = end
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find a good sentence boundary within the given range.
        
        Args:
            text: Full text
            start: Start position to search from
            end: End position to search to
            
        Returns:
            Position of sentence boundary, or end if none found
        """
        # Look for sentence endings
        sentence_endings = ['.', '!', '?']
        
        for i in range(end - 1, start - 1, -1):
            if text[i] in sentence_endings:
                # Check if next character is whitespace or end of text
                if i + 1 >= len(text) or text[i + 1].isspace():
                    return i + 1
        
        # If no sentence boundary found, look for paragraph breaks
        for i in range(end - 1, start - 1, -1):
            if text[i] == '\n':
                return i + 1
        
        return end
    
    def process_document(self, file_path: Union[str, Path], additional_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Complete document processing pipeline.
        
        Args:
            file_path: Path to document file
            additional_metadata: Additional metadata to include
            
        Returns:
            List of processed document chunks
        """
        file_path = Path(file_path)
        
        # Load document
        logger.info(f"Loading document: {file_path}")
        raw_text = self.load_document(file_path)
        
        # Clean text
        logger.info("Cleaning text")
        clean_text = self.clean_text(raw_text)
        
        # Create metadata
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_type': file_path.suffix.lower(),
            'original_length': len(raw_text),
            'cleaned_length': len(clean_text),
            **(additional_metadata or {})
        }
        
        # Chunk text
        logger.info("Chunking text")
        chunks = self.chunk_text(clean_text, metadata)
        
        logger.info(f"Created {len(chunks)} chunks from document {file_path}")
        return chunks
    
    def process_directory(self, directory_path: Union[str, Path], file_extensions: Optional[List[str]] = None) -> List[DocumentChunk]:
        """Process all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            file_extensions: List of file extensions to process (e.g., ['.txt', '.pdf'])
            
        Returns:
            List of all document chunks
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        file_extensions = file_extensions or ['.txt', '.pdf', '.md']
        all_chunks = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    chunks = self.process_document(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
        
        logger.info(f"Processed {len(all_chunks)} chunks from directory {directory_path}")
        return all_chunks