#!/usr/bin/env python3
"""
Simple test script for RAG system functionality.
This script tests the core components without requiring external services.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_document_processing():
    """Test document processing functionality."""
    logger.info("Testing document processing...")
    
    try:
        from rag_system.utils.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test with sample document
        sample_doc = Path("./data/documents/sample_document.txt")
        if not sample_doc.exists():
            logger.error(f"Sample document not found: {sample_doc}")
            return False
        
        chunks = processor.process_document(sample_doc)
        
        if chunks:
            logger.info(f"✓ Document processing successful: {len(chunks)} chunks created")
            logger.info(f"  - First chunk length: {len(chunks[0].content)}")
            logger.info(f"  - Source: {chunks[0].source}")
            return True
        else:
            logger.error("✗ Document processing failed: no chunks created")
            return False
            
    except Exception as e:
        logger.error(f"✗ Document processing test failed: {e}")
        return False

def test_embedding_model():
    """Test embedding model functionality."""
    logger.info("Testing embedding model...")
    
    try:
        from rag_system.embeddings.embedding_model import EmbeddingModel
        
        model = EmbeddingModel()
        
        # Test text encoding
        test_text = "This is a test sentence for embedding."
        embeddings = model.encode_text(test_text)
        
        if embeddings is not None and len(embeddings.shape) > 0:
            logger.info(f"✓ Embedding model working: embedding shape {embeddings.shape}")
            logger.info(f"  - Model: {model.model_name}")
            logger.info(f"  - Dimension: {model.embedding_dimension}")
            return True
        else:
            logger.error("✗ Embedding model failed: invalid embeddings")
            return False
            
    except Exception as e:
        logger.error(f"✗ Embedding model test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        from rag_system.core.config import config
        
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  - Ollama model: {config.ollama_model}")
        logger.info(f"  - Qdrant collection: {config.qdrant_collection_name}")
        logger.info(f"  - Chunk size: {config.chunk_size}")
        logger.info(f"  - Embedding model: {config.embedding_model}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def test_integration():
    """Test basic integration without external services."""
    logger.info("Testing basic integration...")
    
    try:
        from rag_system.utils.document_processor import DocumentProcessor
        from rag_system.embeddings.embedding_model import EmbeddingModel
        
        # Process document
        processor = DocumentProcessor()
        sample_doc = Path("./data/documents/sample_document.txt")
        chunks = processor.process_document(sample_doc)
        
        if not chunks:
            logger.error("✗ Integration test failed: no chunks to process")
            return False
        
        # Generate embeddings
        embedding_model = EmbeddingModel()
        encoded_chunks = embedding_model.encode_chunks(chunks[:3])  # Test with first 3 chunks
        
        if encoded_chunks:
            logger.info(f"✓ Basic integration test successful")
            logger.info(f"  - Processed {len(chunks)} chunks")
            logger.info(f"  - Generated embeddings for {len(encoded_chunks)} chunks")
            logger.info(f"  - Embedding dimension: {len(encoded_chunks[0]['embedding'])}")
            return True
        else:
            logger.error("✗ Integration test failed: no embeddings generated")
            return False
            
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False

def run_tests():
    """Run all tests."""
    logger.info("="*60)
    logger.info("RAG SYSTEM COMPONENT TESTS")
    logger.info("="*60)
    
    tests = [
        ("Configuration", test_config),
        ("Document Processing", test_document_processing),
        ("Embedding Model", test_embedding_model),
        ("Basic Integration", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nTests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        logger.info("🎉 All core components are working correctly!")
        logger.info("\nNext steps:")
        logger.info("1. Start Qdrant: docker-compose up -d qdrant")
        logger.info("2. Install and run Ollama with Llama 3.1 model")
        logger.info("3. Test full system: python main.py health")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)