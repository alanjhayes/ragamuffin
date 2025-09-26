"""Main entry point for RAG system."""

import argparse
import logging
import sys
from pathlib import Path

from rag_system.rag_pipeline import RAGPipeline
from rag_system.api.app import create_app
from rag_system.core.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_cli():
    """Setup command-line interface."""
    parser = argparse.ArgumentParser(description="RAG System with Ollama and Qdrant")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', default=config.api_host, help='API host')
    api_parser.add_argument('--port', type=int, default=config.api_port, help='API port')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('paths', nargs='+', help='Document paths to ingest')
    ingest_parser.add_argument('--clear', action='store_true', help='Clear existing documents')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature')
    query_parser.add_argument('--diverse', action='store_true', help='Use diverse retrieval')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('--temperature', type=float, default=0.7, help='LLM temperature')
    
    # Health check command
    subparsers.add_parser('health', help='Check system health')
    
    # Stats command
    subparsers.add_parser('stats', help='Show collection statistics')
    
    return parser

def run_api(host: str, port: int):
    """Run the API server."""
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")

def run_ingest(paths: list, clear: bool):
    """Run document ingestion."""
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline()
    
    if not rag.initialize():
        logger.error("Failed to initialize RAG pipeline")
        return False
    
    logger.info(f"Ingesting documents from: {paths}")
    success = rag.ingest_documents(paths, clear_existing=clear)
    
    if success:
        logger.info("Document ingestion completed successfully")
        return True
    else:
        logger.error("Document ingestion failed")
        return False

def run_query(question: str, temperature: float, diverse: bool):
    """Run a single query."""
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline()
    
    if not rag.initialize():
        logger.error("Failed to initialize RAG pipeline")
        return
    
    logger.info(f"Processing query: {question}")
    response = rag.query(
        question=question,
        temperature=temperature,
        use_diverse_retrieval=diverse
    )
    
    print("\n" + "="*80)
    print(f"QUESTION: {response.query}")
    print("="*80)
    print(f"ANSWER: {response.answer}")
    print("="*80)
    print(f"Retrieved {len(response.retrieved_documents)} documents")
    print(f"Retrieval time: {response.retrieval_time:.2f}s")
    print(f"Generation time: {response.generation_time:.2f}s")
    print(f"Total time: {response.total_time:.2f}s")
    
    if response.confidence_score:
        print(f"Confidence: {response.confidence_score:.2f}")
    
    print("\nSOURCES:")
    for i, doc in enumerate(response.retrieved_documents, 1):
        print(f"{i}. {doc.source} (score: {doc.score:.3f})")

def run_chat(temperature: float):
    """Run interactive chat mode."""
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline()
    
    if not rag.initialize():
        logger.error("Failed to initialize RAG pipeline")
        return
    
    print("\n" + "="*80)
    print("RAG CHAT MODE")
    print("Type 'exit' or 'quit' to end the conversation")
    print("="*80)
    
    messages = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            response = rag.chat(messages, temperature=temperature)
            print(f"\nAssistant: {response}")
            
            messages.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            print(f"Error: {e}")

def run_health_check():
    """Run health check."""
    logger.info("Running health check...")
    rag = RAGPipeline()
    
    # Try to initialize (but don't require it for health check)
    rag.initialize()
    
    health = rag.health_check()
    
    print("\n" + "="*80)
    print("HEALTH CHECK")
    print("="*80)
    print(f"Overall Status: {health.get('overall_status', 'unknown')}")
    print(f"Pipeline Initialized: {health.get('pipeline_initialized', False)}")
    
    print("\nOllama:")
    ollama_status = health.get('ollama', {})
    print(f"  Status: {ollama_status.get('status', 'unknown')}")
    print(f"  Model: {ollama_status.get('model_name', 'unknown')}")
    print(f"  Available: {ollama_status.get('model_available', False)}")
    
    print("\nRetriever:")
    retriever_status = health.get('retriever', {})
    print(f"  Status: {retriever_status.get('status', 'unknown')}")
    
    vector_db = retriever_status.get('vector_database', {})
    print(f"  Vector DB: {vector_db.get('status', 'unknown')}")
    print(f"  Collection Exists: {vector_db.get('collection_exists', False)}")

def run_stats():
    """Show collection statistics."""
    logger.info("Getting collection statistics...")
    rag = RAGPipeline()
    
    if not rag.initialize():
        logger.error("Failed to initialize RAG pipeline")
        return
    
    stats = rag.get_collection_stats()
    
    print("\n" + "="*80)
    print("COLLECTION STATISTICS")
    print("="*80)
    
    collection_info = stats.get('collection_info', {})
    print(f"Collection: {config.qdrant_collection_name}")
    print(f"Documents: {collection_info.get('points_count', 0)}")
    print(f"Vector Size: {collection_info.get('vector_size', 0)}")
    print(f"Status: {collection_info.get('status', 'unknown')}")
    
    embedding_model = stats.get('embedding_model', {})
    print(f"\nEmbedding Model: {embedding_model.get('model_name', 'unknown')}")
    print(f"Embedding Dimension: {embedding_model.get('embedding_dimension', 0)}")
    
    retrieval_config = stats.get('retrieval_config', {})
    print(f"\nRetrieval Config:")
    print(f"  Similarity Threshold: {retrieval_config.get('similarity_threshold', 0)}")
    print(f"  Max Results: {retrieval_config.get('max_results', 0)}")

def main():
    """Main entry point."""
    parser = setup_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'api':
            run_api(args.host, args.port)
        elif args.command == 'ingest':
            success = run_ingest(args.paths, args.clear)
            sys.exit(0 if success else 1)
        elif args.command == 'query':
            run_query(args.question, args.temperature, args.diverse)
        elif args.command == 'chat':
            run_chat(args.temperature)
        elif args.command == 'health':
            run_health_check()
        elif args.command == 'stats':
            run_stats()
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()