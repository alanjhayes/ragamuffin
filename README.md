# RAG System with Ollama and Qdrant

A complete Retrieval Augmented Generation (RAG) system built with Ollama for local LLM inference and Qdrant as the vector database, running in Docker.

## Features

- **Local LLM Integration**: Uses Ollama for running Llama 3.1 models locally
- **Vector Search**: Qdrant vector database for efficient similarity search
- **Document Processing**: Supports PDF, TXT, and Markdown files
- **Smart Chunking**: Intelligent text chunking with overlap for better context
- **Multiple Interfaces**: CLI, REST API, and chat modes
- **Configurable**: Extensive configuration options via environment variables
- **Scalable**: Containerized deployment with Docker Compose

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Ollama installed locally

### 2. Setup

1. **Clone and navigate to the project**:
   ```bash
   cd ragamuffin
   ```

2. **Start Qdrant with Docker Compose**:
   ```bash
   docker-compose up -d qdrant
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Install and start Ollama** (if not already installed):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Llama 3.1 model
   ollama pull llama3.1:8b
   ```

### 3. Usage

#### Command Line Interface

**Ingest documents**:
```bash
python main.py ingest ./data/documents --clear
```

**Query the system**:
```bash
python main.py query "What is retrieval augmented generation?"
```

**Interactive chat**:
```bash
python main.py chat
```

**Check system health**:
```bash
python main.py health
```

**View collection statistics**:
```bash
python main.py stats
```

#### REST API

**Start the API server**:
```bash
python main.py api --host 0.0.0.0 --port 8000
```

**API Endpoints**:
- `GET /` - API information
- `POST /query` - Query the RAG system
- `POST /chat` - Chat interface
- `POST /ingest` - Ingest documents
- `GET /health` - Health check
- `GET /stats` - Collection statistics
- `POST /clear` - Clear all documents

**Example API usage**:
```bash
# Query
curl -X POST "http://localhost:8000/query" \\
     -H "Content-Type: application/json" \\
     -d '{"question": "What is RAG?"}'

# Health check
curl "http://localhost:8000/health"
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │   RAG Pipeline  │    │   Responses     │
│                 │    │                 │    │                 │
│ PDF, TXT, MD    │───▶│ 1. Chunking     │───▶│ Answers with    │
│ Files           │    │ 2. Embedding    │    │ Sources         │
│                 │    │ 3. Storage      │    │                 │
└─────────────────┘    │ 4. Retrieval    │    └─────────────────┘
                       │ 5. Generation   │
                       └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼──┐ ┌────▼────┐ ┌──▼────┐
            │ Qdrant   │ │ Sentence│ │Ollama │
            │ Vector   │ │Transform│ │Llama  │
            │ Database │ │Embedding│ │ 3.1   │
            └──────────┘ └─────────┘ └───────┘
```

## Configuration

Configure the system via environment variables in `.env`:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_documents
QDRANT_VECTOR_SIZE=384

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=500

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## Project Structure

```
ragamuffin/
├── rag_system/
│   ├── core/                 # Core configuration and Ollama client
│   ├── embeddings/           # Sentence transformer embeddings
│   ├── database/             # Qdrant vector database integration
│   ├── retrieval/            # Document retrieval system
│   ├── utils/                # Document processing utilities
│   ├── api/                  # FastAPI REST interface
│   └── rag_pipeline.py       # Main RAG pipeline
├── data/
│   ├── documents/            # Input documents
│   └── processed/            # Processed data cache
├── tests/                    # Test suite
├── docker-compose.yml        # Docker services
├── requirements.txt          # Python dependencies
├── main.py                   # CLI entry point
└── README.md                 # This file
```

## Features in Detail

### Document Processing
- **Multi-format Support**: PDF, TXT, Markdown
- **Smart Chunking**: Sentence-boundary aware chunking
- **Metadata Preservation**: Source tracking and chunk indexing
- **Configurable Parameters**: Chunk size and overlap

### Vector Search
- **Qdrant Integration**: High-performance vector database
- **Cosine Similarity**: Efficient similarity search
- **Metadata Filtering**: Search with additional constraints
- **Diverse Retrieval**: Anti-redundancy mechanisms

### Generation
- **Local LLM**: Ollama with Llama 3.1
- **Context-Aware**: RAG prompting with retrieved context
- **Configurable**: Temperature and token limits
- **Chat Mode**: Conversation history support

### API Features
- **RESTful Design**: Standard HTTP methods
- **Request Validation**: Pydantic models
- **Background Processing**: Async document ingestion
- **Health Monitoring**: Comprehensive health checks
- **CORS Support**: Cross-origin requests enabled

## Development

### Adding New Document Types
1. Extend `DocumentProcessor` in `rag_system/utils/document_processor.py`
2. Add new file type handling in `load_document` method
3. Update supported extensions list

### Custom Embedding Models
1. Modify `EMBEDDING_MODEL` in configuration
2. Ensure model compatibility with sentence-transformers
3. Update vector size in Qdrant configuration

### Custom Retrieval Strategies
1. Extend `DocumentRetriever` in `rag_system/retrieval/retriever.py`
2. Implement new retrieval methods
3. Add configuration options

## Troubleshooting

### Common Issues

**Ollama Connection Failed**:
- Ensure Ollama is running: `ollama serve`
- Check model availability: `ollama list`
- Verify URL in configuration

**Qdrant Connection Failed**:
- Start Qdrant: `docker-compose up -d qdrant`
- Check Qdrant health: `curl http://localhost:6333/health`
- Verify port configuration

**Memory Issues**:
- Reduce chunk size and batch size
- Use smaller embedding models
- Limit max context length

**Performance Issues**:
- Enable GPU for embeddings if available
- Tune retrieval parameters
- Optimize chunk sizes

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

For questions or issues, please open a GitHub issue.