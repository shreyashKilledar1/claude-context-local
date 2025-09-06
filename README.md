# Claude Code Embedding Search

An intelligent code search system that uses Google's EmbeddingGemma model and AST-based chunking to provide semantic search capabilities for Python codebases, integrated with Claude Code via MCP (Model Context Protocol).

## Features

- **Intelligent AST-based chunking**: Preserves function and class boundaries
- **Semantic search**: Natural language queries to find code
- **Rich metadata**: File paths, folder structure, semantic tags
- **MCP integration**: Direct integration with Claude Code
- **Local processing**: All embeddings stored locally, no API calls
- **Fast search**: FAISS for efficient similarity search

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   cd claude_embedding_search
   uv sync
   ```

## Quick Start

### 1. Index Your Codebase

```bash
# Index a Python project
./scripts/index_codebase.py /path/to/your/python/project

# With custom storage location
./scripts/index_codebase.py /path/to/project --storage-dir /custom/location

# Clear existing index and reindex
./scripts/index_codebase.py /path/to/project --clear
```

### 2. Add MCP Server to Claude Code

```bash
# Add the server globally (available in all projects)
claude mcp add code-search --scope user -- python /full/path/to/mcp_server/server.py

# Or add locally to current project
claude mcp add code-search -- python /full/path/to/mcp_server/server.py
```

### 3. Use in Claude Code

Once configured, you can use natural language queries in Claude Code:

- "Find authentication functions"
- "Show database connection code"
- "Find error handling patterns"
- "Look for API endpoint definitions"

## Architecture

```
claude_embedding_search/
├── chunking/                   # AST-based code chunking
│   └── python_ast_chunker.py  # Python-specific chunking logic
├── embeddings/                 # EmbeddingGemma integration
│   └── embedder.py            # Embedding generation
├── search/                     # Search functionality
│   ├── indexer.py             # FAISS index management
│   └── searcher.py            # Intelligent search logic
├── mcp_server/                 # Claude Code integration
│   └── server.py              # MCP server implementation
└── scripts/                    # Command-line tools
    └── index_codebase.py      # Indexing utility
```

## Intelligent Chunking

The system uses AST parsing to create semantically meaningful chunks:

- **Complete functions** with docstrings and decorators
- **Full classes** with all methods as separate chunks
- **Module-level code** blocks and constants
- **Preserved context** with imports and parent references

Each chunk includes rich metadata:
- File path and folder structure
- Function/class names and relationships
- Semantic tags (auth, database, api, etc.)
- Complexity scores
- Line numbers for precise location

## Search Features

### Natural Language Queries
```python
# These all work:
"Find user authentication code"
"Show database query functions" 
"API endpoint handlers"
"Error handling patterns"
```

### Advanced Filtering
```python
# Search specific file patterns
search_code("authentication", file_pattern="auth")

# Search specific code types
search_code("validation", chunk_type="function")

# Include contextual information
search_code("login", include_context=True)
```

### Similar Code Discovery
```python
# Find code similar to a specific function
find_similar_code("auth.py:45-67:function:login")
```

## MCP Tools Available in Claude Code

Once integrated, Claude Code will have access to these tools:

### `search_code(query, k=5, ...)`
Search for code using natural language.

### `index_directory(directory_path, ...)`
Index a new Python project.

### `find_similar_code(chunk_id, k=5)`
Find code similar to a specific chunk.

### `get_index_status()`
Check index statistics and model status.

### `clear_index()`
Clear the search index.

## Configuration

### Environment Variables

- `CODE_SEARCH_STORAGE`: Custom storage directory (default: `~/.claude_code_search`)

### Model Configuration

The system uses `google/embeddinggemma-300m` by default. The model will be automatically downloaded on first use.

## Storage

Data is stored in the configured storage directory:

```
~/.claude_code_search/
├── models/          # Downloaded models
├── index/           # FAISS indices and metadata
│   ├── code.index   # Vector index
│   ├── metadata.db  # Chunk metadata (SQLite)
│   └── stats.json   # Index statistics
```

## Performance

- **Model size**: ~300MB (EmbeddingGemma-300m)
- **Embedding dimension**: 768 (can be reduced for speed)
- **Index types**: Flat (exact) or IVF (approximate) based on dataset size
- **Batch processing**: Configurable batch sizes for embedding generation

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed with `uv sync`
2. **Model download fails**: Check internet connection and disk space
3. **Memory issues**: Reduce batch size in indexing script
4. **No search results**: Verify the codebase was indexed successfully

### Debugging

Enable verbose logging:
```bash
./scripts/index_codebase.py /path/to/project --verbose
```

Check index status:
```python
# In Claude Code
get_index_status()
```

## Examples

### Indexing a Django Project
```bash
./scripts/index_codebase.py /path/to/django_project
```

### Searching for Specific Patterns
```python
# In Claude Code
search_code("user authentication middleware")
search_code("database models with foreign keys") 
search_code("API serialization logic")
```

### Finding Similar Functions
```python
# First find a function
results = search_code("login validation")
# Then find similar ones
find_similar_code(results[0]['chunk_id'])
```

## Contributing

This is a research project focused on intelligent code chunking and search. Feel free to experiment with:

- Different chunking strategies
- Alternative embedding models  
- Enhanced metadata extraction
- Performance optimizations

## License

MIT License - feel free to modify and use as needed.