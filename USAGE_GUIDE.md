# Claude Code Search - Usage Guide

## 🚀 Installation & Setup

### 1. Install in Claude Code

```bash
# Global installation (available in all projects)
claude mcp add code-search --scope user -- python /full/path/to/mcp_server/server.py

# Project-specific installation
cd /your/project
claude mcp add code-search -- python /full/path/to/mcp_server/server.py
```

### 2. Verify Installation
```bash
claude mcp list  # Should show "code-search"
```

## 📁 Project-Specific Storage

Each project gets its own isolated storage directory:

```
~/.claude_code_search/
├── models/                           # Shared AI models (~300MB, downloaded once)
└── projects/                         # Project-specific indices
    ├── my-web-app_abc12345/          # Project: /path/to/my-web-app
    │   ├── project_info.json         # Project metadata
    │   └── index/
    │       ├── code.index            # Vector embeddings (FAISS)
    │       ├── metadata.db           # Code metadata (SQLite)
    │       ├── chunk_ids.pkl         # Chunk identifiers
    │       └── stats.json            # Index statistics
    └── api-service_def67890/         # Project: /path/to/api-service
        ├── project_info.json
        └── index/
            ├── code.index
            ├── metadata.db
            ├── chunk_ids.pkl
            └── stats.json
```

**Benefits:**
- ✅ **Fast search** - Only searches within relevant project
- ✅ **No conflicts** - Projects don't interfere with each other  
- ✅ **Easy switching** - Switch between projects instantly
- ✅ **Independent management** - Index/clear projects separately
- ✅ **Readable names** - Directory names include project name + hash

## 🔄 Typical Workflow

### Option 1: Try the Demo First
```bash
# 1. Index the built-in sample project
index_test_project()

# 2. Explore what's available
list_projects()

# 3. Search the demo project
search_code("user authentication functions")
search_code("database connection code") 
search_code("HTTP API handlers")
search_code("input validation", chunk_type="function")
```

### Option 2: Index Your Own Project
```bash
# 1. Index your project (creates project-specific storage)
index_directory("/path/to/your/python/project")

# 2. Start searching immediately
search_code("authentication logic")
search_code("database queries", chunk_type="function")
search_code("error handling", file_pattern="utils")

# 3. Check what was indexed
get_index_status()
```

### Option 3: Multiple Projects
```bash
# 1. Index first project
index_directory("/path/to/project-a")
search_code("user login")  # Searches in project-a

# 2. Index second project (automatically switches to it)
index_directory("/path/to/project-b") 
search_code("database")    # Searches in project-b

# 3. Switch back to first project
switch_project("/path/to/project-a")
search_code("authentication")  # Back to searching project-a

# 4. List all your projects
list_projects()
```

## 🛠 Available Tools

### Core Operations
- `index_directory(path)` - Index a Python project
- `search_code(query, k=5, ...)` - Search for code using natural language
- `get_index_status()` - Check current project status

### Project Management  
- `list_projects()` - Show all indexed projects
- `switch_project(path)` - Switch to a different project
- `clear_index()` - Clear current project's index

### Demo & Examples
- `index_test_project()` - Index built-in sample project
- `demo_search_examples()` - Run example searches (coming soon)

### Advanced
- `find_similar_code(chunk_id)` - Find similar code chunks
- `get_search_statistics()` - Get detailed index statistics

## 🔍 Search Examples

### Natural Language Queries
```bash
search_code("user authentication functions")
search_code("database connection and transaction handling") 
search_code("HTTP request validation")
search_code("error handling patterns")
search_code("configuration management utilities")
search_code("password hashing and security functions")
```

### Filtered Searches
```bash
# Search only functions
search_code("validation logic", chunk_type="function")

# Search only classes  
search_code("exception handling", chunk_type="class")

# Search specific modules
search_code("user management", file_pattern="auth")
search_code("database operations", file_pattern="db")
```

### Combined Filters
```bash
search_code("input validation", 
           chunk_type="function", 
           file_pattern="utils", 
           k=10)
```

## ⚡ Performance & Storage

### Indexing Performance
- **Small projects** (100-500 files): ~30 seconds
- **Medium projects** (500-2000 files): 1-3 minutes  
- **Large projects** (2000+ files): 3-10 minutes

### Search Performance
- **After indexing**: Instant (<100ms)
- **No re-indexing needed** unless code changes significantly

### Storage Usage
- **AI Model**: ~300MB (downloaded once, shared across projects)
- **Per project**: ~1-10MB depending on project size
- **Metadata**: Minimal (few MB per project)

## 🎯 When to Re-index

You only need to re-index when:
- ✅ **Major code changes** (new modules, refactoring)
- ✅ **New functionality** you want to search for
- ✅ **Project structure changes** 

You don't need to re-index for:
- ❌ Minor bug fixes
- ❌ Variable name changes
- ❌ Comment updates
- ❌ Documentation changes

## 🤖 How It Works

1. **AST Parsing**: Analyzes Python code structure (functions, classes, methods)
2. **Semantic Chunking**: Creates meaningful code chunks with context
3. **AI Embeddings**: Converts code to 768-dimensional vectors using Google's EmbeddingGemma
4. **Vector Search**: Uses FAISS for fast similarity search
5. **Intelligent Filtering**: Combines embeddings with metadata filtering

## 💡 Tips & Best Practices

### Search Tips
- **Use natural language** - describe what the code does, not variable names
- **Start broad, then narrow** - "authentication" → "user login validation" 
- **Combine with filters** - improves precision significantly
- **Use context** - the system understands semantic relationships

### Project Management
- **Index when starting work** on a new codebase
- **Re-index after major changes** or when you can't find what you expect
- **Use multiple projects** to keep different codebases separate
- **Switch projects** as needed - it's instant

### Example Workflow
```bash
# Morning: Working on web app
switch_project("/projects/my-web-app")
search_code("user authentication middleware")
search_code("database migrations", chunk_type="function")

# Afternoon: Switch to API service  
switch_project("/projects/api-service")
search_code("request validation")
search_code("error handling", file_pattern="handlers")

# Check what's indexed across projects
list_projects()
```

This approach gives you fast, isolated search for each project while sharing the AI model efficiently!