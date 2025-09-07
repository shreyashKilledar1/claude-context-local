"""Multi-language chunker that combines AST and tree-sitter approaches."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from .python_ast_chunker import PythonASTChunker, CodeChunk
from .tree_sitter_fixed import TreeSitterChunker, TreeSitterChunk

logger = logging.getLogger(__name__)


class MultiLanguageChunker:
    """Unified chunker supporting multiple programming languages."""
    
    # Supported extensions
    SUPPORTED_EXTENSIONS = {
        '.py',    # Python
        '.js',    # JavaScript  
        '.jsx',   # JSX
        '.ts',    # TypeScript
        '.tsx',   # TSX
        '.svelte' # Svelte
    }
    
    # Common large/build/tooling directories to skip during traversal
    DEFAULT_IGNORED_DIRS = {
        '__pycache__', '.git', '.hg', '.svn',
        '.venv', 'venv', 'env', '.env', '.direnv',
        'node_modules', '.pnpm-store', '.yarn',
        '.pytest_cache', '.mypy_cache', '.ruff_cache', '.pytype', '.ipynb_checkpoints',
        'build', 'dist', 'out', 'public',
        '.next', '.nuxt', '.svelte-kit', '.angular', '.astro', '.vite',
        '.cache', '.parcel-cache', '.turbo',
        'coverage', '.coverage', '.nyc_output',
        '.gradle', '.idea', '.vscode', '.docusaurus', '.vercel', '.serverless', '.terraform', '.mvn', '.tox',
        'target', 'bin', 'obj'
    }
    
    def __init__(self, root_path: Optional[str] = None):
        """Initialize multi-language chunker.
        
        Args:
            root_path: Optional root path for relative path calculation
        """
        self.root_path = root_path
        # Use AST chunker for Python (more mature implementation)
        self.python_chunker = PythonASTChunker(root_path) if root_path else None
        # Use tree-sitter for other languages
        self.tree_sitter_chunker = TreeSitterChunker()
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file type is supported.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file type is supported
        """
        suffix = Path(file_path).suffix.lower()
        return suffix in self.SUPPORTED_EXTENSIONS
    
    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        """Chunk a file into semantic units.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of CodeChunk objects
        """
        if not self.is_supported(file_path):
            logger.debug(f"File type not supported: {file_path}")
            return []
        
        suffix = Path(file_path).suffix.lower()

        # Prefer tree-sitter for Python; fallback to AST if TS not available
        if suffix == '.py':
            try:
                tree_chunks = self.tree_sitter_chunker.chunk_file(file_path)
                if tree_chunks:
                    return self._convert_tree_chunks(tree_chunks, file_path)
            except Exception as e:
                logger.warning(f"Tree-sitter chunking failed for {file_path}: {e}, falling back to AST")
            # Fallback to AST if configured
            if self.python_chunker:
                try:
                    return self.python_chunker.chunk_file(file_path)
                except Exception as e:
                    logger.error(f"AST chunking also failed for {file_path}: {e}")
                return []
        
        # Use tree-sitter for all other languages (and Python fallback)
        try:
            tree_chunks = self.tree_sitter_chunker.chunk_file(file_path)
            # Convert TreeSitterChunk to CodeChunk
            return self._convert_tree_chunks(tree_chunks, file_path)
        except Exception as e:
            logger.error(f"Failed to chunk file {file_path}: {e}")
            return []
    
    def _convert_tree_chunks(self, tree_chunks: List[TreeSitterChunk], file_path: str) -> List[CodeChunk]:
        """Convert tree-sitter chunks to CodeChunk format.
        
        Args:
            tree_chunks: List of TreeSitterChunk objects
            file_path: Path to the source file
            
        Returns:
            List of CodeChunk objects
        """
        code_chunks = []
        
        for tchunk in tree_chunks:
            # Extract metadata
            name = tchunk.metadata.get('name')
            docstring = tchunk.metadata.get('docstring')
            decorators = tchunk.metadata.get('decorators', [])
            
            # Map tree-sitter node types to our chunk types
            chunk_type_map = {
                'function_declaration': 'function',
                'function_definition': 'function',
                'arrow_function': 'function',
                'function': 'function',
                'class_declaration': 'class',
                'class_definition': 'class',
                'method_definition': 'method',
                'interface_declaration': 'interface',
                'type_alias_declaration': 'type',
                'enum_declaration': 'enum',
                'script_element': 'script',
                'style_element': 'style',
            }
            
            chunk_type = chunk_type_map.get(tchunk.node_type, tchunk.node_type)
            
            # Extract parent name and adjust chunk type for methods
            parent_name = tchunk.metadata.get('parent_name')
            
            # If we have a parent_name and it's a function, it's actually a method
            if parent_name and chunk_type == 'function':
                chunk_type = 'method'
            
            # Build folder structure from file path
            path = Path(file_path)
            folder_parts = []
            if self.root_path:
                try:
                    rel_path = path.relative_to(self.root_path)
                    folder_parts = list(rel_path.parent.parts)
                except ValueError:
                    folder_parts = [path.parent.name] if path.parent.name else []
            else:
                folder_parts = [path.parent.name] if path.parent.name else []
            
            # Extract semantic tags from metadata
            tags = []
            if tchunk.metadata.get('is_async'):
                tags.append('async')
            if tchunk.metadata.get('is_generator'):
                tags.append('generator')
            if tchunk.metadata.get('is_export'):
                tags.append('export')
            if tchunk.metadata.get('has_generics'):
                tags.append('generic')
            if tchunk.metadata.get('is_component'):
                tags.append('component')
            
            # Add language tag
            tags.append(tchunk.language)
            
            # Create CodeChunk
            chunk = CodeChunk(
                file_path=str(path),
                relative_path=str(path.relative_to(self.root_path)) if self.root_path else str(path),
                folder_structure=folder_parts,
                chunk_type=chunk_type,
                content=tchunk.content,
                start_line=tchunk.start_line,
                end_line=tchunk.end_line,
                name=name,
                parent_name=parent_name,
                docstring=docstring,
                decorators=decorators,
                imports=[],  # Tree-sitter doesn't extract imports yet
                complexity_score=0,  # Not calculated for tree-sitter chunks
                tags=tags
            )
            
            code_chunks.append(chunk)
        
        return code_chunks
    
    def chunk_directory(self, directory_path: str, extensions: Optional[List[str]] = None) -> List[CodeChunk]:
        """Chunk all supported files in a directory.
        
        Args:
            directory_path: Path to directory
            extensions: Optional list of extensions to process (default: all supported)
            
        Returns:
            List of CodeChunk objects from all files
        """
        all_chunks = []
        dir_path = Path(directory_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        # Use provided extensions or all supported
        if extensions:
            valid_extensions = set(extensions) & self.SUPPORTED_EXTENSIONS
        else:
            valid_extensions = self.SUPPORTED_EXTENSIONS
        
        # Find all files with supported extensions
        for ext in valid_extensions:
            for file_path in dir_path.rglob(f'*{ext}'):
                # Skip common large/build/tooling directories
                if any(part in self.DEFAULT_IGNORED_DIRS for part in file_path.parts):
                    continue
                
                try:
                    chunks = self.chunk_file(str(file_path))
                    all_chunks.extend(chunks)
                    logger.debug(f"Chunked {len(chunks)} from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to chunk {file_path}: {e}")
        
        logger.info(f"Total chunks from directory: {len(all_chunks)}")
        return all_chunks
