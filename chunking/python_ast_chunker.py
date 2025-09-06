"""Python AST-based intelligent code chunking."""

import ast
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CodeChunk:
    """Represents a semantically meaningful chunk of code."""
    content: str
    chunk_type: str  # function, class, method, module_level, import_block
    start_line: int
    end_line: int
    
    # Rich metadata
    file_path: str
    relative_path: str  # path relative to project root
    folder_structure: List[str]  # ['src', 'utils', 'auth'] for nested folders
    
    # Code structure metadata
    name: Optional[str] = None  # function/class name
    parent_name: Optional[str] = None  # parent class name for methods
    docstring: Optional[str] = None
    decorators: List[str] = None
    imports: List[str] = None  # relevant imports for this chunk
    
    # Context metadata
    complexity_score: int = 0  # estimated complexity
    tags: List[str] = None  # semantic tags like 'database', 'auth', 'error_handling'
    
    def __post_init__(self):
        if self.decorators is None:
            self.decorators = []
        if self.imports is None:
            self.imports = []
        if self.tags is None:
            self.tags = []
        
        # Extract folder structure from path
        if self.file_path and not self.folder_structure:
            path_parts = Path(self.relative_path).parent.parts
            self.folder_structure = list(path_parts) if path_parts != ('.',) else []


class PythonASTChunker:
    """Intelligent Python code chunker using AST parsing."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.semantic_keywords = {
            'auth': ['login', 'authenticate', 'token', 'password', 'user', 'session'],
            'database': ['db', 'query', 'sql', 'model', 'table', 'connection'],
            'api': ['request', 'response', 'endpoint', 'route', 'handler'],
            'error_handling': ['try', 'except', 'error', 'exception', 'raise'],
            'testing': ['test', 'mock', 'assert', 'fixture', 'pytest'],
            'config': ['config', 'settings', 'env', 'environment'],
        }
    
    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        """Parse a Python file and extract intelligent chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parse AST
            tree = ast.parse(source)
            
            # Get relative path for metadata
            relative_path = str(Path(file_path).relative_to(self.project_root))
            
            chunks = []
            module_imports = self._extract_imports(tree)
            
            # Extract top-level chunks
            for node in tree.body:
                chunk_list = self._extract_chunks_from_node(
                    node, source, file_path, relative_path, module_imports
                )
                chunks.extend(chunk_list)
            
            return chunks
            
        except Exception as e:
            # Fallback to simple chunking if AST parsing fails
            return self._fallback_chunk(file_path)
    
    def _extract_chunks_from_node(
        self, node: ast.AST, source: str, file_path: str, 
        relative_path: str, module_imports: List[str], parent_name: str = None
    ) -> List[CodeChunk]:
        """Extract chunks from an AST node."""
        chunks = []
        source_lines = source.split('\n')
        
        if isinstance(node, ast.FunctionDef):
            chunk = self._create_function_chunk(
                node, source_lines, file_path, relative_path, 
                module_imports, parent_name
            )
            chunks.append(chunk)
            
        elif isinstance(node, ast.ClassDef):
            # Create chunk for the class
            class_chunk = self._create_class_chunk(
                node, source_lines, file_path, relative_path, module_imports
            )
            chunks.append(class_chunk)
            
            # Extract methods as separate chunks
            for method_node in node.body:
                if isinstance(method_node, ast.FunctionDef):
                    method_chunk = self._create_function_chunk(
                        method_node, source_lines, file_path, relative_path,
                        module_imports, parent_name=node.name
                    )
                    chunks.append(method_chunk)
        
        elif isinstance(node, ast.Assign) or isinstance(node, ast.AnnAssign):
            # Module-level assignments/constants
            chunk = self._create_module_level_chunk(
                node, source_lines, file_path, relative_path, module_imports
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_function_chunk(
        self, node: ast.FunctionDef, source_lines: List[str], 
        file_path: str, relative_path: str, module_imports: List[str],
        parent_name: str = None
    ) -> CodeChunk:
        """Create a chunk for a function/method."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Extract function content with proper indentation
        content_lines = source_lines[start_line - 1:end_line]
        content = '\n'.join(content_lines)
        
        # Extract decorators
        decorators = [ast.unparse(d) for d in node.decorator_list] if node.decorator_list else []
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Calculate complexity (simple heuristic)
        complexity = self._calculate_complexity(node)
        
        # Extract semantic tags
        tags = self._extract_semantic_tags(content)
        
        chunk_type = 'method' if parent_name else 'function'
        
        return CodeChunk(
            content=content,
            chunk_type=chunk_type,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            relative_path=relative_path,
            folder_structure=[],  # Will be populated in __post_init__
            name=node.name,
            parent_name=parent_name,
            docstring=docstring,
            decorators=decorators,
            imports=module_imports,
            complexity_score=complexity,
            tags=tags
        )
    
    def _create_class_chunk(
        self, node: ast.ClassDef, source_lines: List[str],
        file_path: str, relative_path: str, module_imports: List[str]
    ) -> CodeChunk:
        """Create a chunk for a class definition."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # For classes, include only the class definition and docstring, not methods
        class_def_end = start_line
        if node.body:
            # Find where the class definition ends (before first method/attribute)
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    break
                class_def_end = item.end_lineno or item.lineno
        
        content_lines = source_lines[start_line - 1:class_def_end]
        content = '\n'.join(content_lines)
        
        # Extract decorators
        decorators = [ast.unparse(d) for d in node.decorator_list] if node.decorator_list else []
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract semantic tags
        tags = self._extract_semantic_tags(content)
        
        return CodeChunk(
            content=content,
            chunk_type='class',
            start_line=start_line,
            end_line=class_def_end,
            file_path=file_path,
            relative_path=relative_path,
            folder_structure=[],
            name=node.name,
            docstring=docstring,
            decorators=decorators,
            imports=module_imports,
            tags=tags
        )
    
    def _create_module_level_chunk(
        self, node: ast.AST, source_lines: List[str],
        file_path: str, relative_path: str, module_imports: List[str]
    ) -> Optional[CodeChunk]:
        """Create chunk for module-level code."""
        if not hasattr(node, 'lineno'):
            return None
            
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Skip trivial assignments
        if end_line - start_line < 2:
            return None
        
        content_lines = source_lines[start_line - 1:end_line]
        content = '\n'.join(content_lines)
        
        tags = self._extract_semantic_tags(content)
        
        return CodeChunk(
            content=content,
            chunk_type='module_level',
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            relative_path=relative_path,
            folder_structure=[],
            imports=module_imports,
            tags=tags
        )
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from the module."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate a simple complexity score for a code block."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        
        return complexity
    
    def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags based on content analysis."""
        content_lower = content.lower()
        tags = []
        
        for tag, keywords in self.semantic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _fallback_chunk(self, file_path: str) -> List[CodeChunk]:
        """Fallback chunking for files that can't be parsed."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            relative_path = str(Path(file_path).relative_to(self.project_root))
            
            # Simple line-based chunking
            lines = content.split('\n')
            chunk_size = 50  # lines per chunk
            chunks = []
            
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i + chunk_size]
                chunk_content = '\n'.join(chunk_lines)
                
                chunks.append(CodeChunk(
                    content=chunk_content,
                    chunk_type='fallback',
                    start_line=i + 1,
                    end_line=min(i + chunk_size, len(lines)),
                    file_path=file_path,
                    relative_path=relative_path,
                    folder_structure=[]
                ))
            
            return chunks
            
        except Exception:
            return []
    
    def chunk_directory(self, directory: str = None) -> List[CodeChunk]:
        """Chunk all Python files in a directory."""
        if directory is None:
            directory = self.project_root
        
        directory = Path(directory)
        chunks = []
        
        for py_file in directory.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            file_chunks = self.chunk_file(str(py_file))
            chunks.extend(file_chunks)
        
        return chunks
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = {
            '__pycache__', '.venv', 'venv', '.git', 'node_modules',
            '.pytest_cache', '.mypy_cache', 'build', 'dist'
        }
        
        # Check if any parent directory should be skipped
        for part in file_path.parts:
            if part in skip_patterns:
                return True
        
        # Skip test files for now (can be enabled later)
        if 'test_' in file_path.name or file_path.name.endswith('_test.py'):
            return True
        
        return False