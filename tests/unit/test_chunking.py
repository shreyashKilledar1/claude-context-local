"""Unit tests for AST-based code chunking."""

import pytest
from pathlib import Path
from chunking.python_ast_chunker import PythonASTChunker, CodeChunk


class TestPythonASTChunker:
    """Test cases for PythonASTChunker."""

    def test_init(self, temp_project_dir):
        """Test chunker initialization."""
        chunker = PythonASTChunker(str(temp_project_dir))
        assert chunker.project_root == temp_project_dir
        assert hasattr(chunker, 'semantic_keywords')
        assert 'auth' in chunker.semantic_keywords
        assert 'database' in chunker.semantic_keywords

    def test_chunk_simple_function(self, chunker, temp_project_dir):
        """Test chunking a simple function."""
        # Create a simple Python file
        test_file = temp_project_dir / "simple.py"
        test_code = '''
def hello_world(name):
    """Greet someone."""
    return f"Hello, {name}!"
'''
        test_file.write_text(test_code)
        
        chunks = chunker.chunk_file(str(test_file))
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert chunk.chunk_type == 'function'
        assert chunk.name == 'hello_world'
        assert chunk.start_line == 2
        assert chunk.end_line == 4
        assert chunk.docstring == 'Greet someone.'
        assert 'Hello, {name}!' in chunk.content

    def test_chunk_class_with_methods(self, chunker, temp_project_dir):
        """Test chunking a class with methods."""
        test_file = temp_project_dir / "class_test.py"
        test_code = '''
class Calculator:
    """A simple calculator."""
    
    def __init__(self):
        """Initialize calculator."""
        self.result = 0
    
    def add(self, value):
        """Add value to result."""
        self.result += value
        return self.result
    
    def multiply(self, value):
        """Multiply result by value."""
        self.result *= value
        return self.result
'''
        test_file.write_text(test_code)
        
        chunks = chunker.chunk_file(str(test_file))
        
        # Should have: class definition + 3 methods
        assert len(chunks) == 4
        
        # Check class chunk
        class_chunk = next(c for c in chunks if c.chunk_type == 'class')
        assert class_chunk.name == 'Calculator'
        assert class_chunk.docstring == 'A simple calculator.'
        
        # Check method chunks
        method_chunks = [c for c in chunks if c.chunk_type == 'method']
        assert len(method_chunks) == 3
        
        method_names = {c.name for c in method_chunks}
        assert method_names == {'__init__', 'add', 'multiply'}
        
        # Check parent references
        for method_chunk in method_chunks:
            assert method_chunk.parent_name == 'Calculator'

    def test_semantic_tag_extraction(self, chunker, temp_project_dir):
        """Test extraction of semantic tags."""
        test_file = temp_project_dir / "auth_test.py"
        test_code = '''
import hashlib

def authenticate_user(username, password):
    """Authenticate user with password."""
    if not username or not password:
        raise ValueError("Invalid credentials")
    
    # Hash the password
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Database query
    user = db.query("SELECT * FROM users WHERE username = ?", username)
    return user is not None

def handle_login_error(error):
    """Handle login errors and exceptions."""
    try:
        log_error(error)
    except Exception as e:
        raise RuntimeError("Error handling failed")
'''
        test_file.write_text(test_code)
        
        chunks = chunker.chunk_file(str(test_file))
        
        assert len(chunks) == 2
        
        # Check authentication function tags
        auth_chunk = next(c for c in chunks if c.name == 'authenticate_user')
        assert 'auth' in auth_chunk.tags
        assert 'database' in auth_chunk.tags
        assert 'error_handling' in auth_chunk.tags
        
        # Check error handling function tags
        error_chunk = next(c for c in chunks if c.name == 'handle_login_error')
        assert 'error_handling' in error_chunk.tags

    def test_decorator_extraction(self, chunker, temp_project_dir):
        """Test extraction of function decorators."""
        test_file = temp_project_dir / "decorators_test.py"
        test_code = '''
@login_required
@admin_only  
def delete_user(user_id):
    """Delete a user (admin only)."""
    return db.delete('users', user_id)

@property
def is_admin(self):
    """Check if user is admin."""
    return 'admin' in self.roles

@staticmethod
@validate_input
def hash_password(password):
    """Hash a password."""
    return bcrypt.hash(password)
'''
        test_file.write_text(test_code)
        
        chunks = chunker.chunk_file(str(test_file))
        
        # Find chunks by name
        delete_chunk = next(c for c in chunks if c.name == 'delete_user')
        admin_chunk = next(c for c in chunks if c.name == 'is_admin')
        hash_chunk = next(c for c in chunks if c.name == 'hash_password')
        
        # Check decorators
        assert set(delete_chunk.decorators) == {'login_required', 'admin_only'}
        assert 'property' in admin_chunk.decorators
        assert set(hash_chunk.decorators) == {'staticmethod', 'validate_input'}

    def test_complexity_calculation(self, chunker, temp_project_dir):
        """Test calculation of code complexity."""
        test_file = temp_project_dir / "complexity_test.py"
        test_code = '''
def simple_function():
    """Simple function with no branching."""
    return "hello"

def complex_function(data):
    """Function with multiple branches and loops."""
    result = []
    
    for item in data:
        if item > 0:
            if item % 2 == 0:
                result.append(item * 2)
            else:
                result.append(item + 1)
        else:
            try:
                processed = process_negative(item)
                result.append(processed)
            except ValueError:
                continue
            except Exception as e:
                handle_error(e)
    
    return result
'''
        test_file.write_text(test_code)
        
        chunks = chunker.chunk_file(str(test_file))
        
        simple_chunk = next(c for c in chunks if c.name == 'simple_function')
        complex_chunk = next(c for c in chunks if c.name == 'complex_function')
        
        # Simple function should have low complexity
        assert simple_chunk.complexity_score == 1
        
        # Complex function should have higher complexity (loops, ifs, try-catch)
        assert complex_chunk.complexity_score > 5

    def test_folder_structure_metadata(self, chunker, sample_codebase):
        """Test extraction of folder structure metadata."""
        # Get one of the sample files
        auth_file = sample_codebase['auth']
        
        chunks = chunker.chunk_file(str(auth_file))
        
        # All chunks should have correct folder structure
        for chunk in chunks:
            assert chunk.folder_structure == ['src', 'auth']
            assert chunk.relative_path == 'src/auth/authenticator.py'
            assert chunk.file_path == str(auth_file)

    def test_import_extraction(self, chunker, sample_codebase):
        """Test extraction of import statements."""
        auth_file = sample_codebase['auth']
        
        chunks = chunker.chunk_file(str(auth_file))
        
        # All chunks should have the same imports from the module
        expected_imports = [
            'hashlib', 'logging', 'typing.Optional', 'typing.Dict', 
            'typing.List', 'datetime.datetime', 'datetime.timedelta'
        ]
        
        for chunk in chunks:
            # Should contain some of the expected imports
            assert len(chunk.imports) > 0
            # Check that at least some expected imports are present
            import_names = [imp.split('.')[-1] for imp in chunk.imports]
            assert any(name in import_names for name in ['hashlib', 'logging', 'datetime'])

    def test_chunk_directory(self, chunker, sample_codebase):
        """Test chunking entire directory."""
        chunks = chunker.chunk_directory()
        
        # Should have chunks from all sample files
        assert len(chunks) > 10  # We expect many chunks from sample codebase
        
        # Check that we have chunks from different files
        file_paths = {chunk.relative_path for chunk in chunks}
        expected_files = {
            'src/auth/authenticator.py',
            'src/database/manager.py', 
            'src/api/endpoints.py',
            'src/utils/helpers.py'
        }
        
        assert expected_files.issubset(file_paths)
        
        # Check chunk types diversity
        chunk_types = {chunk.chunk_type for chunk in chunks}
        assert 'function' in chunk_types
        assert 'class' in chunk_types
        assert 'method' in chunk_types

    def test_fallback_chunking(self, chunker, temp_project_dir):
        """Test fallback chunking for unparseable files."""
        # Create a file with syntax errors
        bad_file = temp_project_dir / "bad_syntax.py" 
        bad_file.write_text("def broken_function(\n    # Missing closing parenthesis")
        
        chunks = chunker.chunk_file(str(bad_file))
        
        # Should still produce chunks via fallback method
        assert len(chunks) > 0
        
        # All chunks should be marked as fallback type
        for chunk in chunks:
            assert chunk.chunk_type == 'fallback'

    def test_should_skip_file(self, chunker, temp_project_dir):
        """Test file skipping logic."""
        # Create files that should be skipped
        skip_files = [
            temp_project_dir / "__pycache__" / "test.py",
            temp_project_dir / ".venv" / "lib" / "test.py", 
            temp_project_dir / "test_something.py",
            temp_project_dir / "something_test.py"
        ]
        
        for file_path in skip_files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            assert chunker._should_skip_file(file_path)
        
        # Create file that should not be skipped
        good_file = temp_project_dir / "src" / "good_file.py"
        good_file.parent.mkdir(parents=True, exist_ok=True)
        assert not chunker._should_skip_file(good_file)

    def test_module_level_code_chunking(self, chunker, temp_project_dir):
        """Test chunking of module-level code."""
        test_file = temp_project_dir / "module_level.py"
        test_code = '''
import os
import sys

# Configuration constants
DEBUG = True
API_VERSION = "v1"
MAX_RETRIES = 3

# Complex module-level setup
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

def regular_function():
    """A regular function."""
    pass
'''
        test_file.write_text(test_code)
        
        chunks = chunker.chunk_file(str(test_file))
        
        # Should have the function and some module-level chunks
        chunk_types = [c.chunk_type for c in chunks]
        assert 'function' in chunk_types
        
        # Check that we have the function
        func_chunks = [c for c in chunks if c.chunk_type == 'function']
        assert len(func_chunks) == 1
        assert func_chunks[0].name == 'regular_function'


class TestCodeChunk:
    """Test cases for CodeChunk dataclass."""
    
    def test_code_chunk_creation(self):
        """Test creation of CodeChunk instances."""
        chunk = CodeChunk(
            content="def test(): pass",
            chunk_type="function",
            start_line=1,
            end_line=1,
            file_path="/test/file.py",
            relative_path="file.py",
            folder_structure=[],
            name="test"
        )
        
        assert chunk.content == "def test(): pass"
        assert chunk.chunk_type == "function"
        assert chunk.name == "test"
        assert chunk.decorators == []  # Should be initialized by __post_init__
        assert chunk.imports == []
        assert chunk.tags == []

    def test_folder_structure_extraction(self):
        """Test automatic folder structure extraction."""
        chunk = CodeChunk(
            content="def test(): pass",
            chunk_type="function", 
            start_line=1,
            end_line=1,
            file_path="/project/src/auth/handlers.py",
            relative_path="src/auth/handlers.py",
            folder_structure=[]  # Will be auto-populated
        )
        
        # __post_init__ should extract folder structure from relative_path
        assert chunk.folder_structure == ['src', 'auth']

    def test_chunk_with_all_metadata(self):
        """Test chunk with complete metadata."""
        chunk = CodeChunk(
            content='@login_required\ndef authenticate(user):\n    """Auth user."""\n    return True',
            chunk_type="function",
            start_line=5,
            end_line=8, 
            file_path="/project/auth.py",
            relative_path="auth.py",
            folder_structure=[],
            name="authenticate",
            parent_name=None,
            docstring="Auth user.",
            decorators=["login_required"],
            imports=["functools", "hashlib"],
            complexity_score=3,
            tags=["auth", "security"]
        )
        
        assert chunk.name == "authenticate"
        assert chunk.docstring == "Auth user."
        assert "login_required" in chunk.decorators
        assert "auth" in chunk.tags
        assert chunk.complexity_score == 3
