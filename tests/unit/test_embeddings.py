"""Unit tests for embedding generation."""

import pytest
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock
from embeddings.embedder import CodeEmbedder, EmbeddingResult
from chunking.python_ast_chunker import CodeChunk


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for tests."""
    with patch('embeddings.embedder.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        mock_st.return_value = mock_model
        yield mock_st


@pytest.fixture 
def embedder_with_mock(mock_sentence_transformer, mock_storage_dir):
    """Create a CodeEmbedder with mocked dependencies."""
    embedder = CodeEmbedder(
        model_name="test/model", 
        cache_dir=str(mock_storage_dir)
    )
    return embedder


class TestCodeEmbedder:
    """Test cases for CodeEmbedder."""
    
    def test_init(self, test_config):
        """Test embedder initialization."""
        embedder = CodeEmbedder(
            model_name="test/model",
            cache_dir="/tmp/cache",
            device="cpu"
        )
        
        assert embedder.model_name == "test/model"
        assert embedder.cache_dir == "/tmp/cache"
        assert embedder.device == "cpu"
        assert embedder._model is None  # Lazy loading

    def test_model_loading(self, embedder_with_mock):
        """Test lazy model loading."""
        # Model should not be loaded yet
        assert embedder_with_mock._model is None
        
        # Access model property - this should trigger loading  
        model = embedder_with_mock.model
        assert model is not None
        assert embedder_with_mock._model is not None
        
        # Should use cached model on second access
        model2 = embedder_with_mock.model
        assert model2 is model

    @patch('embeddings.embedder.SentenceTransformer')
    def test_model_loading_failure(self, mock_st):
        """Test model loading failure handling."""
        mock_st.side_effect = Exception("Model loading failed")
        
        embedder = CodeEmbedder()
        
        with pytest.raises(Exception, match="Model loading failed"):
            _ = embedder.model

    def test_create_embedding_prompt_function(self):
        """Test embedding prompt creation for functions."""
        chunk = CodeChunk(
            content='def authenticate(user):\n    """Auth user."""\n    return True',
            chunk_type="function",
            start_line=5,
            end_line=7,
            file_path="/project/auth.py",
            relative_path="auth.py",
            folder_structure=[],
            name="authenticate",
            docstring="Auth user.",
            tags=["auth", "security"]
        )
        
        embedder = CodeEmbedder()
        prompt = embedder.create_embedding_prompt(chunk)
        
        assert prompt.startswith("task: code search | ")
        assert "function authenticate" in prompt
        assert "from auth.py" in prompt
        assert "(auth, security)" in prompt
        assert "Auth user." in prompt
        assert chunk.content in prompt

    def test_create_embedding_prompt_method(self):
        """Test embedding prompt creation for methods."""
        chunk = CodeChunk(
            content='    def save(self):\n        """Save user."""\n        pass',
            chunk_type="method",
            start_line=10,
            end_line=12,
            file_path="/project/src/models/user.py",
            relative_path="src/models/user.py", 
            folder_structure=['src', 'models'],
            name="save",
            parent_name="User",
            docstring="Save user.",
            tags=["database"]
        )
        
        embedder = CodeEmbedder()
        prompt = embedder.create_embedding_prompt(chunk)
        
        assert "method save in class User" in prompt
        assert "from src/models/user.py" in prompt
        assert "(database)" in prompt
        assert "Save user." in prompt

    def test_create_embedding_prompt_class(self):
        """Test embedding prompt creation for classes."""
        chunk = CodeChunk(
            content='class UserManager:\n    """Manages users."""',
            chunk_type="class",
            start_line=1,
            end_line=2,
            file_path="/project/managers.py",
            relative_path="managers.py",
            folder_structure=[],
            name="UserManager",
            docstring="Manages users.",
            tags=["auth", "management"]
        )
        
        embedder = CodeEmbedder()
        prompt = embedder.create_embedding_prompt(chunk)
        
        assert "class UserManager" in prompt
        assert "from managers.py" in prompt
        assert "(auth, management)" in prompt
        assert "Manages users." in prompt

    def test_create_embedding_prompt_long_docstring(self):
        """Test prompt creation with long docstring truncation."""
        long_docstring = "A" * 200  # 200 character docstring
        
        chunk = CodeChunk(
            content="def test(): pass",
            chunk_type="function",
            start_line=1,
            end_line=1,
            file_path="/test.py",
            relative_path="test.py",
            folder_structure=[],
            name="test",
            docstring=long_docstring
        )
        
        embedder = CodeEmbedder()
        prompt = embedder.create_embedding_prompt(chunk)
        
        # Should be truncated to 100 chars + "..."
        assert long_docstring[:100] + "..." in prompt
        assert len([part for part in prompt.split(" - ") if "A" * 200 in part]) == 0

    @patch('embeddings.embedder.SentenceTransformer')
    def test_embed_chunk(self, mock_st):
        """Test embedding generation for a single chunk."""
        # Setup mock
        mock_model = Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode_document.return_value = [mock_embedding]
        mock_st.return_value = mock_model
        
        chunk = CodeChunk(
            content="def test(): pass",
            chunk_type="function",
            start_line=1,
            end_line=1,
            file_path="/test.py",
            relative_path="test.py",
            folder_structure=[],
            name="test"
        )
        
        embedder = CodeEmbedder()
        result = embedder.embed_chunk(chunk)
        
        assert isinstance(result, EmbeddingResult)
        assert np.array_equal(result.embedding, mock_embedding)
        assert result.chunk_id == "test.py:1-1:function:test"
        assert result.metadata['file_path'] == "/test.py"
        assert result.metadata['name'] == "test"
        assert result.metadata['chunk_type'] == "function"

    @patch('embeddings.embedder.SentenceTransformer')
    def test_embed_chunks_batch(self, mock_st):
        """Test batch embedding generation."""
        # Setup mock
        mock_model = Mock()
        mock_embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        mock_model.encode_document.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        chunks = [
            CodeChunk(
                content="def test1(): pass",
                chunk_type="function",
                start_line=1,
                end_line=1,
                file_path="/test1.py",
                relative_path="test1.py",
                folder_structure=[],
                name="test1"
            ),
            CodeChunk(
                content="def test2(): pass",
                chunk_type="function", 
                start_line=3,
                end_line=3,
                file_path="/test2.py",
                relative_path="test2.py",
                folder_structure=[],
                name="test2"
            )
        ]
        
        embedder = CodeEmbedder()
        results = embedder.embed_chunks(chunks, batch_size=2)
        
        assert len(results) == 2
        assert isinstance(results[0], EmbeddingResult)
        assert isinstance(results[1], EmbeddingResult)
        
        # Check embeddings
        assert np.array_equal(results[0].embedding, mock_embeddings[0])
        assert np.array_equal(results[1].embedding, mock_embeddings[1])
        
        # Check chunk IDs
        assert results[0].chunk_id == "test1.py:1-1:function:test1"
        assert results[1].chunk_id == "test2.py:3-3:function:test2"

    @patch('embeddings.embedder.SentenceTransformer')
    def test_embed_chunks_with_batching(self, mock_st):
        """Test that large chunk lists are properly batched."""
        # Setup mock
        mock_model = Mock()
        # Simulate 5 chunks, batch size of 2 -> 3 batches
        mock_embeddings = [np.array([i]) for i in range(5)]
        mock_model.encode_document.side_effect = [
            mock_embeddings[:2],  # First batch
            mock_embeddings[2:4], # Second batch  
            mock_embeddings[4:]   # Third batch
        ]
        mock_st.return_value = mock_model
        
        # Create 5 chunks
        chunks = []
        for i in range(5):
            chunk = CodeChunk(
                content=f"def test{i}(): pass",
                chunk_type="function",
                start_line=i+1,
                end_line=i+1,
                file_path=f"/test{i}.py",
                relative_path=f"test{i}.py",
                folder_structure=[],
                name=f"test{i}"
            )
            chunks.append(chunk)
        
        embedder = CodeEmbedder()
        results = embedder.embed_chunks(chunks, batch_size=2)
        
        assert len(results) == 5
        assert mock_model.encode_document.call_count == 3  # 3 batches
        
        # Verify each result
        for i, result in enumerate(results):
            assert np.array_equal(result.embedding, mock_embeddings[i])
        assert result.chunk_id == f"test{i}.py:{i+1}-{i+1}:function:test{i}"

    @patch('embeddings.embedder.SentenceTransformer')
    def test_embed_query(self, mock_st):
        """Test query embedding generation."""
        mock_model = Mock()
        mock_embedding = np.array([0.5, 0.6, 0.7])
        mock_model.encode_query.return_value = mock_embedding
        mock_st.return_value = mock_model
        
        embedder = CodeEmbedder()
        result = embedder.embed_query("find authentication code")
        
        assert np.array_equal(result, mock_embedding)
        mock_model.encode_query.assert_called_once_with(
            "task: code search | query: find authentication code"
        )

    @patch('embeddings.embedder.SentenceTransformer')
    def test_get_model_info_loaded(self, mock_st):
        """Test model info when model is loaded."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.max_seq_length = 512
        mock_st.return_value = mock_model
        
        embedder = CodeEmbedder(model_name="test/model")
        # Trigger model loading
        _ = embedder.model
        
        info = embedder.get_model_info()
        
        assert info['model_name'] == "test/model"
        assert info['embedding_dimension'] == 768
        assert info['max_seq_length'] == 512
        assert info['device'] == "cpu"
        assert info['status'] == "loaded"

    def test_get_model_info_not_loaded(self):
        """Test model info when model is not loaded."""
        embedder = CodeEmbedder()
        info = embedder.get_model_info()
        
        assert info == {"status": "not_loaded"}

    def test_chunk_id_generation_variations(self):
        """Test chunk ID generation for different chunk types."""
        embedder = CodeEmbedder()
        
        # Function chunk
        func_chunk = CodeChunk(
            content="def test(): pass",
            chunk_type="function",
            start_line=1,
            end_line=3,
            file_path="/test.py",
            relative_path="test.py",
            folder_structure=[],
            name="test"
        )
        
        mock_model = Mock()
        mock_model.encode_document.return_value = [np.array([1, 2, 3])]
        embedder._model = mock_model
        
        result = embedder.embed_chunk(func_chunk)
        assert result.chunk_id == "test.py:1-3:function:test"
        
        # Method chunk
        method_chunk = CodeChunk(
            content="    def save(self): pass",
            chunk_type="method",
            start_line=5,
            end_line=5,
            file_path="/user.py",
            relative_path="user.py", 
            folder_structure=[],
            name="save",
            parent_name="User"
        )
        
        mock_model = Mock()
        embedder._model = mock_model
        mock_model.encode_document.return_value = [np.array([4, 5, 6])]
        result = embedder.embed_chunk(method_chunk)
        assert result.chunk_id == "user.py:5-5:method:save"
        
        # Unnamed chunk
        unnamed_chunk = CodeChunk(
            content="# Some comment",
            chunk_type="module_level",
            start_line=1,
            end_line=1,
            file_path="/misc.py",
            relative_path="misc.py",
            folder_structure=[],
            name=None
        )
        
        mock_model = Mock()
        embedder._model = mock_model
        mock_model.encode_document.return_value = [np.array([7, 8, 9])]
        result = embedder.embed_chunk(unnamed_chunk)
        assert result.chunk_id == "misc.py:1-1:module_level"

    def test_metadata_content_preview(self):
        """Test metadata content preview generation."""
        embedder = CodeEmbedder()
        
        # Short content
        short_chunk = CodeChunk(
            content="def test(): pass",
            chunk_type="function",
            start_line=1,
            end_line=1,
            file_path="/test.py",
            relative_path="test.py",
            folder_structure=[],
            name="test"
        )
        
        mock_model = Mock()
        embedder._model = mock_model
        mock_model.encode_document.return_value = [np.array([1, 2, 3])]
        result = embedder.embed_chunk(short_chunk)
        assert result.metadata['content_preview'] == "def test(): pass"
        
        # Long content (should be truncated)
        long_content = "def long_function():\n" + "    # comment\n" * 50
        long_chunk = CodeChunk(
            content=long_content,
            chunk_type="function",
            start_line=1,
            end_line=51,
            file_path="/long.py",
            relative_path="long.py",
            folder_structure=[],
            name="long_function"
        )
        
        mock_model = Mock()
        embedder._model = mock_model
        mock_model.encode_document.return_value = [np.array([4, 5, 6])]
        result = embedder.embed_chunk(long_chunk)
        preview = result.metadata['content_preview']
        assert len(preview) <= 203  # 200 + "..."
        assert preview.endswith("...")


class TestEmbeddingResult:
    """Test cases for EmbeddingResult dataclass."""
    
    def test_embedding_result_creation(self):
        """Test EmbeddingResult creation."""
        embedding = np.array([0.1, 0.2, 0.3])
        metadata = {'file_path': '/test.py', 'name': 'test'}
        
        result = EmbeddingResult(
            embedding=embedding,
            chunk_id="test_chunk",
            metadata=metadata
        )
        
        assert np.array_equal(result.embedding, embedding)
        assert result.chunk_id == "test_chunk"
        assert result.metadata == metadata

    def test_embedding_result_with_complex_metadata(self):
        """Test EmbeddingResult with complete metadata."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        
        metadata = {
            'file_path': '/project/src/auth/handlers.py',
            'relative_path': 'src/auth/handlers.py',
            'folder_structure': ['src', 'auth'],
            'chunk_type': 'function',
            'name': 'authenticate_user',
            'docstring': 'Authenticate a user with credentials.',
            'tags': ['auth', 'security'],
            'complexity_score': 5,
            'decorators': ['login_required'],
            'imports': ['hashlib', 'logging']
        }
        
        result = EmbeddingResult(
            embedding=embedding,
            chunk_id="auth_handlers:10-25:function:authenticate_user",
            metadata=metadata
        )
        
        assert result.chunk_id == "auth_handlers:10-25:function:authenticate_user"
        assert result.metadata['name'] == 'authenticate_user'
        assert result.metadata['tags'] == ['auth', 'security']
        assert result.metadata['complexity_score'] == 5
