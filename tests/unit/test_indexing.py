"""Unit tests for indexing and search functionality."""

import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from search.indexer import CodeIndexManager
from search.searcher import IntelligentSearcher, SearchResult
from embeddings.embedder import EmbeddingResult


class TestCodeIndexManager:
    """Test cases for CodeIndexManager."""
    
    def test_init(self, mock_storage_dir):
        """Test index manager initialization."""
        index_manager = CodeIndexManager(str(mock_storage_dir))
        
        assert index_manager.storage_dir == mock_storage_dir
        assert index_manager.index_path == mock_storage_dir / "code.index"
        assert index_manager.metadata_path == mock_storage_dir / "metadata.db"
        assert index_manager._index is None  # Lazy loading

    @patch('search.indexer.faiss')
    def test_create_flat_index(self, mock_faiss, mock_storage_dir):
        """Test creation of flat FAISS index."""
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        index_manager = CodeIndexManager(str(mock_storage_dir))
        index_manager.create_index(768, "flat")
        
        mock_faiss.IndexFlatIP.assert_called_once_with(768)
        assert index_manager._index == mock_index

    @patch('search.indexer.faiss')
    def test_create_ivf_index(self, mock_faiss, mock_storage_dir):
        """Test creation of IVF FAISS index."""
        mock_quantizer = Mock()
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_quantizer
        mock_faiss.IndexIVFFlat.return_value = mock_index
        
        index_manager = CodeIndexManager(str(mock_storage_dir))
        index_manager.create_index(768, "ivf")
        
        mock_faiss.IndexFlatIP.assert_called_once_with(768)
        mock_faiss.IndexIVFFlat.assert_called_once_with(mock_quantizer, 768, 96)  # 768 // 8 = 96
        assert index_manager._index == mock_index

    def test_create_invalid_index_type(self, mock_storage_dir):
        """Test error handling for invalid index type."""
        index_manager = CodeIndexManager(str(mock_storage_dir))
        
        with pytest.raises(ValueError, match="Unsupported index type"):
            index_manager.create_index(768, "invalid_type")

    @patch('search.indexer.faiss')
    @patch('search.indexer.SqliteDict')
    def test_add_embeddings_new_index(self, mock_sqlite, mock_faiss, mock_storage_dir):
        """Test adding embeddings to a new index."""
        # Setup mocks
        mock_index = Mock()
        mock_index.ntotal = 0
        mock_index.is_trained = True
        mock_index.d = 3  # Real integer instead of mock
        mock_index.__class__.__name__ = 'IndexFlatIP'
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        # Create a mock database that supports item assignment and close method
        class MockDB(dict):
            def close(self):
                pass
            def get(self, key, default=None):
                return super().get(key, default)
        
        mock_db = MockDB()
        mock_sqlite.return_value = mock_db
        
        # Create embedding results
        embeddings = [
            EmbeddingResult(
                embedding=np.array([0.1, 0.2, 0.3]),
                chunk_id="test1.py:1-5:function:test1",
                metadata={'name': 'test1', 'file_path': '/test1.py'}
            ),
            EmbeddingResult(
                embedding=np.array([0.4, 0.5, 0.6]),
                chunk_id="test2.py:10-15:function:test2", 
                metadata={'name': 'test2', 'file_path': '/test2.py'}
            )
        ]
        
        index_manager = CodeIndexManager(str(mock_storage_dir))
        index_manager.add_embeddings(embeddings)
        
        # Verify index creation and training
        mock_faiss.IndexFlatIP.assert_called_once_with(3)
        mock_faiss.normalize_L2.assert_called_once()
        mock_index.add.assert_called_once()
        
        # Verify metadata storage
        assert len(mock_db) == 2
        
        # Verify chunk IDs were stored
        assert len(index_manager._chunk_ids) == 2
        assert "test1.py:1-5:function:test1" in index_manager._chunk_ids
        assert "test2.py:10-15:function:test2" in index_manager._chunk_ids

    @patch('search.indexer.faiss')
    @patch('search.indexer.SqliteDict')
    def test_add_embeddings_ivf_training(self, mock_sqlite, mock_faiss, mock_storage_dir):
        """Test IVF index training during embedding addition."""
        # Setup mocks for IVF index
        mock_quantizer = Mock()
        mock_index = Mock()
        mock_index.ntotal = 0
        mock_index.is_trained = False  # Needs training
        mock_index.d = 3  # Real integer instead of mock
        mock_index.__class__.__name__ = 'IndexIVFFlat'
        mock_faiss.IndexFlatIP.return_value = mock_quantizer
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        # Create a mock database that supports item assignment and close method
        class MockDB(dict):
            def close(self):
                pass
            def get(self, key, default=None):
                return super().get(key, default)
        
        mock_db = MockDB()
        mock_sqlite.return_value = mock_db
        
        # Create large dataset to trigger IVF
        embeddings = []
        for i in range(1500):  # > 1000 triggers IVF
            embeddings.append(EmbeddingResult(
                embedding=np.array([0.1 * i, 0.2 * i, 0.3 * i]),
                chunk_id=f"test{i}.py:1-5:function:test{i}",
                metadata={'name': f'test{i}'}
            ))
        
        index_manager = CodeIndexManager(str(mock_storage_dir))
        index_manager.add_embeddings(embeddings)
        
        # Verify IVF index was created and trained
        mock_faiss.IndexIVFFlat.assert_called_once()
        mock_index.train.assert_called_once()
        mock_index.add.assert_called_once()

    @patch('search.indexer.faiss')
    @patch('search.indexer.SqliteDict')
    def test_search_basic(self, mock_sqlite, mock_faiss, mock_storage_dir):
        """Test basic search functionality."""
        # Setup mocks
        mock_index = Mock()
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            np.array([[0.9, 0.7]]),  # similarities
            np.array([[0, 1]])       # indices
        )
        mock_faiss.normalize_L2 = Mock()
        
        # Create a mock database with predefined data
        mock_db = {
            'chunk1.py:1-5:function:chunk1': {
                'index_id': 0,
                'metadata': {
                    'name': 'chunk1',
                    'file_path': '/test1.py'
                }
            },
            'chunk2.py:10-15:function:chunk2': {
                'index_id': 1,
                'metadata': {
                    'name': 'chunk2',
                    'file_path': '/test2.py'
                }
            }
        }
        mock_db_obj = Mock()
        mock_db_obj.get.side_effect = lambda chunk_id: mock_db.get(chunk_id)
        mock_sqlite.return_value = mock_db_obj
        
        index_manager = CodeIndexManager(str(mock_storage_dir))
        index_manager._index = mock_index
        index_manager._chunk_ids = ['chunk1.py:1-5:function:chunk1', 'chunk2.py:10-15:function:chunk2']
        
        query_embedding = np.array([0.1, 0.2, 0.3])
        results = index_manager.search(query_embedding, k=2)
        
        assert len(results) == 2
        assert results[0][1] == 0.9  # First result similarity
        assert results[1][1] == 0.7  # Second result similarity
        
        mock_faiss.normalize_L2.assert_called_once()
        mock_index.search.assert_called_once()

    @patch('search.indexer.faiss')
    @patch('search.indexer.SqliteDict')
    def test_search_with_filters(self, mock_sqlite, mock_faiss, mock_storage_dir):
        """Test search with metadata filters."""
        # Setup mocks
        mock_index = Mock()
        mock_index.ntotal = 3
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),
            np.array([[0, 1, 2]])
        )
        mock_faiss.normalize_L2 = Mock()
        
        # Setup metadata with different chunk types
        mock_db = {
            'func1.py:1-5:function:func1': {
                'index_id': 0,
                'metadata': {'name': 'func1', 'chunk_type': 'function', 'tags': ['auth']}
            },
            'class1.py:10-15:class:class1': {
                'index_id': 1,
                'metadata': {'name': 'class1', 'chunk_type': 'class', 'tags': ['database']}
            },
            'func2.py:20-25:function:func2': {
                'index_id': 2,
                'metadata': {'name': 'func2', 'chunk_type': 'function', 'tags': ['api']}
            }
        }
        
        mock_db_obj = Mock()
        mock_db_obj.get.side_effect = lambda chunk_id: mock_db.get(chunk_id)
        mock_sqlite.return_value = mock_db_obj
        
        index_manager = CodeIndexManager(str(mock_storage_dir))
        index_manager._index = mock_index
        index_manager._chunk_ids = ['func1.py:1-5:function:func1', 'class1.py:10-15:class:class1', 'func2.py:20-25:function:func2']
        
        query_embedding = np.array([0.1, 0.2, 0.3])
        
        # Filter for functions only
        filters = {'chunk_type': 'function'}
        results = index_manager.search(query_embedding, k=5, filters=filters)
        
        # Should only return function chunks
        assert len(results) == 2  # func1 and func2
        for chunk_id, similarity, metadata in results:
            assert metadata['chunk_type'] == 'function'

    def test_matches_filters(self, mock_storage_dir):
        """Test filter matching logic."""
        index_manager = CodeIndexManager(str(mock_storage_dir))
        
        metadata = {
            'chunk_type': 'function',
            'tags': ['auth', 'security'],
            'relative_path': 'src/auth/handlers.py',
            'folder_structure': ['src', 'auth'],
            'name': 'authenticate_user'
        }
        
        # Test chunk type filter
        assert index_manager._matches_filters(metadata, {'chunk_type': 'function'})
        assert not index_manager._matches_filters(metadata, {'chunk_type': 'class'})
        
        # Test tags filter
        assert index_manager._matches_filters(metadata, {'tags': ['auth']})
        assert index_manager._matches_filters(metadata, {'tags': ['security']})
        assert not index_manager._matches_filters(metadata, {'tags': ['database']})
        
        # Test file pattern filter
        assert index_manager._matches_filters(metadata, {'file_pattern': ['auth']})
        assert index_manager._matches_filters(metadata, {'file_pattern': ['handlers']})
        assert not index_manager._matches_filters(metadata, {'file_pattern': ['database']})
        
        # Test folder structure filter
        assert index_manager._matches_filters(metadata, {'folder_structure': ['auth']})
        assert index_manager._matches_filters(metadata, {'folder_structure': ['src']})
        assert not index_manager._matches_filters(metadata, {'folder_structure': ['api']})

    @patch('search.indexer.faiss')
    @patch('search.indexer.SqliteDict')
    def test_get_chunk_by_id(self, mock_sqlite, mock_faiss, mock_storage_dir):
        """Test retrieving chunk by ID."""
        mock_db_obj = Mock()
        mock_db_obj.get.return_value = {
            'index_id': 0,
            'metadata': {'name': 'test_func', 'file_path': '/test.py'}
        }
        mock_sqlite.return_value = mock_db_obj
        
        index_manager = CodeIndexManager(str(mock_storage_dir))
        
        result = index_manager.get_chunk_by_id('test.py:1-5:function:test_func')
        
        assert result['name'] == 'test_func'
        assert result['file_path'] == '/test.py'
        mock_db_obj.get.assert_called_once_with('test.py:1-5:function:test_func')

    @patch('search.indexer.faiss')
    @patch('search.indexer.SqliteDict')
    def test_save_index(self, mock_sqlite, mock_faiss, mock_storage_dir):
        """Test saving index to disk."""
        mock_index = Mock()
        mock_index.ntotal = 2
        mock_index.d = 768
        mock_index.__class__.__name__ = 'IndexFlatIP'
        mock_faiss.write_index = Mock()
        
        # Mock the database
        class MockDB:
            def get(self, key, default=None):
                return {
                    'metadata': {
                        'relative_path': 'test.py',
                        'folder_structure': ['src'],
                        'chunk_type': 'function',
                        'tags': ['test']
                    }
                }
            def close(self):
                pass
        
        mock_db_obj = MockDB()
        mock_sqlite.return_value = mock_db_obj
        
        index_manager = CodeIndexManager(str(mock_storage_dir))
        index_manager._index = mock_index
        index_manager._chunk_ids = ['test1', 'test2']
        
        index_manager.save_index()
        
        mock_faiss.write_index.assert_called_once_with(
            mock_index, str(mock_storage_dir / "code.index")
        )
        
        # Check chunk IDs were saved
        chunk_id_path = mock_storage_dir / "chunk_ids.pkl"
        assert chunk_id_path.exists()
        
        # Check stats were saved
        stats_path = mock_storage_dir / "stats.json"
        assert stats_path.exists()

    def test_update_stats(self, mock_storage_dir):
        """Test statistics update."""
        with patch('search.indexer.SqliteDict') as mock_sqlite:
            class MockDB:
                def get(self, chunk_id):
                    return {
                        'metadata': {
                            'relative_path': 'test.py',
                            'folder_structure': ['src', 'auth'],
                            'chunk_type': 'function',
                            'tags': ['auth', 'security']
                        }
                    }
                def close(self):
                    pass
            
            mock_db_obj = MockDB()
            mock_sqlite.return_value = mock_db_obj
            
            index_manager = CodeIndexManager(str(mock_storage_dir))
            index_manager._chunk_ids = ['test1', 'test2']
            mock_index = Mock()
            mock_index.ntotal = 2
            mock_index.d = 768
            mock_index.__class__.__name__ = 'IndexFlatIP'
            index_manager._index = mock_index
            
            index_manager._update_stats()
            
            stats_path = mock_storage_dir / "stats.json"
            assert stats_path.exists()
            
            with open(stats_path) as f:
                stats = json.load(f)
            
            assert stats['total_chunks'] == 2
            assert stats['embedding_dimension'] == 768
            assert 'chunk_types' in stats
            assert 'top_tags' in stats

    @patch('search.indexer.SqliteDict')
    def test_clear_index(self, mock_sqlite, mock_storage_dir):
        """Test clearing the entire index."""
        mock_db_obj = Mock()
        mock_sqlite.return_value = mock_db_obj
        
        # Create some files to be cleared
        (mock_storage_dir / "code.index").touch()
        (mock_storage_dir / "metadata.db").touch()
        (mock_storage_dir / "chunk_ids.pkl").touch()
        (mock_storage_dir / "stats.json").touch()
        
        index_manager = CodeIndexManager(str(mock_storage_dir))
        index_manager._index = Mock()
        index_manager._chunk_ids = ['test1', 'test2']
        # Initialize the database so it gets closed
        index_manager._metadata_db = mock_db_obj
        
        index_manager.clear_index()
        
        # Check files were removed
        assert not (mock_storage_dir / "code.index").exists()
        assert not (mock_storage_dir / "metadata.db").exists() 
        assert not (mock_storage_dir / "chunk_ids.pkl").exists()
        assert not (mock_storage_dir / "stats.json").exists()
        
        # Check state was reset
        assert index_manager._index is None
        assert index_manager._chunk_ids == []
        mock_db_obj.close.assert_called_once()


class TestIntelligentSearcher:
    """Test cases for IntelligentSearcher."""
    
    def test_init(self, mock_storage_dir):
        """Test searcher initialization."""
        mock_index_manager = Mock()
        mock_embedder = Mock()
        
        searcher = IntelligentSearcher(mock_index_manager, mock_embedder)
        
        assert searcher.index_manager == mock_index_manager
        assert searcher.embedder == mock_embedder
        assert hasattr(searcher, 'query_patterns')

    def test_detect_query_intent(self, mock_storage_dir):
        """Test query intent detection."""
        mock_index_manager = Mock()
        mock_embedder = Mock()
        
        searcher = IntelligentSearcher(mock_index_manager, mock_embedder)
        
        # Test authentication intent
        auth_intents = searcher._detect_query_intent("find user auth function definition")
        assert 'authentication' in auth_intents
        assert 'function_search' in auth_intents
        
        # Test database intent
        db_intents = searcher._detect_query_intent("show database connection code")
        assert 'database' in db_intents
        
        # Test API intent
        api_intents = searcher._detect_query_intent("find REST API endpoints")
        assert 'api' in api_intents
        
        # Test error handling intent
        error_intents = searcher._detect_query_intent("exception handling patterns")
        assert 'error_handling' in error_intents

    def test_optimize_query(self, mock_storage_dir):
        """Test query optimization."""
        mock_index_manager = Mock()
        mock_embedder = Mock()
        
        searcher = IntelligentSearcher(mock_index_manager, mock_embedder)
        
        # Test abbreviation expansion
        optimized = searcher._optimize_query("find auth db code")
        assert "authentication" in optimized
        assert "database" in optimized
        
        # Test case preservation
        optimized = searcher._optimize_query("API auth functions")
        assert "authentication" in optimized
        assert "application programming interface" in optimized

    def test_search_integration(self, mock_storage_dir):
        """Test full search integration."""
        # Setup mocks
        mock_index_manager = Mock()
        mock_index_manager.search.return_value = [
            ('chunk1.py:1-5:function:auth_func', 0.9, {
                'name': 'auth_func',
                'chunk_type': 'function',
                'content_preview': 'def auth_func(): pass',
                'file_path': '/test.py',
                'relative_path': 'test.py',
                'folder_structure': [],
                'start_line': 1,
                'end_line': 5,
                'tags': ['auth'],
                'docstring': 'Auth function',
                'complexity_score': 2
            })
        ]
        mock_index_manager.get_similar_chunks.return_value = []
        
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = np.array([0.1, 0.2, 0.3])
        
        searcher = IntelligentSearcher(mock_index_manager, mock_embedder)
        
        results = searcher.search("find authentication functions", k=1)
        
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.name == 'auth_func'
        assert result.chunk_type == 'function'
        assert result.similarity_score == 0.9
        assert 'auth' in result.tags

    def test_search_by_file_pattern(self, mock_storage_dir):
        """Test search with file pattern filter."""
        mock_index_manager = Mock()
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = np.array([0.1, 0.2])
        
        searcher = IntelligentSearcher(mock_index_manager, mock_embedder)
        searcher.search = Mock()  # Mock the main search method
        
        searcher.search_by_file_pattern("auth code", ["auth", "security"])
        
        searcher.search.assert_called_once_with(
            "auth code", k=5, filters={'file_pattern': ["auth", "security"]}
        )

    def test_search_by_chunk_type(self, mock_storage_dir):
        """Test search with chunk type filter."""
        mock_index_manager = Mock()
        mock_embedder = Mock()
        mock_embedder.embed_query.return_value = np.array([0.1, 0.2])
        
        searcher = IntelligentSearcher(mock_index_manager, mock_embedder)
        searcher.search = Mock()
        
        searcher.search_by_chunk_type("user management", "class")
        
        searcher.search.assert_called_once_with(
            "user management", k=5, filters={'chunk_type': 'class'}
        )

    def test_rank_results(self, mock_storage_dir):
        """Test result ranking logic."""
        mock_index_manager = Mock()
        mock_embedder = Mock()
        
        searcher = IntelligentSearcher(mock_index_manager, mock_embedder)
        
        # Create test results with different characteristics
        results = [
            SearchResult(
                chunk_id='chunk1',
                similarity_score=0.8,
                content_preview='def test(): pass',
                file_path='/test1.py',
                relative_path='test1.py',
                folder_structure=[],
                chunk_type='function',
                name='test',
                parent_name=None,
                start_line=1,
                end_line=1,
                docstring='Test function',  # Has docstring
                tags=['auth'],  # Matches intent
                context_info={}
            ),
            SearchResult(
                chunk_id='chunk2',
                similarity_score=0.9,  # Higher base similarity
                content_preview='class Manager: pass',
                file_path='/test2.py',
                relative_path='test2.py', 
                folder_structure=[],
                chunk_type='class',
                name='Manager',
                parent_name=None,
                start_line=1,
                end_line=1,
                docstring=None,  # No docstring
                tags=[],  # No matching tags
                context_info={}
            )
        ]
        
        # Rank with auth intent - should boost chunk1 despite lower base similarity
        ranked = searcher._rank_results(results, "find auth functions", ['auth'])
        
        # chunk1 should be ranked higher due to tag match and docstring
        assert ranked[0].chunk_id == 'chunk1'
        assert ranked[1].chunk_id == 'chunk2'

    def test_get_search_suggestions(self, mock_storage_dir):
        """Test search suggestion generation."""
        mock_index_manager = Mock()
        mock_index_manager.get_stats.return_value = {
            'top_tags': {'auth': 10, 'database': 8, 'api': 5},
            'chunk_types': {'function': 20, 'class': 10, 'method': 15}
        }
        mock_embedder = Mock()
        
        searcher = IntelligentSearcher(mock_index_manager, mock_embedder)
        
        # Test tag-based suggestions
        suggestions = searcher.get_search_suggestions("auth")
        assert any("auth" in suggestion for suggestion in suggestions)
        
        # Test chunk type suggestions
        suggestions = searcher.get_search_suggestions("function") 
        assert any("function" in suggestion for suggestion in suggestions)


class TestSearchResult:
    """Test cases for SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            chunk_id='test.py:1-5:function:test',
            similarity_score=0.95,
            content_preview='def test(): pass',
            file_path='/project/test.py',
            relative_path='test.py',
            folder_structure=[],
            chunk_type='function',
            name='test',
            parent_name=None,
            start_line=1,
            end_line=5,
            docstring='Test function',
            tags=['testing'],
            context_info={'similar_chunks': []}
        )
        
        assert result.chunk_id == 'test.py:1-5:function:test'
        assert result.similarity_score == 0.95
        assert result.name == 'test'
        assert result.chunk_type == 'function'
        assert result.tags == ['testing']

    def test_search_result_with_context(self):
        """Test SearchResult with context information."""
        context_info = {
            'similar_chunks': [
                {
                    'chunk_id': 'other.py:10-15:function:similar',
                    'similarity': 0.8,
                    'name': 'similar',
                    'chunk_type': 'function'
                }
            ],
            'file_context': {
                'total_chunks_in_file': 5,
                'folder_path': 'src/utils'
            }
        }
        
        result = SearchResult(
            chunk_id='utils.py:1-10:function:helper',
            similarity_score=0.85,
            content_preview='def helper(): return True',
            file_path='/project/src/utils/helpers.py',
            relative_path='src/utils/helpers.py',
            folder_structure=['src', 'utils'],
            chunk_type='function',
            name='helper',
            parent_name=None,
            start_line=1,
            end_line=10,
            docstring='Helper function',
            tags=['utility'],
            context_info=context_info
        )
        
        assert len(result.context_info['similar_chunks']) == 1
        assert result.context_info['file_context']['total_chunks_in_file'] == 5
        assert result.folder_structure == ['src', 'utils']
