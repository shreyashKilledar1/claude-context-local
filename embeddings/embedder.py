"""EmbeddingGemma wrapper for generating code embeddings."""

import os
from pathlib import Path
import logging
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import torch
except Exception:
    torch = None

from chunking.python_ast_chunker import CodeChunk


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embedding: np.ndarray
    chunk_id: str
    metadata: Dict[str, Any]


class CodeEmbedder:
    """Wrapper for EmbeddingGemma model to generate code embeddings."""
    
    def __init__(
        self, 
        model_name: str = "google/embeddinggemma-300m",
        cache_dir: Optional[str] = None,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self._model = None
        self._logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    @property
    def model(self):
        """Lazy loading of the model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the EmbeddingGemma model."""
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not found. Install with: "
                "pip install sentence-transformers>=5.0.0"
            )
        
        self._logger.info(f"Loading model: {self.model_name}")

        # If the model appears to be cached locally, enable offline mode to avoid network HEAD/GET checks
        local_model_dir = None
        try:
            if self._is_model_cached():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                self._logger.info("Model cache detected. Enabling offline mode for faster startup.")
                # Prefer loading directly from local cache path to avoid any remote HEAD/GET
                local_model_dir = self._find_local_model_dir()
                if local_model_dir:
                    self._logger.info(f"Loading model from local cache path: {local_model_dir}")
        except Exception as _e:
            # Best-effort; continue without failing if detection has issues
            self._logger.debug(f"Offline mode detection skipped: {_e}")
        
        try:
            model_source = str(local_model_dir) if local_model_dir else self.model_name
            resolved_device = self._resolve_device(self.device)
            self._model = SentenceTransformer(
                model_source,
                cache_folder=self.cache_dir,
                device=resolved_device
            )
            # Persist resolved device for later info
            self.device = resolved_device
            self._logger.info(f"Model loaded successfully on device: {self._model.device}")
            
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            raise
    
    def create_embedding_prompt(self, chunk: CodeChunk) -> str:
        """Create an optimized prompt for embedding generation."""
        # Use task-specific prompts as recommended by EmbeddingGemma
        base_prompt = "task: code search | "
        
        # Add context based on chunk type
        if chunk.chunk_type == 'function':
            context = f"function {chunk.name}"
            if chunk.parent_name:
                context = f"method {chunk.name} in class {chunk.parent_name}"
        elif chunk.chunk_type == 'class':
            context = f"class {chunk.name}"
        elif chunk.chunk_type == 'method':
            context = f"method {chunk.name} in class {chunk.parent_name}"
        else:
            context = f"{chunk.chunk_type} code"
        
        # Add file context
        file_context = f"from {chunk.relative_path}"
        
        # Add semantic tags if available
        tags_context = ""
        if chunk.tags:
            tags_context = f" ({', '.join(chunk.tags)})"
        
        # Combine docstring if available
        docstring_context = ""
        if chunk.docstring:
            # Truncate docstring to avoid token limit
            docstring_preview = chunk.docstring[:100] + "..." if len(chunk.docstring) > 100 else chunk.docstring
            docstring_context = f" - {docstring_preview}"
        
        prompt = f"{base_prompt}{context} {file_context}{tags_context}{docstring_context}\n\n{chunk.content}"
        
        return prompt
    
    def embed_chunk(self, chunk: CodeChunk) -> EmbeddingResult:
        """Generate embedding for a single code chunk."""
        prompt = self.create_embedding_prompt(chunk)
        
        # Use encode_document for code content
        # Disable progress bars for small, single-item encode
        embedding = self.model.encode_document([prompt], show_progress_bar=False)[0]
        
        # Create unique chunk ID
        chunk_id = f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}:{chunk.chunk_type}"
        if chunk.name:
            chunk_id += f":{chunk.name}"
        
        # Prepare metadata
        metadata = {
            'file_path': chunk.file_path,
            'relative_path': chunk.relative_path,
            'folder_structure': chunk.folder_structure,
            'chunk_type': chunk.chunk_type,
            'start_line': chunk.start_line,
            'end_line': chunk.end_line,
            'name': chunk.name,
            'parent_name': chunk.parent_name,
            'docstring': chunk.docstring,
            'decorators': chunk.decorators,
            'imports': chunk.imports,
            'complexity_score': chunk.complexity_score,
            'tags': chunk.tags,
            'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        }
        
        return EmbeddingResult(
            embedding=embedding,
            chunk_id=chunk_id,
            metadata=metadata
        )
    
    def embed_chunks(self, chunks: List[CodeChunk], batch_size: int = 32) -> List[EmbeddingResult]:
        """Generate embeddings for multiple chunks with batching."""
        results = []
        
        self._logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_prompts = [self.create_embedding_prompt(chunk) for chunk in batch]
            
            # Generate embeddings for batch
            batch_embeddings = self.model.encode_document(batch_prompts, show_progress_bar=False)
            
            # Create results
            for j, (chunk, embedding) in enumerate(zip(batch, batch_embeddings)):
                chunk_id = f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}:{chunk.chunk_type}"
                if chunk.name:
                    chunk_id += f":{chunk.name}"
                
                metadata = {
                    'file_path': chunk.file_path,
                    'relative_path': chunk.relative_path,
                    'folder_structure': chunk.folder_structure,
                    'chunk_type': chunk.chunk_type,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'name': chunk.name,
                    'parent_name': chunk.parent_name,
                    'docstring': chunk.docstring,
                    'decorators': chunk.decorators,
                    'imports': chunk.imports,
                    'complexity_score': chunk.complexity_score,
                    'tags': chunk.tags,
                    'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                }
                
                results.append(EmbeddingResult(
                    embedding=embedding,
                    chunk_id=chunk_id,
                    metadata=metadata
                ))
            
            if i + batch_size < len(chunks):
                self._logger.info(f"Processed {i + batch_size}/{len(chunks)} chunks")
        
        self._logger.info("Embedding generation completed")
        return results
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query."""
        # Use query-specific prompt
        query_prompt = f"task: code search | query: {query}"
        
        # Use encode_query for search queries
        embedding = self.model.encode_query(query_prompt)
        return embedding
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self._model.get_sentence_embedding_dimension(),
            "max_seq_length": getattr(self._model, 'max_seq_length', 'unknown'),
            "device": str(self._model.device),
            "status": "loaded"
        }

    def _is_model_cached(self) -> bool:
        """Best-effort check if the target model seems cached in cache_dir.

        We look for a directory under cache_dir that contains the model key
        (final path segment of model name) and a SentenceTransformers config file.
        """
        if not self.cache_dir:
            return False
        try:
            model_key = self.model_name.split('/')[-1].lower()
            cache_root = Path(self.cache_dir)
            if not cache_root.exists():
                return False
            for path in cache_root.rglob('config_sentence_transformers.json'):
                parent_str = str(path.parent).lower()
                if model_key in parent_str:
                    return True
            # Fallback: look for folders that include the model key
            for d in cache_root.glob('**/*'):
                if d.is_dir() and model_key in d.name.lower():
                    if (d / 'config_sentence_transformers.json').exists() or (d / 'README.md').exists():
                        return True
        except Exception:
            return False
        return False

    def _find_local_model_dir(self) -> Optional[Path]:
        """Locate the cached model directory if available."""
        if not self.cache_dir:
            return None
        try:
            model_key = self.model_name.split('/')[-1].lower()
            cache_root = Path(self.cache_dir)
            if not cache_root.exists():
                return None
            for path in cache_root.rglob('config_sentence_transformers.json'):
                parent = path.parent
                if model_key in str(parent).lower():
                    return parent
            # Fallback: return the most likely directory matching the model key
            candidates = [d for d in cache_root.glob('**/*') if d.is_dir() and model_key in d.name.lower()]
            return candidates[0] if candidates else None
        except Exception:
            return None

    def _resolve_device(self, requested: Optional[str]) -> str:
        """Resolve target device string.
        - "auto": prefer cuda, then mps, else cpu
        - explicit values are validated and coerced to available devices
        """
        req = (requested or "auto").lower()
        # If torch is not available, default to CPU
        if torch is None:
            return "cpu"
        if req in ("auto", "none", ""):
            if torch.cuda.is_available():
                return "cuda"
            # MPS for Apple Silicon
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        # Validate explicit devices
        if req.startswith("cuda"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "mps":
            try:
                return "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
            except Exception:
                return "cpu"
        # Default fallback
        return "cpu"
