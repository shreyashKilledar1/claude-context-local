#!/usr/bin/env python3
"""Download the EmbeddingGemma model for testing and development."""

import os
import sys
import logging
from pathlib import Path

from embeddings.embedder import CodeEmbedder


def download_model(model_name: str = "google/embeddinggemma-300m", storage_dir: str = None):
    """Download the embedding model."""
    if storage_dir is None:
        storage_dir = os.path.expanduser("~/.claude_code_search")
    
    print(f"Downloading model: {model_name}")
    print(f"Storage directory: {storage_dir}")
    
    try:
        # Create embedder instance which will download the model
        embedder = CodeEmbedder(
            model_name=model_name,
            storage_dir=storage_dir,
            batch_size=4  # Small batch for testing
        )
        
        print("Loading model...")
        embedder.load_model()
        
        # Test the model works
        print("Testing model...")
        test_text = "def hello_world():\n    return 'Hello, World!'"
        embedding = embedder.model.encode([test_text])
        
        print(f"Model loaded successfully!")
        print(f"Embedding dimension: {embedding.shape[1]}")
        print(f"Model info: {embedder.get_model_info()}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download embedding model for testing")
    parser.add_argument(
        "--model", 
        default="google/embeddinggemma-300m",
        help="Model name to download"
    )
    parser.add_argument(
        "--storage-dir",
        help="Storage directory (default: ~/.claude_code_search)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    success = download_model(args.model, args.storage_dir)
    sys.exit(0 if success else 1)
