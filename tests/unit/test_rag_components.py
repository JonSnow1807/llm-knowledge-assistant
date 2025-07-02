#!/usr/bin/env python3
"""
Unit tests for individual RAG pipeline components.
This shows you understand software engineering best practices.
"""

import unittest
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag_pipeline import RAGPipeline
import torch
import numpy as np

class TestRAGComponents(unittest.TestCase):
    """Test individual components of the RAG pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = "models/fine_tuned/final_model"
        print("Loading model for unit tests...")
        cls.rag = RAGPipeline(cls.model_path)
        print("✅ Model loaded for testing")
    
    def test_embedding_model_loaded(self):
        """Test that embedding model loads correctly."""
        self.assertIsNotNone(self.rag.embedding_model)
        
        # Test embedding generation
        test_text = "This is a test sentence."
        embedding = self.rag.embedding_model.encode([test_text])
        
        self.assertEqual(len(embedding), 1)
        self.assertGreater(len(embedding[0]), 300)  # Should be 384-dimensional
        print("✅ Embedding model test passed")
    
    def test_faiss_index_functionality(self):
        """Test FAISS index search functionality."""
        self.assertIsNotNone(self.rag.faiss_index)
        self.assertGreater(self.rag.faiss_index.ntotal, 1000)  # Should have many vectors
        
        # Test retrieval
        docs = self.rag.retrieve_documents("machine learning", top_k=3)
        
        self.assertEqual(len(docs), 3)
        self.assertIn('text', docs[0])
        self.assertIn('score', docs[0])
        print(f"✅ FAISS index test passed - {self.rag.faiss_index.ntotal} vectors")
    
    def test_model_generation(self):
        """Test that model can generate responses."""
        test_query = "What is AI?"
        test_docs = [{"text": "AI is artificial intelligence."}]
        
        response = self.rag.generate_response(test_query, test_docs, max_new_tokens=20)
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 5)  # Should generate meaningful text
        print(f"✅ Generation test passed - Response: {response[:50]}...")
    
    def test_end_to_end_query(self):
        """Test complete query processing."""
        result = self.rag.query("What is machine learning?")
        
        self.assertIn('answer', result)
        self.assertIn('response_time_ms', result)
        self.assertGreater(len(result['answer']), 20)
        self.assertLess(result['response_time_ms'], 5000)  # Should be under 5s
        print(f"✅ End-to-end test passed - {result['response_time_ms']}ms")

if __name__ == '__main__':
    unittest.main(verbosity=2)