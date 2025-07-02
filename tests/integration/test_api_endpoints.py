#!/usr/bin/env python3
"""
Integration tests for the Flask API.
Tests the complete system as users would interact with it.
"""

import unittest
import requests
import json
import time
import threading
import subprocess
import sys
from pathlib import Path

class TestAPIEndpoints(unittest.TestCase):
    """Test Flask API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Start Flask server for testing."""
        cls.base_url = "http://localhost:5000"
        print("Note: Make sure Flask app is running at localhost:5000")
        print("Run: python app.py in another terminal")
        
        # Wait for server to be ready
        time.sleep(2)
        
        # Verify server is responding
        try:
            response = requests.get(f"{cls.base_url}/health", timeout=5)
            if response.status_code != 200:
                raise Exception("Server not responding")
            print("✅ Flask server detected and ready")
        except Exception as e:
            print(f"❌ Flask server not available: {e}")
            print("Please start the Flask app: python app.py")
            raise
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.base_url}/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        print("✅ Health endpoint test passed")
    
    def test_query_endpoint_basic(self):
        """Test basic query functionality."""
        payload = {
            "query": "What is machine learning?",
            "top_k": 3
        }
        
        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn('answer', data)
        self.assertIn('response_time_ms', data)
        self.assertGreater(len(data['answer']), 10)
        self.assertLess(data['response_time_ms'], 5000)
        print(f"✅ Basic query test passed - {data['response_time_ms']}ms")
    
    def test_query_with_sources(self):
        """Test query with source documents returned."""
        payload = {
            "query": "Explain neural networks",
            "top_k": 5,
            "return_sources": True
        }
        
        response = requests.post(f"{self.base_url}/query", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn('sources', data)
        self.assertGreater(len(data['sources']), 0)
        self.assertLessEqual(len(data['sources']), 5)
        print(f"✅ Sources test passed - {len(data['sources'])} sources returned")
    
    def test_error_handling(self):
        """Test API error handling."""
        # Test missing query
        response = requests.post(f"{self.base_url}/query", json={})
        self.assertEqual(response.status_code, 400)
        
        # Test invalid top_k
        payload = {"query": "test", "top_k": -1}
        response = requests.post(f"{self.base_url}/query", json=payload)
        self.assertEqual(response.status_code, 400)
        
        print("✅ Error handling tests passed")

if __name__ == '__main__':
    unittest.main(verbosity=2)