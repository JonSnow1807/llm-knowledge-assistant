"""
Flask API for serving the LLM-powered Knowledge Assistant.
Provides REST endpoints for RAG queries with optimized performance.
"""

import os
import time
import logging
from datetime import datetime
from functools import wraps
from typing import Dict, Optional
import json
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from werkzeug.exceptions import BadRequest

from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Global variable for RAG pipeline (initialized on startup)
rag_pipeline = None

# Configuration
CONFIG = {
    'MAX_QUERY_LENGTH': 500,
    'MAX_RESPONSE_LENGTH': 1000,
    'DEFAULT_TOP_K': 5,
    'CACHE_SIZE': 100,  # Simple in-memory cache
    'TIMEOUT_SECONDS': 30
}

# Simple in-memory cache for frequently asked questions
response_cache = {}


def init_rag_pipeline():
    """Initialize the RAG pipeline on server startup."""
    global rag_pipeline
    logger.info("🚀 Initializing RAG pipeline...")
    
    try:
        rag_pipeline = RAGPipeline()
        logger.info("✅ RAG pipeline initialized successfully")
        
        # Warm up the model with a dummy query
        logger.info("🔥 Warming up model...")
        _ = rag_pipeline.query("Hello", top_k=1)
        logger.info("✅ Model warmed up")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
        raise


def timer_decorator(f):
    """Decorator to measure endpoint execution time."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # ms
        
        # Add timing to response headers if it's a Flask response
        if hasattr(result, 'headers'):
            result.headers['X-Response-Time'] = f"{execution_time:.0f}ms"
        
        return result
    return wrapper


def validate_request(data: Dict) -> Dict:
    """
    Validate incoming request data.
    
    Args:
        data: Request JSON data
        
    Returns:
        Validated and cleaned data
        
    Raises:
        BadRequest: If validation fails
    """
    # Check required fields
    if 'query' not in data:
        raise BadRequest("Missing required field: 'query'")
    
    query = data['query'].strip()
    
    # Validate query
    if not query:
        raise BadRequest("Query cannot be empty")
    
    if len(query) > CONFIG['MAX_QUERY_LENGTH']:
        raise BadRequest(f"Query too long. Maximum length: {CONFIG['MAX_QUERY_LENGTH']}")
    
    # Extract optional parameters
    top_k = data.get('top_k', CONFIG['DEFAULT_TOP_K'])
    return_sources = data.get('return_sources', False)
    use_cache = data.get('use_cache', True)
    
    # Validate parameters
    if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
        raise BadRequest("top_k must be an integer between 1 and 20")
    
    return {
        'query': query,
        'top_k': top_k,
        'return_sources': return_sources,
        'use_cache': use_cache
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': rag_pipeline is not None
    })


@app.route('/query', methods=['POST'])
@timer_decorator
def query_endpoint():
    """
    Main query endpoint for RAG queries.
    
    Expected JSON payload:
    {
        "query": "Your question here",
        "top_k": 5,  # Optional, number of documents to retrieve
        "return_sources": false,  # Optional, whether to return source documents
        "use_cache": true  # Optional, whether to use cache
    }
    
    Returns:
    {
        "answer": "Generated answer",
        "response_time_ms": 250,
        "sources": [...],  # If return_sources=true
        "cached": false
    }
    """
    try:
        # Validate request
        data = validate_request(request.get_json())
        
        # Check cache if enabled
        cache_key = f"{data['query']}:{data['top_k']}"
        if data['use_cache'] and cache_key in response_cache:
            logger.info(f"Cache hit for query: {data['query'][:50]}...")
            cached_response = response_cache[cache_key].copy()
            cached_response['cached'] = True
            return jsonify(cached_response)
        
        # Process query
        logger.info(f"Processing query: {data['query'][:50]}...")
        start_time = time.time()
        
        result = rag_pipeline.query(
            question=data['query'],
            top_k=data['top_k'],
            return_sources=data['return_sources']
        )
        
        # Prepare response
        response = {
            'answer': result['answer'][:CONFIG['MAX_RESPONSE_LENGTH']],
            'response_time_ms': result['response_time_ms'],
            'cached': False
        }
        
        if data['return_sources']:
            response['sources'] = result['sources']
        
        # Update cache (maintain size limit)
        if data['use_cache']:
            if len(response_cache) >= CONFIG['CACHE_SIZE']:
                # Remove oldest entry (simple FIFO)
                response_cache.pop(next(iter(response_cache)))
            response_cache[cache_key] = response
        
        # Log performance
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Query processed in {total_time:.0f}ms")
        
        if total_time > 350:
            logger.warning(f"⚠️ Response time ({total_time:.0f}ms) exceeds target (350ms)")
        
        return jsonify(response)
        
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def stats_endpoint():
    """Get performance statistics."""
    try:
        stats = rag_pipeline.get_performance_stats()
        
        return jsonify({
            'performance': stats,
            'cache': {
                'size': len(response_cache),
                'max_size': CONFIG['CACHE_SIZE']
            },
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the response cache."""
    response_cache.clear()
    return jsonify({
        'message': 'Cache cleared successfully',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


# Initialize on import (for production with gunicorn)
if rag_pipeline is None:
    init_rag_pipeline()


if __name__ == '__main__':
    # Development server
    logger.info("Starting Flask development server...")
    
    # Use threaded=False to avoid issues with PyTorch models
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False for production
        threaded=False
    )
