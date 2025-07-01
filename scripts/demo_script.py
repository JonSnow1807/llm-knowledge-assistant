#!/usr/bin/env python3
"""
Interactive demo script for showcasing your LLM Knowledge Assistant.
Use this for presentations and interviews.
"""

import requests
import json
import time

API_URL = "http://localhost:5000"

def demo_query(question, description="", show_sources=False):
    """Demo a single query with nice formatting."""
    print(f"\n{'='*60}")
    if description:
        print(f"🎯 {description}")
    print(f"❓ Question: {question}")
    print("💭 Thinking...")
    
    payload = {
        "query": question,
        "top_k": 3,
        "return_sources": show_sources
    }
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/query", json=payload)
    total_time = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Response ({total_time:.0f}ms total, {data['response_time_ms']}ms processing):")
        print(f"📝 {data['answer']}")
        
        if show_sources and 'sources' in data:
            print(f"\n📚 Sources consulted: {len(data['sources'])} documents")
    else:
        print(f"❌ Error: {response.status_code}")

def run_demo():
    """Run a complete demo showcasing different capabilities."""
    print("🎓 LLM Knowledge Assistant - Live Demo")
    print("🏗️  Architecture: Fine-tuned Llama-3.1-8B + RAG + FAISS Vector Search")
    print("⚡ Performance: ~2s response time for expert-level answers")
    
    # Check if API is running
    try:
        health = requests.get(f"{API_URL}/health")
        if health.status_code == 200:
            print("✅ System Status: Online and Ready")
        else:
            print("❌ System appears to be offline")
            return
    except:
        print("❌ Cannot connect to API. Make sure Flask app is running.")
        return
    
    # Demo different question types
    demo_query(
        "What is machine learning?",
        "Basic concept explanation"
    )
    
    demo_query(
        "What's the difference between supervised and unsupervised learning?",
        "Comparative analysis"
    )
    
    demo_query(
        "Explain overfitting and how to prevent it",
        "Problem-solving knowledge"
    )
    
    demo_query(
        "What are convolutional neural networks?",
        "Technical deep-dive",
        show_sources=True
    )
    
    print(f"\n{'='*60}")
    print("🎉 Demo Complete!")
    print("💡 Key Features Demonstrated:")
    print("   ✅ Expert-level technical accuracy")
    print("   ✅ Fast response times (~2 seconds)")
    print("   ✅ Comprehensive knowledge retrieval")
    print("   ✅ Production-ready API interface")

if __name__ == "__main__":
    run_demo()