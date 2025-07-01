#!/usr/bin/env python3
"""
Test the ultra-fast optimized version.
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline

def speed_test():
    print("⚡ Ultra-Speed Test")
    print("=" * 30)
    
    rag = RAGPipeline("models/fine_tuned/final_model")
    
    # Test short questions for maximum speed
    test_queries = [
        "What is ML?",
        "Define AI.",
        "What is overfitting?",
        "Explain CNN.",
        "What is NLP?"
    ]
    
    times = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        
        start = time.time()
        result = rag.query(query)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        
        print(f"  Time: {elapsed:.0f}ms")
        print(f"  Response: {result['answer'][:100]}...")
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Results:")
    print(f"  Average: {avg_time:.0f}ms")
    print(f"  Range: {min(times):.0f}-{max(times):.0f}ms")
    print(f"  Target: {'✅ ACHIEVED' if avg_time < 350 else '❌ Still optimizing'}")

if __name__ == "__main__":
    speed_test()