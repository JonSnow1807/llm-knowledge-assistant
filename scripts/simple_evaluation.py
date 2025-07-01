#!/usr/bin/env python3
"""
Simplified evaluation script that works around the ROUGE issue
and focuses on core metrics.
"""

import sys
import time
import torch
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline
from datasets import load_from_disk
import random

def simple_accuracy_test(rag_pipeline, num_samples=50):
    """
    Simple accuracy test without complex metrics.
    Tests exact match and semantic similarity.
    """
    print("🎯 Running Simplified Accuracy Test")
    print(f"Testing on {num_samples} validation samples...")
    
    try:
        # Load validation dataset
        dataset = load_from_disk("data/processed/fine_tuning_dataset")
        val_dataset = dataset['validation']
        
        # Sample random examples
        total_samples = len(val_dataset)
        sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
        
        correct_exact = 0
        correct_partial = 0
        total_latency = 0
        results = []
        
        for i, idx in enumerate(sample_indices):
            print(f"   Testing {i+1}/{len(sample_indices)}...")
            
            example = val_dataset[idx]
            
            # Extract question and answer from the tokenized format
            if 'input_ids' in example:
                # This might be tokenized - let's work with it
                continue  # Skip tokenized examples for simplicity
            elif 'text' in example:
                text = example['text']
            else:
                continue
            
            # Parse instruction format
            if "### Input:" in text and "### Response:" in text:
                question = text.split("### Input:")[1].split("### Response:")[0].strip()
                expected = text.split("### Response:")[-1].strip()
                
                # Generate response
                start_time = time.time()
                result = rag_pipeline.query(question)
                latency = (time.time() - start_time) * 1000
                total_latency += latency
                
                response = result['answer'].strip()
                
                # Simple exact match (normalized)
                expected_norm = expected.lower().strip()
                response_norm = response.lower().strip()
                
                exact_match = expected_norm == response_norm
                
                # Partial match (key word overlap)
                expected_words = set(expected_norm.split())
                response_words = set(response_norm.split())
                
                if len(expected_words) > 0:
                    overlap = len(expected_words.intersection(response_words)) / len(expected_words)
                    partial_match = overlap > 0.6
                else:
                    partial_match = exact_match
                
                if exact_match:
                    correct_exact += 1
                if partial_match:
                    correct_partial += 1
                
                results.append({
                    'question': question[:100],
                    'expected': expected[:100],
                    'response': response[:100],
                    'exact_match': exact_match,
                    'partial_match': partial_match,
                    'latency_ms': latency
                })
        
        if results:
            exact_accuracy = (correct_exact / len(results)) * 100
            partial_accuracy = (correct_partial / len(results)) * 100
            avg_latency = total_latency / len(results)
            
            print(f"\n📊 ACCURACY RESULTS:")
            print(f"   🎯 Exact Match: {exact_accuracy:.1f}% ({correct_exact}/{len(results)})")
            print(f"   📈 Partial Match: {partial_accuracy:.1f}% ({correct_partial}/{len(results)})")
            print(f"   ⚡ Avg Latency: {avg_latency:.0f}ms")
            
            return {
                'exact_match_accuracy': exact_accuracy,
                'partial_match_accuracy': partial_accuracy,
                'num_examples': len(results),
                'avg_latency_ms': avg_latency,
                'samples': results[:5]
            }
        else:
            print("❌ No valid samples found for testing")
            return None
            
    except Exception as e:
        print(f"❌ Accuracy test failed: {e}")
        return None

def diagnose_slow_generation(rag_pipeline):
    """
    Diagnose why generation is so slow (13 seconds instead of <350ms).
    """
    print("\n🔍 Diagnosing Generation Speed")
    print("=" * 50)
    
    test_queries = [
        "What is AI?",  # Very short
        "Explain machine learning briefly.",  # Medium
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        
        # Time components separately
        start_total = time.time()
        
        # Retrieval timing
        start_retrieval = time.time()
        retrieved_docs = rag_pipeline.retrieve_documents(query)
        retrieval_time = (time.time() - start_retrieval) * 1000
        
        # Generation timing  
        start_generation = time.time()
        response = rag_pipeline.generate_response(query, retrieved_docs, max_new_tokens=50)  # Shorter response
        generation_time = (time.time() - start_generation) * 1000
        
        total_time = (time.time() - start_total) * 1000
        
        print(f"   📊 Retrieval: {retrieval_time:.0f}ms")
        print(f"   🤖 Generation: {generation_time:.0f}ms") 
        print(f"   ⚡ Total: {total_time:.0f}ms")
        print(f"   📝 Response: {response[:100]}...")
        
        if generation_time > 1000:
            print(f"   ⚠️  Generation is very slow ({generation_time:.0f}ms)")

def main():
    """Simplified evaluation focusing on core functionality."""
    print("🔍 Simplified LLM Evaluation")
    print("=" * 50)
    
    # Load model
    model_path = "models/fine_tuned/final_model"
    print(f"Loading model: {model_path}")
    
    try:
        rag = RAGPipeline(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Quick functionality test
    print("\n🧪 Quick Functionality Test")
    test_query = "What is machine learning?"
    start_time = time.time()
    result = rag.query(test_query)
    response_time = (time.time() - start_time) * 1000
    
    print(f"✅ Response generated in {response_time:.0f}ms")
    print(f"📝 Response: {result['answer'][:200]}...")
    
    # Diagnose speed issues
    diagnose_slow_generation(rag)
    
    # Simple accuracy test
    accuracy_results = simple_accuracy_test(rag, num_samples=20)  # Smaller sample for speed
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    if accuracy_results:
        print(f"Exact Match Accuracy: {accuracy_results['exact_match_accuracy']:.1f}%")
        print(f"Partial Match Accuracy: {accuracy_results['partial_match_accuracy']:.1f}%")
        print(f"Average Latency: {accuracy_results['avg_latency_ms']:.0f}ms")
        
        # Assessment
        if accuracy_results['exact_match_accuracy'] >= 86:
            print("🎯 ✅ Accuracy target achieved!")
        else:
            gap = 86 - accuracy_results['exact_match_accuracy']
            print(f"📊 Accuracy: {gap:.1f}% away from 86% target")
        
        if accuracy_results['avg_latency_ms'] < 350:
            print("⚡ ✅ Latency target achieved!")
        else:
            excess = accuracy_results['avg_latency_ms'] - 350
            print(f"⚡ Latency: {excess:.0f}ms over 350ms target")
    
    print(f"\n🔧 Speed Optimization Needed:")
    print(f"   Current: ~13,000ms")
    print(f"   Target: <350ms") 
    print(f"   Improvement needed: ~37x faster")

if __name__ == "__main__":
    main()