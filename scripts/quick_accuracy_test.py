#!/usr/bin/env python3
"""
Quick accuracy test that handles different dataset formats.
"""

import sys
import time
import random
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline
from datasets import load_from_disk

def find_dataset_format():
    """Check what format your validation data is in."""
    try:
        dataset = load_from_disk("data/processed/fine_tuning_dataset")
        val_dataset = dataset['validation']
        
        print(f"📊 Validation dataset info:")
        print(f"   Size: {len(val_dataset)}")
        print(f"   Columns: {list(val_dataset.column_names)}")
        
        # Check first example
        first_example = val_dataset[0]
        print(f"   First example keys: {list(first_example.keys())}")
        
        # Try to show content based on format
        if 'text' in first_example:
            print(f"   Sample text: {first_example['text'][:200]}...")
            return 'text'
        elif 'input_ids' in first_example:
            print(f"   Data is tokenized (input_ids format)")
            print(f"   Input IDs length: {len(first_example['input_ids'])}")
            return 'tokenized'
        else:
            print(f"   Unknown format: {first_example}")
            return 'unknown'
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def test_with_manual_questions(rag_pipeline):
    """Test with manually created questions since dataset parsing failed."""
    print("\n🧪 Manual Accuracy Test")
    print("=" * 40)
    
    # Create test questions that should work well with a ML knowledge base
    test_cases = [
        {
            "question": "What is machine learning?",
            "expected_keywords": ["algorithm", "data", "learn", "pattern", "artificial", "intelligence"]
        },
        {
            "question": "What is the difference between supervised and unsupervised learning?",
            "expected_keywords": ["supervised", "unsupervised", "labeled", "data", "target"]
        },
        {
            "question": "What is a neural network?",
            "expected_keywords": ["neural", "network", "nodes", "layers", "weights", "neurons"]
        },
        {
            "question": "What is deep learning?",
            "expected_keywords": ["deep", "learning", "neural", "network", "layers", "hierarchical"]
        },
        {
            "question": "What is overfitting?",
            "expected_keywords": ["overfitting", "training", "generalization", "test", "validation"]
        }
    ]
    
    correct_responses = 0
    total_latency = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        
        print(f"\n   Test {i}/{len(test_cases)}: {question}")
        
        start_time = time.time()
        result = rag_pipeline.query(question)
        latency = (time.time() - start_time) * 1000
        total_latency += latency
        
        response = result['answer'].lower()
        
        # Check how many expected keywords appear in the response
        found_keywords = [kw for kw in expected_keywords if kw in response]
        keyword_score = len(found_keywords) / len(expected_keywords)
        
        # Consider it correct if it contains at least 60% of expected keywords
        is_correct = keyword_score >= 0.6
        if is_correct:
            correct_responses += 1
        
        print(f"      Response time: {latency:.0f}ms")
        print(f"      Keywords found: {len(found_keywords)}/{len(expected_keywords)} ({keyword_score:.1%})")
        print(f"      Result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
        print(f"      Response: {response[:150]}...")
    
    accuracy = (correct_responses / len(test_cases)) * 100
    avg_latency = total_latency / len(test_cases)
    
    print(f"\n📊 Manual Test Results:")
    print(f"   🎯 Accuracy: {accuracy:.1f}% ({correct_responses}/{len(test_cases)})")
    print(f"   ⚡ Average latency: {avg_latency:.0f}ms")
    
    return accuracy, avg_latency

def main():
    print("🔍 Quick Accuracy Test")
    print("=" * 40)
    
    # Check dataset format
    dataset_format = find_dataset_format()
    
    # Load model
    print(f"\n🤖 Loading model...")
    rag = RAGPipeline("models/fine_tuned/final_model")
    print("✅ Model loaded")
    
    # Run manual test
    accuracy, avg_latency = test_with_manual_questions(rag)
    
    # Assessment
    print(f"\n📋 Assessment:")
    if accuracy >= 80:
        print(f"🎯 Excellent accuracy: {accuracy:.1f}%")
    elif accuracy >= 60:
        print(f"📈 Good accuracy: {accuracy:.1f}%")
    else:
        print(f"📊 Accuracy needs improvement: {accuracy:.1f}%")
    
    if avg_latency < 350:
        print(f"⚡ Latency target achieved: {avg_latency:.0f}ms")
    else:
        print(f"⚠️ Latency over target: {avg_latency:.0f}ms (need <350ms)")

if __name__ == "__main__":
    main()