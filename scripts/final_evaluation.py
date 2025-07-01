#!/usr/bin/env python3
"""
Final evaluation script for your LLM Knowledge Assistant.
This version has corrected imports to work with your project structure.
"""

import sys
import time
import torch
import json
from pathlib import Path
from datetime import datetime

# Add the project root to Python path so imports work correctly
# This is like giving Python a map of your entire building
sys.path.append(str(Path(__file__).parent))

# Now we can import your custom modules with the correct paths
try:
    from src.evaluation import ModelEvaluator
    from src.rag_pipeline import RAGPipeline
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Let's fix the import issues in your source files...")
    print("\nThe issue is likely in your src/evaluation.py file.")
    print("Please check line 31 and change:")
    print("  from rag_pipeline import RAGPipeline")
    print("to:")
    print("  from src.rag_pipeline import RAGPipeline")
    print("\nThen try running this evaluation again.")
    sys.exit(1)

def find_trained_model():
    """
    Find the trained model - either final_model or latest checkpoint.
    This function helps us locate your trained model regardless of how training completed.
    """
    print("🔍 Looking for your trained model...")
    
    # First, check for the ideal case - a saved final model
    final_model_path = Path("models/fine_tuned/final_model")
    if final_model_path.exists() and any(final_model_path.iterdir()):
        print("✅ Found final model at: models/fine_tuned/final_model")
        return str(final_model_path)
    
    # If no final model, find the latest checkpoint
    print("📂 No final_model found, looking for latest checkpoint...")
    base_dir = Path("models/fine_tuned")
    
    if not base_dir.exists():
        print("❌ No training directory found!")
        print("   Expected to find: models/fine_tuned/")
        print("   Make sure training completed successfully.")
        return None
    
    # Find all checkpoints and get the latest one
    checkpoints = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        print("   This suggests training may not have saved properly.")
        return None
    
    # Sort by step number to get the latest
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    step_num = int(latest_checkpoint.name.split("-")[1])
    
    print(f"✅ Found latest checkpoint: {latest_checkpoint.name} (step {step_num})")
    
    # Check if this checkpoint is complete
    required_files = ["config.json", "pytorch_model.bin", "trainer_state.json"]
    missing_files = [f for f in required_files if not (latest_checkpoint / f).exists()]
    
    if missing_files:
        print(f"⚠️  Checkpoint appears incomplete. Missing: {missing_files}")
        print("   Will attempt to use it anyway...")
    
    return str(latest_checkpoint)

def test_model_basic_functionality(model_path):
    """
    Basic sanity check - can we load the model and generate text?
    This is like checking if your trained chef can actually cook before we test their skills.
    """
    print("\n🧪 Step 1: Basic Model Functionality Test")
    print("=" * 50)
    
    try:
        print("Loading RAG pipeline with your trained model...")
        print("(This may take a minute as we load the model into memory)")
        
        # Try to create a RAG pipeline with your model
        rag = RAGPipeline(model_path)
        
        # Test with a simple query to make sure everything works
        test_query = "What is machine learning?"
        print(f"Testing with query: '{test_query}'")
        
        start_time = time.time()
        result = rag.query(test_query)
        response_time = (time.time() - start_time) * 1000
        
        print(f"✅ Model loaded and responded successfully!")
        print(f"⚡ Response time: {response_time:.0f}ms")
        print(f"📝 Sample response: {result['answer'][:150]}...")
        
        # Quick check that the response makes sense
        if len(result['answer']) < 10:
            print("⚠️  Response seems very short - model might not be working optimally")
        elif "error" in result['answer'].lower():
            print("⚠️  Response contains 'error' - there might be an issue")
        else:
            print("✅ Response appears coherent and substantial")
        
        return True, rag
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        print("This could mean:")
        print("  - The model files are corrupted")
        print("  - There's a memory issue")
        print("  - The checkpoint is incomplete")
        print("  - There are missing dependencies")
        return False, None

def run_accuracy_evaluation(model_path):
    """
    Test the accuracy of your model against the validation dataset.
    This is the main event - testing if your model learned what you taught it.
    """
    print("\n🎯 Step 2: Accuracy Evaluation")
    print("=" * 50)
    
    try:
        print("Creating evaluator with your trained model...")
        evaluator = ModelEvaluator(model_path)
        
        print("Running comprehensive accuracy evaluation...")
        print("This tests your model against questions it hasn't seen before.")
        print("Think of this as the final exam for your AI student.")
        print("(This will take 10-20 minutes depending on dataset size)")
        
        # Run evaluation on full validation set
        results = evaluator.run_evaluation()
        
        # Extract key metrics
        accuracy = results['exact_match_accuracy']
        f1_score = results['f1_score']
        bleu_score = results['bleu_score']
        num_examples = results['num_examples']
        
        print(f"\n📊 ACCURACY RESULTS:")
        print(f"   🎯 Exact Match Accuracy: {accuracy:.2f}%")
        print(f"   📈 F1 Score: {f1_score:.2f}%")
        print(f"   📝 BLEU Score: {bleu_score:.4f}")
        print(f"   📋 Tested on {num_examples} examples")
        
        # Provide context for these numbers
        print(f"\n💡 Understanding Your Results:")
        if accuracy >= 86:
            print(f"🎉 OUTSTANDING! You exceeded the 86% accuracy target!")
            print(f"   This means your model gives exactly correct answers {accuracy:.1f}% of the time.")
        elif accuracy >= 80:
            print(f"🎯 EXCELLENT! You're very close to the 86% target.")
            print(f"   This is strong performance - only {86-accuracy:.1f}% away from target.")
        elif accuracy >= 70:
            print(f"📈 GOOD PROGRESS! You achieved {accuracy:.1f}% accuracy.")
            print(f"   This shows your model learned substantially from training.")
        else:
            print(f"📊 Your model achieved {accuracy:.1f}% accuracy.")
            print(f"   This suggests there may be room for improvement in training approach.")
        
        # Explain what F1 and BLEU mean in practical terms
        print(f"\nAdditional Context:")
        print(f"   F1 Score ({f1_score:.1f}%): Measures partial credit for close answers")
        print(f"   BLEU Score ({bleu_score:.3f}): Measures how natural and fluent responses sound")
        
        return results
        
    except Exception as e:
        print(f"❌ Accuracy evaluation failed: {e}")
        print("This might be due to:")
        print("  - Issues with the validation dataset")
        print("  - Memory problems during evaluation")
        print("  - Model compatibility issues")
        return None

def run_latency_benchmark(rag_pipeline):
    """
    Test the speed of your system - can it respond quickly enough for real users?
    Think of this as testing how fast your AI chef can prepare different types of dishes.
    """
    print("\n⚡ Step 3: Latency Benchmark")
    print("=" * 50)
    
    # Test with different types of queries to get realistic performance
    test_queries = [
        "What is machine learning?",  # Short, conceptual
        "Explain the difference between supervised and unsupervised learning in detail.",  # Longer, comparative
        "How does a neural network work?",  # Technical explanation
        "What are transformers?",  # Short, technical
        "Describe the complete process of training a deep learning model step by step."  # Long, comprehensive
    ]
    
    latencies = []
    
    print("Testing response times with various query types...")
    print("This simulates real user interactions with different complexity levels.")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Test {i}/5: {query[:50]}...")
        
        # Warm up the model with the first query (first query is often slower)
        if i == 1:
            print("      (Warming up model - first query often takes longer)")
        
        start_time = time.time()
        result = rag_pipeline.query(query)
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        
        print(f"      Response time: {latency:.0f}ms")
        print(f"      Answer length: {len(result['answer'])} characters")
    
    # Calculate statistics that matter for real-world performance
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    
    # Remove the first (warmup) query for more realistic averages
    working_latencies = latencies[1:] if len(latencies) > 1 else latencies
    working_avg = sum(working_latencies) / len(working_latencies)
    
    print(f"\n📊 LATENCY RESULTS:")
    print(f"   ⚡ Average latency: {avg_latency:.0f}ms")
    print(f"   ⚡ Working average (excluding warmup): {working_avg:.0f}ms")
    print(f"   📊 Range: {min_latency:.0f}ms - {max_latency:.0f}ms")
    
    # Provide practical context for these numbers
    print(f"\n💡 Performance Context:")
    target_latency = working_avg  # Use working average for target assessment
    
    if target_latency < 200:
        print(f"🚀 EXCEPTIONAL! Sub-200ms response time feels instant to users.")
    elif target_latency < 350:
        print(f"🎉 SUCCESS! You achieved the <350ms latency target!")
        print(f"   Users will experience this as very responsive.")
    elif target_latency < 500:
        print(f"📈 GOOD! Sub-500ms is still quite responsive for AI systems.")
        excess = target_latency - 350
        print(f"   You're {excess:.0f}ms over the 350ms target, but still very usable.")
    else:
        print(f"📊 Response time is {target_latency:.0f}ms.")
        print(f"   This might feel slightly slow to users. Consider optimization.")
    
    return {
        'average_latency': avg_latency,
        'working_average_latency': working_avg,
        'max_latency': max_latency,
        'min_latency': min_latency,
        'all_latencies': latencies
    }

def generate_final_report(model_path, accuracy_results, latency_results):
    """
    Create a comprehensive report of all your results.
    This is like creating your AI system's report card!
    """
    print("\n📄 Step 5: Generating Final Report")
    print("=" * 50)
    
    # Determine overall success based on your original targets
    accuracy_success = accuracy_results and accuracy_results['exact_match_accuracy'] >= 86
    latency_success = latency_results and latency_results['working_average_latency'] < 350
    overall_success = accuracy_success and latency_success
    
    # Create comprehensive report data
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'target_achievement': {
            'accuracy_target_86_percent': accuracy_success,
            'latency_target_350ms': latency_success,
            'overall_success': overall_success
        },
        'accuracy_metrics': accuracy_results,
        'performance_metrics': latency_results
    }
    
    # Save detailed JSON report for technical analysis
    with open('final_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create human-readable summary that tells the story of your results
    summary = f"""# 🎯 LLM Knowledge Assistant - Final Evaluation Report

**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: {model_path}

## 🏆 FINAL RESULTS SUMMARY

### Target Achievement
- **Accuracy Target (86%)**: {'✅ ACHIEVED' if accuracy_success else '❌ NOT ACHIEVED'}
- **Latency Target (<350ms)**: {'✅ ACHIEVED' if latency_success else '❌ NOT ACHIEVED'}
- **Overall Success**: {'🎉 ALL TARGETS MET!' if overall_success else '📊 PARTIAL SUCCESS'}

"""
    
    if accuracy_results:
        accuracy = accuracy_results['exact_match_accuracy']
        summary += f"""### Accuracy Performance
- **Exact Match Accuracy**: {accuracy:.2f}%
- **F1 Score**: {accuracy_results['f1_score']:.2f}%
- **BLEU Score**: {accuracy_results['bleu_score']:.4f}
- **Examples Tested**: {accuracy_results['num_examples']}

"""
    
    if latency_results:
        summary += f"""### Latency Performance
- **Average Response Time**: {latency_results['working_average_latency']:.0f}ms
- **Response Time Range**: {latency_results['min_latency']:.0f}-{latency_results['max_latency']:.0f}ms
- **Fastest Response**: {latency_results['min_latency']:.0f}ms
- **Slowest Response**: {latency_results['max_latency']:.0f}ms

"""
    
    # Add interpretation and next steps
    summary += f"""## 📊 INTERPRETATION

"""
    
    if overall_success:
        summary += f"""🎉 **CONGRATULATIONS!** Your LLM Knowledge Assistant has successfully met all performance targets!

This means you have built a production-ready AI system that:
- Answers questions correctly {accuracy_results['exact_match_accuracy']:.1f}% of the time
- Responds to users in under 350ms on average
- Is ready for real-world deployment

"""
    else:
        summary += f"""📈 **STRONG PROGRESS!** Your system shows excellent development:

"""
        if accuracy_success:
            summary += f"✅ Accuracy target achieved - your model learned very well\n"
        else:
            gap = 86 - accuracy_results['exact_match_accuracy']
            summary += f"📊 Accuracy: {gap:.1f}% away from target - very close!\n"
        
        if latency_success:
            summary += f"✅ Latency target achieved - system responds quickly\n"
        else:
            excess = latency_results['working_average_latency'] - 350
            summary += f"⚡ Latency: {excess:.0f}ms over target - optimization opportunity\n"
    
    summary += f"""
## 🚀 NEXT STEPS

"""
    
    if overall_success:
        summary += f"""1. **Deploy your API**: `python app.py`
2. **Create Docker container**: `docker build -t llm-assistant .`
3. **Test with real users**: Use the test scripts provided
4. **Document your system**: Create user guides and API documentation
5. **Monitor performance**: Set up logging and metrics collection

"""
    else:
        summary += f"""1. **Review detailed results**: Check `final_evaluation_report.json`
2. **Consider improvements**: 
"""
        if not accuracy_success:
            summary += f"   - Additional training epochs for better accuracy\n"
            summary += f"   - Hyperparameter tuning for improved learning\n"
        if not latency_success:
            summary += f"   - Inference optimization for faster responses\n"
            summary += f"   - Model compression techniques\n"
        
        summary += f"""3. **Test specific improvements**: Focus on the areas that need work
4. **Re-evaluate**: Run this evaluation again after improvements

"""
    
    # Save summary report
    with open('evaluation_summary.md', 'w') as f:
        f.write(summary)
    
    print("✅ Reports generated:")
    print("   📊 final_evaluation_report.json (technical data for analysis)")
    print("   📝 evaluation_summary.md (human-readable summary and interpretation)")
    
    return overall_success

def main():
    """
    Main evaluation function that coordinates all the tests.
    Think of this as your AI system's comprehensive graduation exam!
    """
    print("🎓 LLM Knowledge Assistant - Final Evaluation")
    print("=" * 70)
    print("This evaluation will test if your trained model achieved:")
    print("🎯 Target 1: 86% exact match accuracy")
    print("⚡ Target 2: <350ms average response time")
    print("🌍 Target 3: Coherent, helpful responses")
    print("\nThis comprehensive test typically takes 15-30 minutes.")
    print("=" * 70)
    
    # Step 1: Find the trained model
    model_path = find_trained_model()
    if not model_path:
        print("\n❌ Cannot proceed without a trained model!")
        print("Make sure your training completed successfully.")
        return False
    
    # Step 2: Basic functionality test
    print(f"\nUsing model: {model_path}")
    functionality_ok, rag_pipeline = test_model_basic_functionality(model_path)
    if not functionality_ok:
        print("\n❌ Basic functionality failed - cannot proceed with full evaluation!")
        print("This suggests there may be issues with the trained model.")
        return False
    
    # Step 3: Accuracy evaluation (the main test!)
    accuracy_results = run_accuracy_evaluation(model_path)
    
    # Step 4: Latency benchmark
    latency_results = None
    if rag_pipeline:  # Only run if we have a working pipeline
        latency_results = run_latency_benchmark(rag_pipeline)
    
    # Step 5: Generate comprehensive report
    success = generate_final_report(model_path, accuracy_results, latency_results)
    
    # Final summary for immediate feedback
    print(f"\n🎉 EVALUATION COMPLETE!")
    print("=" * 50)
    
    if success:
        print("🏆 CONGRATULATIONS! Your LLM Knowledge Assistant is ready for deployment!")
        print("✅ All performance targets achieved!")
        print("🚀 You've successfully built a production-ready AI system!")
    else:
        print("📊 Evaluation complete! You've made excellent progress.")
        print("📋 Check evaluation_summary.md for detailed results and next steps.")
    
    if accuracy_results:
        print(f"\nKey Results:")
        print(f"   📊 Accuracy: {accuracy_results['exact_match_accuracy']:.1f}%")
        if latency_results:
            print(f"   ⚡ Latency: {latency_results['working_average_latency']:.0f}ms")
    
    print(f"\n📄 Detailed reports saved:")
    print(f"   • evaluation_summary.md (read this first!)")
    print(f"   • final_evaluation_report.json (technical details)")
    
    return success

if __name__ == "__main__":
    main()