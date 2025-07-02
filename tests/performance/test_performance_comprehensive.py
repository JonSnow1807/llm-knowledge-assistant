#!/usr/bin/env python3
"""
Comprehensive performance testing suite.
This creates detailed performance reports for your portfolio.
"""

import time
import json
import statistics
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.rag_pipeline import RAGPipeline

class PerformanceTester:
    """Comprehensive performance testing."""
    
    def __init__(self):
        self.rag = RAGPipeline("models/fine_tuned/final_model")
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "system_info": {
                "model": "Llama-3.1-8B-Instruct (Fine-tuned)",
                "architecture": "RAG + FAISS Vector Search",
                "hardware": "GPU-accelerated inference"
            },
            "test_suites": {}
        }
    
    def test_latency_by_complexity(self):
        """Test response times across different query complexities."""
        test_cases = {
            "simple": [
                "What is AI?",
                "Define ML",
                "What is NLP?",
                "Explain CNN",
                "What is RNN?"
            ],
            "medium": [
                "What is machine learning?",
                "Explain supervised learning",
                "What are neural networks?",
                "Describe deep learning",
                "What is reinforcement learning?"
            ],
            "complex": [
                "Compare supervised vs unsupervised learning approaches",
                "Explain the transformer architecture and its advantages",
                "Describe overfitting, its causes, and prevention strategies",
                "What are the key differences between CNNs and RNNs?",
                "Explain the gradient descent optimization algorithm"
            ]
        }
        
        complexity_results = {}
        
        for complexity, queries in test_cases.items():
            print(f"\n🧪 Testing {complexity} queries...")
            times = []
            responses = []
            
            for query in queries:
                start_time = time.time()
                result = self.rag.query(query)
                elapsed = (time.time() - start_time) * 1000
                
                times.append(elapsed)
                responses.append(result['answer'][:100])
                print(f"   {query}: {elapsed:.0f}ms")
            
            complexity_results[complexity] = {
                "query_count": len(queries),
                "avg_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
                "sample_responses": responses
            }
        
        self.results["test_suites"]["latency_by_complexity"] = complexity_results
        return complexity_results
    
    def test_accuracy_scenarios(self):
        """Test accuracy across different knowledge domains."""
        test_scenarios = [
            {
                "category": "Basic Definitions",
                "query": "What is machine learning?",
                "expected_keywords": ["algorithm", "data", "learn", "pattern"]
            },
            {
                "category": "Technical Concepts", 
                "query": "Explain gradient descent",
                "expected_keywords": ["optimization", "minimize", "loss", "gradient"]
            },
            {
                "category": "Practical Applications",
                "query": "What is computer vision used for?",
                "expected_keywords": ["image", "recognition", "detection", "vision"]
            },
            {
                "category": "Problem Solving",
                "query": "How do you prevent overfitting?",
                "expected_keywords": ["regularization", "validation", "dropout", "data"]
            }
        ]
        
        accuracy_results = {}
        
        for scenario in test_scenarios:
            category = scenario["category"]
            query = scenario["query"]
            expected = scenario["expected_keywords"]
            
            print(f"\n🎯 Testing {category}...")
            
            start_time = time.time()
            result = self.rag.query(query)
            response_time = (time.time() - start_time) * 1000
            
            response = result['answer'].lower()
            found_keywords = [kw for kw in expected if kw in response]
            accuracy_score = len(found_keywords) / len(expected)
            
            accuracy_results[category] = {
                "query": query,
                "response_time_ms": response_time,
                "accuracy_score": accuracy_score,
                "keywords_found": found_keywords,
                "keywords_expected": expected,
                "response_preview": result['answer'][:150]
            }
            
            print(f"   Accuracy: {accuracy_score:.1%}")
            print(f"   Speed: {response_time:.0f}ms")
        
        self.results["test_suites"]["accuracy_scenarios"] = accuracy_results
        return accuracy_results
    
    def test_stress_performance(self):
        """Test performance under repeated queries."""
        print(f"\n🔥 Stress testing with 10 rapid queries...")
        
        test_query = "What is deep learning?"
        times = []
        
        for i in range(10):
            start_time = time.time()
            self.rag.query(test_query)
            elapsed = (time.time() - start_time) * 1000
            times.append(elapsed)
            print(f"   Query {i+1}: {elapsed:.0f}ms")
        
        stress_results = {
            "query_count": len(times),
            "avg_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_dev_ms": statistics.stdev(times),
            "performance_degradation": max(times) - min(times)
        }
        
        self.results["test_suites"]["stress_test"] = stress_results
        return stress_results
    
    def generate_comprehensive_report(self):
        """Generate detailed performance report."""
        print(f"\n📊 Generating comprehensive performance report...")
        
        # Run all test suites
        self.test_latency_by_complexity()
        self.test_accuracy_scenarios()
        self.test_stress_performance()
        
        # Calculate overall metrics
        all_times = []
        for suite_name, suite_data in self.results["test_suites"].items():
            if suite_name == "latency_by_complexity":
                for complexity, data in suite_data.items():
                    all_times.append(data["avg_ms"])
            elif suite_name == "accuracy_scenarios":
                for category, data in suite_data.items():
                    all_times.append(data["response_time_ms"])
            elif suite_name == "stress_test":
                all_times.extend([suite_data["avg_ms"]])
        
        self.results["overall_performance"] = {
            "total_tests": len(all_times),
            "global_avg_ms": statistics.mean(all_times),
            "global_median_ms": statistics.median(all_times),
            "target_350ms_achieved": statistics.mean(all_times) < 350,
            "performance_category": self._categorize_performance(statistics.mean(all_times))
        }
        
        # Save detailed report
        report_path = "tests/reports/comprehensive_performance_report.json"
        Path("tests/reports").mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Comprehensive report saved: {report_path}")
        return self.results
    
    def _categorize_performance(self, avg_ms):
        """Categorize performance level."""
        if avg_ms < 350:
            return "Excellent - Real-time capable"
        elif avg_ms < 1000:
            return "Very Good - Interactive applications"
        elif avg_ms < 2000:
            return "Good - Knowledge assistant use case"
        else:
            return "Acceptable - Batch processing suitable"

def main():
    """Run comprehensive performance testing."""
    print("🔬 LLM Knowledge Assistant - Comprehensive Performance Testing")
    print("=" * 70)
    
    tester = PerformanceTester()
    results = tester.generate_comprehensive_report()
    
    # Print summary
    overall = results["overall_performance"]
    print(f"\n🏆 FINAL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {overall['total_tests']}")
    print(f"Average Response Time: {overall['global_avg_ms']:.0f}ms")
    print(f"Performance Category: {overall['performance_category']}")
    print(f"350ms Target Achieved: {'✅ Yes' if overall['target_350ms_achieved'] else '❌ No'}")

if __name__ == "__main__":
    main()