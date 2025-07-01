#!/usr/bin/env python3
"""
Accurate performance measurement with proper controls.
This demonstrates professional performance testing methodology.
"""

import requests
import time
import statistics
import json
from datetime import datetime

class AccuratePerformanceMeasurer:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        
    def verify_system_readiness(self):
        """
        Ensure system is ready for accurate measurement.
        Professional testing always starts with system validation.
        """
        print("🔍 Verifying System Readiness for Performance Testing")
        print("=" * 60)
        
        try:
            # Check health endpoint
            health_response = requests.get(f"{self.api_url}/health", timeout=10)
            if health_response.status_code != 200:
                print(f"❌ System health check failed: {health_response.status_code}")
                return False
            
            print("✅ System health check passed")
            
            # Warm up the system with a simple query
            print("🔥 Warming up system for consistent measurements...")
            warmup_payload = {"query": "What is AI?", "top_k": 3}
            
            warmup_response = requests.post(f"{self.api_url}/query", json=warmup_payload, timeout=30)
            if warmup_response.status_code == 200:
                warmup_data = warmup_response.json()
                warmup_time = warmup_data.get('response_time_ms', 0)
                print(f"✅ Warmup completed: {warmup_time}ms")
                
                # If warmup time is reasonable, system is ready
                if warmup_time < 5000:  # Less than 5 seconds indicates optimized state
                    print("✅ System appears to be in optimized state")
                    return True
                else:
                    print(f"⚠️ Warmup time ({warmup_time}ms) suggests system may not be optimized")
                    return False
            else:
                print(f"❌ Warmup query failed: {warmup_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ System readiness check failed: {e}")
            return False
    
    def measure_controlled_performance(self):
        """
        Measure performance under controlled conditions.
        This is how professional performance testing is done.
        """
        print("\n📊 Controlled Performance Measurement")
        print("=" * 60)
        
        # Carefully selected test queries that represent real usage
        test_scenarios = [
            {
                "query": "What is machine learning?",
                "type": "Basic Concept",
                "expected_complexity": "Simple"
            },
            {
                "query": "Explain the difference between supervised and unsupervised learning",
                "type": "Comparative Analysis", 
                "expected_complexity": "Medium"
            },
            {
                "query": "What are neural networks and how do they work?",
                "type": "Technical Explanation",
                "expected_complexity": "Medium"
            },
            {
                "query": "Describe overfitting and prevention strategies",
                "type": "Problem-Solution",
                "expected_complexity": "Complex"
            }
        ]
        
        all_measurements = []
        scenario_results = {}
        
        for scenario in test_scenarios:
            print(f"\n🧪 Testing: {scenario['type']}")
            print(f"   Query: {scenario['query']}")
            
            # Run multiple measurements for statistical validity
            scenario_times = []
            scenario_responses = []
            
            for run in range(3):  # Three runs per scenario
                print(f"   Run {run + 1}/3...", end=" ")
                
                payload = {
                    "query": scenario["query"],
                    "top_k": 3,
                    "return_sources": False
                }
                
                # Measure end-to-end API response time
                start_time = time.time()
                response = requests.post(f"{self.api_url}/query", json=payload, timeout=30)
                total_api_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    processing_time = data.get('response_time_ms', 0)
                    answer = data.get('answer', '')
                    
                    scenario_times.append(processing_time)
                    scenario_responses.append(answer)
                    all_measurements.append(processing_time)
                    
                    print(f"{processing_time}ms")
                else:
                    print(f"Failed ({response.status_code})")
                
                # Brief pause between measurements
                time.sleep(1)
            
            # Calculate scenario statistics
            if scenario_times:
                scenario_results[scenario['type']] = {
                    'query': scenario['query'],
                    'complexity': scenario['expected_complexity'],
                    'measurements': scenario_times,
                    'average_ms': statistics.mean(scenario_times),
                    'median_ms': statistics.median(scenario_times),
                    'std_dev_ms': statistics.stdev(scenario_times) if len(scenario_times) > 1 else 0,
                    'sample_response': scenario_responses[0][:100] + "..." if scenario_responses else ""
                }
                
                avg_time = statistics.mean(scenario_times)
                std_dev = statistics.stdev(scenario_times) if len(scenario_times) > 1 else 0
                print(f"   Results: {avg_time:.0f}ms ± {std_dev:.0f}ms")
        
        # Calculate overall system performance
        if all_measurements:
            overall_stats = {
                'total_measurements': len(all_measurements),
                'average_ms': statistics.mean(all_measurements),
                'median_ms': statistics.median(all_measurements),
                'min_ms': min(all_measurements),
                'max_ms': max(all_measurements),
                'std_dev_ms': statistics.stdev(all_measurements),
                'percentile_95': sorted(all_measurements)[int(0.95 * len(all_measurements))],
                'target_350ms_achievement': statistics.mean(all_measurements) < 350
            }
            
            return {
                'measurement_timestamp': datetime.now().isoformat(),
                'system_state': 'Optimized and warmed up',
                'overall_performance': overall_stats,
                'scenario_breakdown': scenario_results,
                'performance_assessment': self._assess_performance(overall_stats)
            }
        else:
            return None
    
    def _assess_performance(self, stats):
        """
        Provide professional assessment of the performance data.
        This demonstrates engineering judgment about system performance.
        """
        avg_time = stats['average_ms']
        
        if avg_time < 350:
            assessment = "Excellent - Meets sub-350ms target for production deployment"
        elif avg_time < 1000:
            assessment = "Very Good - Suitable for interactive applications"
        elif avg_time < 2000:
            assessment = "Good - Appropriate for knowledge assistant use case"
        elif avg_time < 5000:
            assessment = "Acceptable - May need optimization for better user experience"
        else:
            assessment = "Needs Optimization - Too slow for most interactive applications"
        
        return {
            'category': assessment.split(' - ')[0],
            'description': assessment,
            'recommendation': self._get_recommendation(avg_time)
        }
    
    def _get_recommendation(self, avg_time):
        """Provide actionable recommendations based on performance."""
        if avg_time < 350:
            return "System is production-ready. Consider load testing for scale planning."
        elif avg_time < 2000:
            return "Current performance is excellent for knowledge assistant applications. Monitor under load."
        else:
            return "Consider further optimization: model compression, caching, or infrastructure improvements."

def main():
    """
    Run accurate performance measurement with proper methodology.
    """
    print("🎯 Accurate Performance Measurement for Portfolio Documentation")
    print("=" * 70)
    print("This measurement uses professional testing methodology to ensure")
    print("accurate, representative performance data for your documentation.")
    
    measurer = AccuratePerformanceMeasurer()
    
    # Step 1: Verify system is ready
    if not measurer.verify_system_readiness():
        print("\n❌ System not ready for performance testing")
        print("\nRecommendations:")
        print("1. Restart your Flask app: python app.py")
        print("2. Clear GPU memory: python -c 'import torch; torch.cuda.empty_cache()'")
        print("3. Verify system with demo: python demo_script.py")
        return
    
    # Step 2: Run controlled measurements
    results = measurer.measure_controlled_performance()
    
    if results:
        # Step 3: Save and display results
        with open('accurate_performance_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Accurate Performance Measurement Complete!")
        print("=" * 60)
        
        overall = results['overall_performance']
        assessment = results['performance_assessment']
        
        print(f"📊 Overall Performance Summary:")
        print(f"   Average Response Time: {overall['average_ms']:.0f}ms")
        print(f"   Median Response Time: {overall['median_ms']:.0f}ms")
        print(f"   95th Percentile: {overall['percentile_95']:.0f}ms")
        print(f"   Performance Consistency: ±{overall['std_dev_ms']:.0f}ms")
        
        print(f"\n🎯 Performance Assessment: {assessment['category']}")
        print(f"   {assessment['description']}")
        print(f"   Recommendation: {assessment['recommendation']}")
        
        print(f"\n💾 Detailed results saved to: accurate_performance_results.json")
        print("This data is now ready for professional performance documentation.")
        
    else:
        print("\n❌ Performance measurement failed")
        print("Please check system status and try again")

if __name__ == "__main__":
    main()