#!/usr/bin/env python3
"""
Master test runner for the LLM Knowledge Assistant.
Runs all test suites and generates comprehensive reports.
"""

import subprocess
import sys
from pathlib import Path

def run_test_suite():
    print("🧪 LLM Knowledge Assistant - Complete Test Suite")
    print("=" * 60)
    
    # Test sequence
    tests = [
        ("Unit Tests", "python tests/unit/test_rag_components.py"),
        ("Performance Tests", "python tests/performance/test_performance_comprehensive.py"),
        # Note: Integration tests require Flask app to be running
    ]
    
    results = {}
    
    for test_name, command in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {test_name} passed")
                results[test_name] = "PASSED"
            else:
                print(f"❌ {test_name} failed")
                print(result.stderr)
                results[test_name] = "FAILED"
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            results[test_name] = "ERROR"
    
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    for test, status in results.items():
        emoji = "✅" if status == "PASSED" else "❌"
        print(f"{emoji} {test}: {status}")

if __name__ == "__main__":
    run_test_suite()
