#!/usr/bin/env python3
"""
Main execution script for the LLM Knowledge Assistant pipeline.
Run this to execute the complete training and evaluation pipeline.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging
import time
import requests
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the complete pipeline execution."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        
    def check_environment(self):
        """Check if environment is properly set up."""
        logger.info("🔍 Checking environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            logger.error("❌ Python 3.8+ required")
            return False
        logger.info(f"✅ Python {python_version.major}.{python_version.minor} detected")
        
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("⚠️  No GPU detected. Training will be slow!")
        except ImportError:
            logger.error("❌ PyTorch not installed. Run: pip install -r requirements.txt")
            return False
        
        # Check if directories exist
        required_dirs = ['data/raw', 'src', 'configs']
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                logger.error(f"❌ Directory {dir_path} not found. Run initial_setup.py first!")
                return False
        
        # Check for raw data
        raw_data_files = list(Path('data/raw').glob('*'))
        if not raw_data_files:
            logger.error("❌ No data found in data/raw/. Please add your documents!")
            return False
        logger.info(f"✅ Found {len(raw_data_files)} files in data/raw/")
        
        # Check Hugging Face token
        if not os.getenv('HUGGINGFACE_TOKEN') and not Path.home().joinpath('.huggingface/token').exists():
            logger.error("❌ Hugging Face token not found. Run initial_setup.py to configure!")
            return False
        logger.info("✅ Hugging Face token configured")
        
        return True
    
    def run_data_processing(self):
        """Run the data processing pipeline."""
        logger.info("\n📊 STEP 1: Data Processing")
        logger.info("=" * 50)
        
        try:
            from src.data_processing import DocumentProcessor
            processor = DocumentProcessor()
            processor.process_all()
            logger.info("✅ Data processing completed")
            return True
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            return False
    
    def run_fine_tuning(self, skip_if_exists=True):
        """Run the fine-tuning process."""
        logger.info("\n🎯 STEP 2: Fine-tuning")
        logger.info("=" * 50)
        
        # Check if model already exists
        model_path = Path("models/fine_tuned/final_model")
        if skip_if_exists and model_path.exists():
            logger.info("ℹ️  Fine-tuned model already exists. Skipping training.")
            logger.info("   To retrain, delete models/fine_tuned/ or use --force-retrain")
            return True
        
        try:
            from src.fine_tuning import LlamaFineTuner
            fine_tuner = LlamaFineTuner()
            fine_tuner.run_full_pipeline()
            logger.info("✅ Fine-tuning completed")
            return True
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            return False
    
    def run_evaluation(self):
        """Run model evaluation."""
        logger.info("\n🧪 STEP 3: Evaluation")
        logger.info("=" * 50)
        
        try:
            from src.evaluation import ModelEvaluator
            evaluator = ModelEvaluator()
            
            # Run evaluation on sample for quick test
            results = evaluator.run_evaluation(sample_size=100)
            
            # Generate report
            report = evaluator.generate_report(results)
            
            # Print key metrics
            logger.info("\n📊 Key Metrics:")
            logger.info(f"   Exact Match Accuracy: {results['exact_match_accuracy']:.2f}%")
            logger.info(f"   Mean Latency: {results['latency_metrics']['mean_ms']:.0f}ms")
            
            return True
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return False
    
    def test_api(self):
        """Test the Flask API."""
        logger.info("\n🌐 STEP 4: API Testing")
        logger.info("=" * 50)
        
        # Start Flask server in background
        logger.info("Starting Flask server...")
        server_process = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(10)
        
        try:
            # Test health endpoint
            logger.info("Testing health endpoint...")
            response = requests.get("http://localhost:5000/health")
            if response.status_code == 200:
                logger.info("✅ Health check passed")
            else:
                logger.error("❌ Health check failed")
                return False
            
            # Test query endpoint
            logger.info("Testing query endpoint...")
            test_queries = [
                "What is the main purpose of this system?",
                "How does the retrieval process work?",
                "What are the key performance metrics?"
            ]
            
            for query in test_queries:
                payload = {
                    "query": query,
                    "top_k": 3,
                    "return_sources": True
                }
                
                start_time = time.time()
                response = requests.post(
                    "http://localhost:5000/query",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Query processed in {latency:.0f}ms")
                    logger.info(f"   Answer: {data['answer'][:100]}...")
                else:
                    logger.error(f"❌ Query failed: {response.text}")
            
            # Test stats endpoint
            response = requests.get("http://localhost:5000/stats")
            if response.status_code == 200:
                stats = response.json()
                logger.info("✅ Stats endpoint working")
                logger.info(f"   Performance stats: {stats['performance']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ API testing failed: {e}")
            return False
        finally:
            # Stop server
            server_process.terminate()
            server_process.wait()
            logger.info("Flask server stopped")
    
    def run_complete_pipeline(self, args):
        """Run the complete pipeline."""
        logger.info("🚀 Starting LLM Knowledge Assistant Pipeline")
        logger.info("=" * 70)
        
        # Check environment
        if not self.check_environment():
            logger.error("\n❌ Environment check failed. Please fix the issues above.")
            return False
        
        # Run data processing
        if not args.skip_data_processing:
            if not self.run_data_processing():
                logger.error("\n❌ Data processing failed. Pipeline stopped.")
                return False
        else:
            logger.info("\n⏭️  Skipping data processing (--skip-data-processing flag)")
        
        # Run fine-tuning
        if not args.skip_training:
            if not self.run_fine_tuning(skip_if_exists=not args.force_retrain):
                logger.error("\n❌ Fine-tuning failed. Pipeline stopped.")
                return False
        else:
            logger.info("\n⏭️  Skipping training (--skip-training flag)")
        
        # Run evaluation
        if not args.skip_evaluation:
            if not self.run_evaluation():
                logger.error("\n❌ Evaluation failed. Pipeline stopped.")
                return False
        else:
            logger.info("\n⏭️  Skipping evaluation (--skip-evaluation flag)")
        
        # Test API
        if args.test_api:
            if not self.test_api():
                logger.error("\n❌ API testing failed.")
                return False
        
        logger.info("\n✅ Pipeline completed successfully!")
        logger.info("\n📋 Next steps:")
        logger.info("1. Review the evaluation report: evaluation_report.md")
        logger.info("2. Start the API server: python app.py")
        logger.info("3. Deploy with Docker: docker build -t llm-assistant .")
        
        return True


def create_test_client():
    """Create a simple test client script."""
    client_script = '''#!/usr/bin/env python3
"""
Simple test client for the LLM Knowledge Assistant API.
"""

import requests
import json
import time

API_URL = "http://localhost:5000"

def test_query(query, top_k=5, return_sources=False):
    """Send a query to the API and print results."""
    payload = {
        "query": query,
        "top_k": top_k,
        "return_sources": return_sources
    }
    
    print(f"\\n🔍 Query: {query}")
    
    start_time = time.time()
    response = requests.post(
        f"{API_URL}/query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    total_time = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! (Total time: {total_time:.0f}ms)")
        print(f"📝 Answer: {data['answer']}")
        print(f"⏱️  Response time: {data['response_time_ms']}ms")
        
        if return_sources and 'sources' in data:
            print("\\n📚 Sources:")
            for i, source in enumerate(data['sources'][:3]):
                print(f"   {i+1}. {source['text'][:100]}...")
                print(f"      Score: {source['score']:.3f}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def main():
    # Check health
    print("🏥 Checking API health...")
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        print("✅ API is healthy")
    else:
        print("❌ API is not responding")
        return
    
    # Test queries
    test_queries = [
        "What is the main purpose of this system?",
        "How does the retrieval process work?",
        "What are the key performance metrics?",
        "Explain the LoRA fine-tuning approach",
        "What is the target latency for API responses?"
    ]
    
    for query in test_queries:
        test_query(query, return_sources=True)
        time.sleep(0.5)  # Small delay between queries
    
    # Get stats
    print("\\n📊 Performance Statistics:")
    response = requests.get(f"{API_URL}/stats")
    if response.status_code == 200:
        stats = response.json()
        print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
'''
    
    with open("test_client.py", "w") as f:
        f.write(client_script)
    os.chmod("test_client.py", 0o755)
    logger.info("✅ Created test_client.py")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the LLM Knowledge Assistant pipeline"
    )
    
    parser.add_argument(
        "--skip-data-processing",
        action="store_true",
        help="Skip data processing step"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip fine-tuning step"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step"
    )
    
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if model exists"
    )
    
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test the Flask API after pipeline completion"
    )
    
    args = parser.parse_args()
    
    # Create test client
    create_test_client()
    
    # Run pipeline
    runner = PipelineRunner()
    success = runner.run_complete_pipeline(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()