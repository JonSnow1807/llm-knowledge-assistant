"""
Evaluation module for measuring model performance.
Implements exact match accuracy and other metrics for the knowledge assistant.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from datasets import load_from_disk
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from src.rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation suite for the LLM Knowledge Assistant.
    
    Measures:
    1. Exact match accuracy (primary metric - target: 86%)
    2. F1 score for partial matches
    3. BLEU score for generation quality
    4. Retrieval metrics (precision, recall)
    5. Latency benchmarks
    """
    
    def __init__(self, 
                 model_path: str = "models/fine_tuned/final_model",
                 test_data_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the fine-tuned model
            test_data_path: Path to test dataset (if different from validation)
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline for evaluation...")
        self.rag_pipeline = RAGPipeline(model_path)
        
        # Load evaluation metrics
        self.rouge_metric = evaluate.load('rouge')
        
        # Results storage
        self.results = {
            'predictions': [],
            'references': [],
            'latencies': [],
            'retrieval_scores': []
        }
    
    def load_test_data(self) -> List[Dict]:
        """
        Load test dataset for evaluation.
        
        Returns:
            List of test examples with questions and expected answers
        """
        if self.test_data_path:
            # Load custom test data
            test_data = []
            with open(self.test_data_path, 'r') as f:
                for line in f:
                    test_data.append(json.loads(line))
            logger.info(f"Loaded {len(test_data)} test examples from {self.test_data_path}")
        else:
            # Use validation split from training data
            dataset = load_from_disk("data/processed/fine_tuning_dataset")
            test_data = []
            
            for example in dataset['validation']:
                # Extract question and answer from the formatted text
                text = example['text']
                
                # Parse the instruction format
                if "### Input:" in text and "### Response:" in text:
                    parts = text.split("### Input:")
                    if len(parts) > 1:
                        input_part = parts[1].split("### Response:")[0].strip()
                        response_part = text.split("### Response:")[-1].strip()
                        
                        test_data.append({
                            'question': input_part,
                            'answer': response_part
                        })
            
            logger.info(f"Loaded {len(test_data)} test examples from validation set")
        
        return test_data
    
    def exact_match_score(self, prediction: str, reference: str) -> float:
        """
        Calculate exact match score.
        
        Args:
            prediction: Model's prediction
            reference: Ground truth answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Normalize strings
        pred_normalized = prediction.strip().lower()
        ref_normalized = reference.strip().lower()
        
        # Remove punctuation for comparison
        import string
        translator = str.maketrans('', '', string.punctuation)
        pred_normalized = pred_normalized.translate(translator)
        ref_normalized = ref_normalized.translate(translator)
        
        return float(pred_normalized == ref_normalized)
    
    def token_f1_score(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate token-level F1 score for partial matches.
        
        Args:
            prediction: Model's prediction
            reference: Ground truth answer
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Tokenize
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        # Calculate metrics
        if not pred_tokens and not ref_tokens:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if not pred_tokens or not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        common_tokens = pred_tokens.intersection(ref_tokens)
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """
        Calculate BLEU score for generation quality.
        
        Args:
            prediction: Model's prediction
            reference: Ground truth answer
            
        Returns:
            BLEU score
        """
        # Tokenize
        pred_tokens = prediction.split()
        ref_tokens = [reference.split()]  # BLEU expects list of references
        
        # Calculate BLEU with smoothing
        smoothing = SmoothingFunction()
        bleu = sentence_bleu(
            ref_tokens,
            pred_tokens,
            smoothing_function=smoothing.method1
        )
        
        return bleu
    
    def evaluate_retrieval_quality(self, 
                                 question: str,
                                 expected_answer: str,
                                 retrieved_docs: List[Dict]) -> float:
        """
        Evaluate retrieval quality by checking if relevant information is retrieved.
        
        Args:
            question: The query
            expected_answer: Expected answer
            retrieved_docs: Retrieved documents
            
        Returns:
            Retrieval relevance score (0-1)
        """
        # Combine retrieved texts
        retrieved_text = " ".join([doc['text'] for doc in retrieved_docs])
        
        # Check for key terms from expected answer in retrieved docs
        answer_tokens = set(expected_answer.lower().split())
        retrieved_tokens = set(retrieved_text.lower().split())
        
        # Calculate overlap
        overlap = len(answer_tokens.intersection(retrieved_tokens))
        relevance_score = overlap / len(answer_tokens) if answer_tokens else 0
        
        return min(relevance_score, 1.0)
    
    def run_evaluation(self, 
                      test_data: Optional[List[Dict]] = None,
                      sample_size: Optional[int] = None) -> Dict:
        """
        Run comprehensive evaluation on test data.
        
        Args:
            test_data: Optional test data (loads default if not provided)
            sample_size: Optional sample size for quick evaluation
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("🧪 Starting evaluation...")
        
        # Load test data if not provided
        if test_data is None:
            test_data = self.load_test_data()
        
        # Sample if requested
        if sample_size and sample_size < len(test_data):
            import random
            test_data = random.sample(test_data, sample_size)
            logger.info(f"Using sample of {sample_size} examples")
        
        # Metrics storage
        exact_matches = []
        f1_scores = []
        bleu_scores = []
        retrieval_scores = []
        latencies = []
        
        # Run evaluation
        for example in tqdm(test_data, desc="Evaluating"):
            question = example['question']
            expected_answer = example['answer']
            
            # Time the query
            start_time = time.time()
            
            # Get prediction with retrieval
            result = self.rag_pipeline.query(
                question,
                return_sources=True
            )
            
            prediction = result['answer']
            latency = time.time() - start_time
            
            # Store results
            self.results['predictions'].append(prediction)
            self.results['references'].append(expected_answer)
            self.results['latencies'].append(latency)
            
            # Calculate metrics
            exact_match = self.exact_match_score(prediction, expected_answer)
            f1_result = self.token_f1_score(prediction, expected_answer)
            bleu = self.calculate_bleu(prediction, expected_answer)
            
            # Evaluate retrieval
            retrieval_score = self.evaluate_retrieval_quality(
                question,
                expected_answer,
                result.get('sources', [])
            )
            
            # Store metrics
            exact_matches.append(exact_match)
            f1_scores.append(f1_result['f1'])
            bleu_scores.append(bleu)
            retrieval_scores.append(retrieval_score)
            latencies.append(latency * 1000)  # Convert to ms
        
        # Calculate aggregate metrics
        results = {
            'exact_match_accuracy': np.mean(exact_matches) * 100,
            'f1_score': np.mean(f1_scores) * 100,
            'bleu_score': np.mean(bleu_scores),
            'retrieval_relevance': np.mean(retrieval_scores) * 100,
            'latency_metrics': {
                'mean_ms': np.mean(latencies),
                'median_ms': np.median(latencies),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99)
            },
            'num_examples': len(test_data)
        }
        
        # Calculate ROUGE scores
        rouge_results = self.rouge_metric.compute(
            predictions=self.results['predictions'],
            references=self.results['references']
        )
        
        results['rouge_scores'] = {
            'rouge1': rouge_results['rouge1'] * 100,
            'rouge2': rouge_results['rouge2'] * 100,
            'rougeL': rouge_results['rougeL'] * 100
        }
        
        return results
    
    def generate_report(self, results: Dict, save_path: str = "evaluation_report.md"):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save the report
        """
        report = f"""# LLM Knowledge Assistant Evaluation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: {self.model_path}
**Test Examples**: {results['num_examples']}

## 📊 Performance Metrics

### Accuracy Metrics
- **Exact Match Accuracy**: {results['exact_match_accuracy']:.2f}% {'✅' if results['exact_match_accuracy'] >= 86 else '❌'} (Target: 86%)
- **Token F1 Score**: {results['f1_score']:.2f}%
- **BLEU Score**: {results['bleu_score']:.4f}

### Generation Quality (ROUGE)
- **ROUGE-1**: {results['rouge_scores']['rouge1']:.2f}%
- **ROUGE-2**: {results['rouge_scores']['rouge2']:.2f}%
- **ROUGE-L**: {results['rouge_scores']['rougeL']:.2f}%

### Retrieval Performance
- **Retrieval Relevance**: {results['retrieval_relevance']:.2f}%

### Latency Metrics
- **Mean Response Time**: {results['latency_metrics']['mean_ms']:.0f}ms {'✅' if results['latency_metrics']['mean_ms'] < 350 else '❌'} (Target: <350ms)
- **Median Response Time**: {results['latency_metrics']['median_ms']:.0f}ms
- **95th Percentile**: {results['latency_metrics']['p95_ms']:.0f}ms
- **99th Percentile**: {results['latency_metrics']['p99_ms']:.0f}ms

## 📈 Performance Analysis

"""
        
        # Add performance analysis
        if results['exact_match_accuracy'] >= 86:
            report += "✅ **Exact match accuracy target achieved!**\n\n"
        else:
            gap = 86 - results['exact_match_accuracy']
            report += f"⚠️ **Exact match accuracy is {gap:.2f}% below target.**\n\n"
        
        if results['latency_metrics']['mean_ms'] < 350:
            report += "✅ **Latency target achieved!**\n\n"
        else:
            report += "⚠️ **Latency exceeds target of 350ms.**\n\n"
        
        # Add examples
        report += "## 📝 Example Predictions\n\n"
        
        # Show 5 examples
        for i in range(min(5, len(self.results['predictions']))):
            pred = self.results['predictions'][i]
            ref = self.results['references'][i]
            exact_match = self.exact_match_score(pred, ref)
            
            report += f"### Example {i+1}\n"
            report += f"**Reference**: {ref[:200]}{'...' if len(ref) > 200 else ''}\n\n"
            report += f"**Prediction**: {pred[:200]}{'...' if len(pred) > 200 else ''}\n\n"
            report += f"**Exact Match**: {'✅' if exact_match else '❌'}\n\n"
            report += "---\n\n"
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Evaluation report saved to {save_path}")
        
        return report
    
    def run_benchmark(self, num_queries: int = 100) -> Dict:
        """
        Run performance benchmark with synthetic queries.
        
        Args:
            num_queries: Number of queries to run
            
        Returns:
            Benchmark results
        """
        logger.info(f"🏃 Running benchmark with {num_queries} queries...")
        
        # Generate synthetic queries
        test_queries = [
            f"What is the information about topic {i}?"
            for i in range(num_queries)
        ]
        
        latencies = []
        
        for query in tqdm(test_queries, desc="Benchmarking"):
            start_time = time.time()
            _ = self.rag_pipeline.query(query)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
        
        # Calculate statistics
        results = {
            'num_queries': num_queries,
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'queries_per_second': 1000 / np.mean(latencies)
        }
        
        # Get breakdown from RAG pipeline
        perf_stats = self.rag_pipeline.get_performance_stats()
        results['component_breakdown'] = perf_stats
        
        return results


def main():
    """Run evaluation and generate report."""
    evaluator = ModelEvaluator()
    
    # Run evaluation
    logger.info("🚀 Starting model evaluation...")
    results = evaluator.run_evaluation()
    
    # Generate report
    report = evaluator.generate_report(results)
    print("\n" + report)
    
    # Run benchmark
    logger.info("\n🏃 Running performance benchmark...")
    benchmark_results = evaluator.run_benchmark(num_queries=50)
    
    print("\n📊 Benchmark Results:")
    print(f"Mean latency: {benchmark_results['mean_latency_ms']:.0f}ms")
    print(f"Queries per second: {benchmark_results['queries_per_second']:.1f}")
    print(f"P95 latency: {benchmark_results['p95_latency_ms']:.0f}ms")
    print(f"P99 latency: {benchmark_results['p99_latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()