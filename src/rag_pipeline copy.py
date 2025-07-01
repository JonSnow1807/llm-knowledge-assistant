"""
RAG (Retrieval-Augmented Generation) pipeline implementation.
This module combines FAISS vector search with the fine-tuned Llama model.
"""

import os
import time
import pickle
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline combining:
    1. FAISS for efficient vector similarity search
    2. Fine-tuned Llama model for generation
    3. Optimized inference with caching
    """
    
    def __init__(self, 
                 model_path: str = "models/fine_tuned/final_model",
                 config_path: str = "configs/training_config.yaml"):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_path: Path to the fine-tuned model
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load components
        self._load_embedding_model()
        self._load_faiss_index()
        self._load_generation_model()
        
        # Performance tracking
        self.performance_metrics = {
            'retrieval_times': [],
            'generation_times': [],
            'total_times': []
        }
    
    def _load_embedding_model(self):
        """Load the sentence transformer for encoding queries."""
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            self.config['rag']['embedding_model']
        )
        self.embedding_model.to(self.device)
        logger.info("✓ Embedding model loaded")
    
    def _load_faiss_index(self):
        """Load the FAISS index and metadata."""
        logger.info("Loading FAISS index...")
        
        # Load FAISS index
        index_path = "data/embeddings/faiss_index.bin"
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Please run data_processing.py first!"
            )
        
        self.faiss_index = faiss.read_index(index_path)
        
        # Load chunk metadata
        metadata_path = "data/embeddings/chunk_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            self.chunk_metadata = pickle.load(f)
        
        logger.info(f"✓ FAISS index loaded with {self.faiss_index.ntotal} vectors")
    
    def _load_generation_model(self):
        """Load the fine-tuned Llama model for generation."""
        logger.info("Loading generation model...")
        
        # Check if model exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {self.model_path}. "
                "Please run fine_tuning.py first!"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_auth_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model_name = self.config['model']['name']
        
        # Determine if we need to load with quantization
        if self.config['model']['load_in_8bit']:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                use_auth_token=True,
                torch_dtype=torch.float16
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                use_auth_token=True,
                torch_dtype=torch.float16
            )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            base_model,
            self.model_path,
            device_map="auto"
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info("✓ Generation model loaded")
    
    def retrieve_documents(self, 
                          query: str, 
                          top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant documents using FAISS.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.config['rag']['top_k']
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Gather results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunk_metadata):
                chunk = self.chunk_metadata[idx]
                results.append({
                    'text': chunk['text'],
                    'score': float(dist),
                    'metadata': chunk
                })
        
        retrieval_time = time.time() - start_time
        self.performance_metrics['retrieval_times'].append(retrieval_time)
        
        logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.3f}s")
        
        return results
    
    def generate_response(self,
                         query: str,
                         retrieved_docs: List[Dict],
                         max_new_tokens: int = 200) -> str:
        """
        Generate a response using the fine-tuned model with retrieved context.
        
        Args:
            query: The user's question
            retrieved_docs: Retrieved documents for context
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        start_time = time.time()
        
        # Format context from retrieved documents
        context_texts = [doc['text'] for doc in retrieved_docs[:3]]  # Use top 3
        context = "\n\n".join(context_texts)
        
        # Create prompt with context
        prompt = f"""### Instruction:
Answer the following question based on the provided context. Be accurate and concise.

### Context:
{context}

### Question:
{query}

### Response:
"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['training']['max_seq_length']
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        generation_time = time.time() - start_time
        self.performance_metrics['generation_times'].append(generation_time)
        
        logger.info(f"Generated response in {generation_time:.3f}s")
        
        return response.strip()
    
    def query(self, 
              question: str, 
              top_k: Optional[int] = None,
              return_sources: bool = False) -> Dict:
        """
        End-to-end RAG query processing.
        
        Args:
            question: The user's question
            top_k: Number of documents to retrieve
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary containing answer and optionally sources
        """
        total_start = time.time()
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(question, top_k)
        
        # Step 2: Generate response
        answer = self.generate_response(question, retrieved_docs)
        
        # Calculate total time
        total_time = time.time() - total_start
        self.performance_metrics['total_times'].append(total_time)
        
        # Prepare response
        result = {
            'answer': answer,
            'response_time_ms': int(total_time * 1000),
            'retrieval_time_ms': int(self.performance_metrics['retrieval_times'][-1] * 1000),
            'generation_time_ms': int(self.performance_metrics['generation_times'][-1] * 1000)
        }
        
        if return_sources:
            result['sources'] = [
                {
                    'text': doc['text'],
                    'score': doc['score'],
                    'source': doc['metadata'].get('source', 'unknown')
                }
                for doc in retrieved_docs
            ]
        
        logger.info(f"Total query time: {total_time:.3f}s ({total_time*1000:.0f}ms)")
        
        return result
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        def calculate_stats(times):
            if not times:
                return {'mean': 0, 'median': 0, 'p95': 0, 'p99': 0}
            
            times_ms = [t * 1000 for t in times]
            return {
                'mean': np.mean(times_ms),
                'median': np.median(times_ms),
                'p95': np.percentile(times_ms, 95),
                'p99': np.percentile(times_ms, 99)
            }
        
        return {
            'retrieval': calculate_stats(self.performance_metrics['retrieval_times']),
            'generation': calculate_stats(self.performance_metrics['generation_times']),
            'total': calculate_stats(self.performance_metrics['total_times']),
            'num_queries': len(self.performance_metrics['total_times'])
        }
    
    def update_index(self, new_documents: List[Dict]):
        """
        Update the FAISS index with new documents.
        
        Args:
            new_documents: List of new documents to add
        """
        logger.info(f"Adding {len(new_documents)} new documents to index...")
        
        # Chunk new documents
        new_chunks = []
        new_metadata = []
        
        for doc in new_documents:
            chunks = self._chunk_document(doc['content'])
            for chunk in chunks:
                new_chunks.append(chunk['text'])
                new_metadata.append({
                    'text': chunk['text'],
                    'source': doc.get('source', 'unknown')
                })
        
        # Create embeddings
        embeddings = self.embedding_model.encode(
            new_chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.faiss_index.add(embeddings)
        self.chunk_metadata.extend(new_metadata)
        
        # Save updated index
        self._save_index()
        
        logger.info(f"✓ Index updated. Total vectors: {self.faiss_index.ntotal}")
    
    def _chunk_document(self, text: str) -> List[Dict]:
        """Chunk a document for indexing."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['rag']['chunk_size'],
            chunk_overlap=self.config['rag']['chunk_overlap'],
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        return [{'text': chunk} for chunk in chunks]
    
    def _save_index(self):
        """Save the FAISS index and metadata."""
        faiss.write_index(self.faiss_index, "data/embeddings/faiss_index.bin")
        with open("data/embeddings/chunk_metadata.pkl", "wb") as f:
            pickle.dump(self.chunk_metadata, f)


def main():
    """Demo the RAG pipeline."""
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Example queries
    test_queries = [
        "What is the main purpose of this system?",
        "How does the retrieval process work?",
        "What are the key performance metrics?"
    ]
    
    print("\n🔍 Testing RAG Pipeline")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuestion: {query}")
        result = rag.query(query, return_sources=True)
        print(f"Answer: {result['answer']}")
        print(f"Response time: {result['response_time_ms']}ms")
        print(f"  - Retrieval: {result['retrieval_time_ms']}ms")
        print(f"  - Generation: {result['generation_time_ms']}ms")
    
    # Print performance stats
    print("\n📊 Performance Statistics")
    print("=" * 50)
    stats = rag.get_performance_stats()
    
    for component, metrics in stats.items():
        if component != 'num_queries':
            print(f"\n{component.capitalize()}:")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value:.1f}ms")
    
    print(f"\nTotal queries processed: {stats['num_queries']}")


if __name__ == "__main__":
    main()