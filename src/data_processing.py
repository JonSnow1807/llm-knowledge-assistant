#!/usr/bin/env python3
"""
Memory-safe data processing for large datasets.
Processes data in chunks to avoid OOM kills.
"""

import os
import json
import gc
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Iterator

import faiss
import numpy as np
import pickle
import psutil
import torch
import yaml
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv

# Setup
load_dotenv()
if tok := os.getenv("HUGGINGFACE_TOKEN"):
    login(tok)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class MemorySafeProcessor:
    def __init__(self, cfg_path: str = "configs/training_config.yaml"):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        
        # Tokenizer
        log.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model"]["name"], 
            token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Force CPU for embedder to save GPU memory
        log.info("Loading embedder on CPU...")
        self.embedder = SentenceTransformer(
            self.cfg["rag"]["embedding_model"], 
            device="cpu"
        )
        
        self.chunk_size = self.cfg["rag"]["chunk_size"]
        self.max_seq_len = self.cfg["training"]["max_seq_length"]
    
    def process_jsonl_in_batches(self, filepath: Path, batch_size: int = 5000) -> Iterator[List[Dict]]:
        """Yield batches of documents from JSONL file."""
        batch = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    batch.append(json.loads(line))
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
        if batch:
            yield batch
    
    def create_qa_dataset(self):
        """Process Q&A pairs in batches and create dataset."""
        log.info("Creating Q&A dataset in batches...")
        
        qa_file = Path("data/raw/qa_pairs.jsonl")
        if not qa_file.exists():
            log.error(f"{qa_file} not found!")
            return None
        
        all_examples = []
        total_processed = 0
        
        # Process in batches
        for batch_num, batch in enumerate(self.process_jsonl_in_batches(qa_file, batch_size=5000)):
            log.info(f"Processing batch {batch_num + 1} ({len(batch)} docs)")
            
            # Create Q&A pairs
            for doc in batch:
                if "question" in doc and "answer" in doc:
                    q = doc["question"].strip()
                    a = doc["answer"].strip()
                    
                    if len(q) >= 20 and len(a) >= 20:
                        instruction = random.choice([
                            "Answer the following programming question:",
                            "Provide a solution to this coding problem:",
                            "Help solve this technical issue:",
                        ])
                        
                        # Format for training
                        text = f"### Instruction:\n{instruction}\n\n### Input:\n{q}\n\n### Response:\n{a}"
                        
                        # Tokenize immediately to check length
                        tokens = self.tokenizer(
                            text, 
                            truncation=True, 
                            max_length=self.max_seq_len,
                            padding="max_length"
                        )
                        
                        all_examples.append({
                            "input_ids": tokens["input_ids"],
                            "attention_mask": tokens["attention_mask"],
                            "labels": tokens["input_ids"].copy()
                        })
                        
                        total_processed += 1
            
            # Clear memory
            gc.collect()
            
            # Log memory usage
            mem = psutil.Process().memory_info().rss / 1e9
            log.info(f"Memory usage: {mem:.2f} GB")
            
            # Stop if we have enough examples
            if total_processed >= 50000:  # Limit for safety
                log.warning("Reached 50k examples limit")
                break
        
        log.info(f"Total examples created: {total_processed}")
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": [ex["input_ids"] for ex in all_examples],
            "attention_mask": [ex["attention_mask"] for ex in all_examples],
            "labels": [ex["labels"] for ex in all_examples]
        })
        
        # Split
        n_train = int(len(dataset) * 0.95)
        train_dataset = dataset.select(range(n_train))
        val_dataset = dataset.select(range(n_train, len(dataset)))
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    
    def build_faiss_index_streaming(self):
        """Build FAISS index with minimal memory usage."""
        log.info("Building FAISS index (ultra-low memory mode)...")
        
        qa_file = Path("data/raw/qa_pairs.jsonl")
        dim = self.embedder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        
        metadata = []
        buffer = []
        
        def flush_buffer():
            if not buffer:
                return
            
            # Embed in small batches
            texts = [b[0] for b in buffer]
            embeddings = self.embedder.encode(
                texts,
                batch_size=4,  # Very small batch
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Normalize and add to index
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            # Store metadata
            for _, meta in buffer:
                metadata.append(meta)
            
            # Clear
            buffer.clear()
            gc.collect()
        
        # Process file
        total_chunks = 0
        for batch_num, batch in enumerate(self.process_jsonl_in_batches(qa_file, batch_size=1000)):
            for doc in batch:
                if "question" in doc and "answer" in doc:
                    # Create searchable text
                    text = f"Question: {doc['question']}\nAnswer: {doc['answer']}"
                    
                    # Simple chunking
                    for i in range(0, len(text), self.chunk_size):
                        chunk = text[i:i + self.chunk_size].strip()
                        if chunk:
                            buffer.append((chunk, {"text": chunk, "source": "stackoverflow"}))
                            total_chunks += 1
                            
                            if len(buffer) >= 500:  # Small buffer
                                flush_buffer()
            
            # Log progress
            log.info(f"Processed batch {batch_num + 1}, total chunks: {total_chunks}")
        
        # Final flush
        flush_buffer()
        
        log.info(f"FAISS index complete: {index.ntotal} vectors")
        return index, metadata
    
    def run(self):
        """Run the complete pipeline."""
        log.info("Starting memory-safe processing...")
        
        # Check memory
        mem = psutil.virtual_memory()
        log.info(f"Available memory: {mem.available / 1e9:.1f} GB")
        
        # Step 1: Create dataset
        dataset = self.create_qa_dataset()
        if dataset is None:
            return
        
        # Save dataset
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk("data/processed/fine_tuning_dataset")
        log.info("Dataset saved")
        
        # Clear memory before FAISS
        del dataset
        gc.collect()
        
        # Step 2: Build FAISS index
        index, metadata = self.build_faiss_index_streaming()
        
        # Save FAISS
        Path("data/embeddings").mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, "data/embeddings/faiss_index.bin")
        with open("data/embeddings/chunk_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        log.info("✅ Processing complete!")


if __name__ == "__main__":
    processor = MemorySafeProcessor()
    processor.run()