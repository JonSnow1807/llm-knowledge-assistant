#!/usr/bin/env python3
"""
Initial setup script for the LLM Knowledge Assistant project.
Run this first to set up your Lightning AI environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    print("📁 Creating project directory structure...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/embeddings",
        "src",
        "models/fine_tuned",
        "configs",
        "notebooks"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created {dir_path}")
    
    # Create __init__.py in src directory
    Path("src/__init__.py").touch()
    print("   ✓ Created src/__init__.py")

def check_gpu_availability():
    """Check if GPU is available and print information."""
    print("\n🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ Number of GPUs: {torch.cuda.device_count()}")
            print(f"   ✓ CUDA version: {torch.version.cuda}")
            
            # Check memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✓ GPU memory: {gpu_memory:.2f} GB")
            
            # Important: Check if we have enough memory for 8B model
            if gpu_memory < 16:
                print("\n   ⚠️  WARNING: You have less than 16GB GPU memory.")
                print("      You may need to use gradient checkpointing and smaller batch sizes.")
            
            return True
        else:
            print("   ❌ No GPU detected. Fine-tuning will be very slow on CPU.")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed yet. Install requirements first.")
        return False

def setup_huggingface_token():
    """Set up Hugging Face token for accessing gated models."""
    print("\n🔑 Setting up Hugging Face access...")
    
    # Check if token already exists
    hf_token_file = Path.home() / ".huggingface" / "token"
    if hf_token_file.exists():
        print("   ✓ Hugging Face token already configured")
        return
    
    print("\n   ℹ️  You need a Hugging Face token to access Llama models.")
    print("   1. Go to https://huggingface.co/settings/tokens")
    print("   2. Create a token with 'read' permissions")
    print("   3. Make sure you have access to meta-llama models")
    
    token = input("\n   Enter your Hugging Face token: ").strip()
    
    if token:
        # Create .env file
        with open(".env", "a") as f:
            f.write(f"\nHUGGINGFACE_TOKEN={token}\n")
        print("   ✓ Token saved to .env file")
        
        # Also login using huggingface-cli
        try:
            subprocess.run(["huggingface-cli", "login", "--token", token], 
                         check=True, capture_output=True)
            print("   ✓ Logged in to Hugging Face")
        except:
            print("   ⚠️  Could not login automatically. You may need to run:")
            print("      huggingface-cli login")

def create_training_config():
    """Create a default training configuration file."""
    print("\n📝 Creating training configuration...")
    
    config_content = """# Training Configuration for LLM Knowledge Assistant

# Model configuration
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  use_auth_token: true
  load_in_8bit: false  # Set to true if you have memory constraints
  device_map: "auto"

# LoRA configuration
lora:
  r: 16  # LoRA rank - higher values = more parameters but better quality
  lora_alpha: 32  # LoRA scaling parameter
  target_modules:  # Which layers to apply LoRA to
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"

# Training configuration
training:
  num_epochs: 3
  batch_size: 4  # Adjust based on your GPU memory
  gradient_accumulation_steps: 4  # Effective batch size = 4 * 4 = 16
  learning_rate: 2e-4
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500
  eval_steps: 100
  save_total_limit: 3
  fp16: true  # Mixed precision training
  gradient_checkpointing: true  # Save memory at the cost of speed
  max_seq_length: 512  # Maximum sequence length
  
# Dataset configuration
dataset:
  train_split: 0.9
  val_split: 0.1
  max_samples: null  # Set to a number to limit dataset size for testing
  
# RAG configuration
rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 500
  chunk_overlap: 50
  top_k: 5  # Number of documents to retrieve
  
# Evaluation configuration
evaluation:
  metrics: ["exact_match", "f1", "bleu"]
  
# Paths
paths:
  output_dir: "./models/fine_tuned"
  logs_dir: "./logs"
"""
    
    Path("configs").mkdir(exist_ok=True)
    with open("configs/training_config.yaml", "w") as f:
        f.write(config_content)
    
    print("   ✓ Created configs/training_config.yaml")

def create_example_data_script():
    """Create a script to help users prepare their data."""
    print("\n📄 Creating data preparation helper...")
    
    script_content = '''"""
Example script showing how to prepare your 5k documents for training.
Modify this based on your actual document format.
"""

import json
import os
from pathlib import Path

def prepare_training_data():
    """
    Convert your raw documents into the format needed for fine-tuning.
    
    Expected input format: Text files, JSON, CSV, or any structured format
    Output format: JSONL file with instruction-input-output pairs
    """
    
    # Example: If your documents are Q&A pairs or knowledge base articles
    training_data = []
    
    # MODIFY THIS SECTION based on your document format
    # Example 1: If you have text files with Q&A pairs
    raw_data_path = Path("data/raw")
    
    # Placeholder - replace with your actual data processing
    for i in range(10):  # This is just an example
        training_example = {
            "instruction": "Answer the following question based on the knowledge base:",
            "input": f"Example question {i}?",
            "output": f"Example answer {i} based on domain knowledge."
        }
        training_data.append(training_example)
    
    # Save in the format expected by the fine-tuning script
    output_path = Path("data/processed/training_data.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\\n")
    
    print(f"✓ Saved {len(training_data)} training examples to {output_path}")
    
    # Also create a small validation set
    val_data = training_data[:int(len(training_data) * 0.1)]
    with open("data/processed/validation_data.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\\n")
    
    print(f"✓ Saved {len(val_data)} validation examples")

if __name__ == "__main__":
    print("📊 Preparing training data...")
    print("⚠️  This is an example script. Modify it based on your actual data format!")
    prepare_training_data()
'''
    
    with open("prepare_data_example.py", "w") as f:
        f.write(script_content)
    
    print("   ✓ Created prepare_data_example.py")

def main():
    """Run all setup steps."""
    print("🚀 Setting up LLM Knowledge Assistant Project")
    print("=" * 50)
    
    # Step 1: Create directories
    create_directory_structure()
    
    # Step 2: Check GPU
    gpu_available = check_gpu_availability()
    
    # Step 3: Setup HF token
    setup_huggingface_token()
    
    # Step 4: Create config
    create_training_config()
    
    # Step 5: Create example data script
    create_example_data_script()
    
    print("\n✅ Initial setup complete!")
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Place your 5k documents in data/raw/")
    print("3. Modify prepare_data_example.py to process your specific data format")
    print("4. Run the data preparation script")
    print("5. Start fine-tuning!")
    
    if not gpu_available:
        print("\n⚠️  GPU not detected. Consider using Lightning AI's GPU instances.")

if __name__ == "__main__":
    main()