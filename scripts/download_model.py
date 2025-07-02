#!/usr/bin/env python3
"""
Download the fine-tuned model from Hugging Face Hub.
This script downloads the model components needed for the RAG system.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, login
import shutil

def download_model():
    """Download the fine-tuned model from Hugging Face Hub."""
    
    print("🤗 LLM Knowledge Assistant - Model Download")
    print("=" * 50)
    
    # Model configuration
    model_repo = "chinmays18/llm-knowledge-assistant-8b"
    local_model_dir = Path("models/fine_tuned/final_model")
    
    print(f"📥 Downloading model: {model_repo}")
    print(f"📁 Local destination: {local_model_dir}")
    
    # Create directories
    local_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if HF token is needed
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(hf_token)
        print("✅ Authenticated with Hugging Face")
    
    # Files to download for a LoRA model
    files_to_download = [
        "adapter_config.json",
        "adapter_model.bin",  # or adapter_model.safetensors
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
        "generation_config.json"
    ]
    
    print("📦 Downloading model files...")
    
    try:
        for filename in files_to_download:
            try:
                print(f"   Downloading {filename}...")
                downloaded_file = hf_hub_download(
                    repo_id=model_repo,
                    filename=filename,
                    local_dir=local_model_dir,
                    local_dir_use_symlinks=False
                )
                print(f"   ✅ Downloaded {filename}")
            except Exception as e:
                print(f"   ⚠️  Could not download {filename}: {e}")
        
        print("🎉 Model download completed!")
        print(f"📁 Model available at: {local_model_dir}")
        
        # Verify download
        essential_files = ["adapter_config.json", "adapter_model.bin"]
        missing_files = [f for f in essential_files if not (local_model_dir / f).exists()]
        
        if missing_files:
            print(f"⚠️  Warning: Missing essential files: {missing_files}")
            print("   The model may not work correctly.")
        else:
            print("✅ All essential files downloaded successfully!")
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the model repository exists")
        print("3. Ensure you have access to the model (if private)")
        print("4. Set HUGGINGFACE_TOKEN if required")
        return False
    
    return True

def main():
    """Main download function."""
    success = download_model()
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Verify download: ls -la models/fine_tuned/final_model/")
        print("2. Test the model: python -c 'from src.rag_pipeline import RAGPipeline; rag = RAGPipeline(\"models/fine_tuned/final_model\")'")
        print("3. Start the API: python app.py")
    else:
        print("\n🔧 Alternative options:")
        print("1. Use your local trained model (if available)")
        print("2. Train the model first: python src/fine_tuning.py")
        print("3. Check the model repository URL")

if __name__ == "__main__":
    main()