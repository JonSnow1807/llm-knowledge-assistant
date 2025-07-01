"""
Fine-tuning module implementing LoRA (Low-Rank Adaptation) for Llama models.
This module handles the complete fine-tuning pipeline with memory optimization.
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime
import wandb
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_from_disk
import evaluate

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class LlamaFineTuner:
    """
    Fine-tune Llama models using LoRA (Low-Rank Adaptation).
    
    This class implements:
    1. Efficient model loading with optional quantization
    2. LoRA configuration and setup
    3. Training with gradient accumulation and mixed precision
    4. Evaluation and checkpointing
    """
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """Initialize the fine-tuner with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['name']
        self.output_dir = self.config['paths']['output_dir']
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb for experiment tracking (optional but recommended)
        self.use_wandb = os.getenv("WANDB_API_KEY") is not None
        if self.use_wandb:
            wandb.init(
                project="llm-knowledge-assistant",
                name=f"lora-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
    
    def setup_model_and_tokenizer(self):
        """
        Load the base model and tokenizer with optimizations.
        
        This method handles:
        - Loading large models efficiently
        - Optional 8-bit quantization for memory savings
        - Preparing model for LoRA training
        """
        logger.info(f"🔄 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=True,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Configure model loading based on available resources
        if self.config['model']['load_in_8bit']:
            # 8-bit quantization for memory efficiency
            logger.info("📊 Loading model in 8-bit precision for memory efficiency")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                
                token=True,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # Full precision loading
            logger.info("📊 Loading model in full precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                
                token=True,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.config['training']['fp16'] else torch.float32
            )
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")
        
        # Disable cache for training
        self.model.config.use_cache = False
        
        logger.info(f"✓ Model loaded successfully")
        self._log_model_info()
    
    def _log_model_info(self):
        """Log model information for debugging."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model Statistics:")
        logger.info(f"   - Total parameters: {total_params:,}")
        logger.info(f"   - Trainable parameters: {trainable_params:,}")
        logger.info(f"   - Memory footprint: ~{total_params * 2 / 1e9:.2f} GB (fp16)")
    
    def setup_lora(self):
        """
        Configure and apply LoRA to the model.
        
        LoRA (Low-Rank Adaptation) adds small trainable matrices to specific layers,
        allowing efficient fine-tuning with minimal parameters.
        """
        logger.info("🔧 Configuring LoRA...")
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.config['lora']['r'],  # Rank - controls capacity
            lora_alpha=self.config['lora']['lora_alpha'],  # Scaling factor
            target_modules=self.config['lora']['target_modules'],  # Which layers to adapt
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Log LoRA information
        self.model.print_trainable_parameters()
        
        logger.info("✓ LoRA configuration applied")
    
    def load_datasets(self):
        """Load the preprocessed datasets."""
        logger.info("📁 Loading datasets...")
        
        dataset_path = "data/processed/fine_tuning_dataset"
        try:
            self.dataset = load_from_disk(dataset_path)
            logger.info(f"✓ Loaded dataset from {dataset_path}")
            logger.info(f"   - Training samples: {len(self.dataset['train'])}")
            logger.info(f"   - Validation samples: {len(self.dataset['validation'])}")
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            logger.error("Please run data_processing.py first to prepare the datasets!")
            raise
    
    def create_training_arguments(self):
        """
        Create training arguments for the Trainer.
        
        This configures all aspects of training including:
        - Batch sizes and gradient accumulation
        - Learning rate and scheduling
        - Evaluation and saving strategies
        """
        # Calculate total training steps
        total_steps = (
            len(self.dataset['train']) // 
            (self.config['training']['batch_size'] * 
             self.config['training']['gradient_accumulation_steps']) * 
            self.config['training']['num_epochs']
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=self.config['training']['learning_rate'],
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            fp16=self.config['training']['fp16'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            report_to=["wandb"] if self.use_wandb else ["none"],
            logging_dir=f"{self.output_dir}/logs",
            dataloader_num_workers=4,
            remove_unused_columns=False,
            group_by_length=True,  # Groups sequences of similar length for efficiency
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        )
        
        logger.info(f"📊 Training Configuration:")
        logger.info(f"   - Total training steps: {total_steps}")
        logger.info(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        logger.info(f"   - Learning rate: {training_args.learning_rate}")
        
        return training_args
    
    def setup_trainer(self, training_args):
        """
        Set up the Hugging Face Trainer with custom callbacks and data collator.
        """
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8  # Efficient padding for tensor cores
        )
        
        # Custom compute metrics function
        def compute_metrics(eval_preds):
            """Compute perplexity for evaluation."""
            predictions, labels = eval_preds
            
            # Calculate perplexity
            loss = predictions.mean()
            perplexity = torch.exp(torch.tensor(loss))
            
            return {
                "perplexity": perplexity.item(),
                "loss": loss
            }
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if self.config['training']['eval_steps'] > 0 else None,
        )
        
        logger.info("✓ Trainer initialized")
    
    def train(self):
        """
        Execute the training process.
        
        This method:
        1. Runs the training loop
        2. Handles checkpointing
        3. Saves the final model
        """
        logger.info("🚀 Starting training...")
        
        try:
            # Start training
            # Check for existing checkpoint
            import os
            checkpoint_dir = self.output_dir
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                    resume_from = os.path.join(checkpoint_dir, latest_checkpoint)
                    logger.info(f"📂 Resuming from checkpoint: {resume_from}")
                    train_result = self.trainer.train(resume_from_checkpoint=resume_from)
                else:
                    train_result = self.trainer.train()
            else:
                train_result = self.trainer.train()
            
            
            # Log final metrics
            logger.info("✅ Training completed!")
            logger.info(f"📊 Final training loss: {train_result.training_loss:.4f}")
            
            # Save the final model
            self.save_model()
            
            # Log metrics to wandb if available
            if self.use_wandb:
                wandb.log({
                    "final_train_loss": train_result.training_loss,
                    "total_train_steps": train_result.global_step
                })
            
            return train_result
            
        except KeyboardInterrupt:
            logger.info("⚠️  Training interrupted by user")
            self.save_model(interrupted=True)
            raise
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def save_model(self, interrupted: bool = False):
        """
        Save the fine-tuned model and tokenizer.
        
        Args:
            interrupted: Whether saving due to interruption
        """
        save_path = f"{self.output_dir}/final_model"
        if interrupted:
            save_path = f"{self.output_dir}/interrupted_checkpoint"
        
        logger.info(f"💾 Saving model to {save_path}")
        
        # Save model
        self.trainer.save_model(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save LoRA config for reference
        lora_config_path = f"{save_path}/lora_config.yaml"
        with open(lora_config_path, 'w') as f:
            yaml.dump(self.config['lora'], f)
        
        logger.info("✓ Model saved successfully")
    
    def run_full_pipeline(self):
        """
        Execute the complete fine-tuning pipeline.
        
        This is the main entry point that orchestrates all steps.
        """
        logger.info("=" * 50)
        logger.info("🎯 LLM Knowledge Assistant Fine-Tuning Pipeline")
        logger.info("=" * 50)
        
        # Step 1: Load datasets
        self.load_datasets()
        
        # Step 2: Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Step 3: Apply LoRA
        self.setup_lora()
        
        # Step 4: Create training arguments
        training_args = self.create_training_arguments()
        
        # Step 5: Setup trainer
        self.setup_trainer(training_args)
        
        # Step 6: Train
        self.train()
        
        logger.info("\n🎉 Fine-tuning pipeline completed successfully!")
        logger.info(f"📁 Model saved to: {self.output_dir}/final_model")
        
        # Cleanup
        if self.use_wandb:
            wandb.finish()


def main():
    """Run the fine-tuning pipeline."""
    # Check for GPU
    if not torch.cuda.is_available():
        logger.warning("⚠️  No GPU detected! Training will be extremely slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run fine-tuning
    fine_tuner = LlamaFineTuner()
    fine_tuner.run_full_pipeline()


if __name__ == "__main__":
    main()