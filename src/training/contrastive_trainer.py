import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pandas as pd
from typing import List, Tuple
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class MedicalEmbeddingTrainer:
    """
    Trainer for fine-tuning PubMedBERT using contrastive learning.
    
    Usage:
        trainer = MedicalEmbeddingTrainer()
        train_examples, val_examples = trainer.load_data(
            train_file="data/processed/train.csv",
            val_file="data/processed/val.csv"
        )
        trainer.train(
            train_examples=train_examples,
            val_examples=val_examples,
            output_path="models/pubmedbert-medical-embeddings"
        )
    """
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        """
        Initialize trainer.
        
        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Model loaded successfully!")
    
    def load_data(self, train_file: str, val_file: str) -> Tuple[List[InputExample], List[InputExample]]:
        """
        Load training and validation data.
        
        Args:
            train_file: Path to training CSV
            val_file: Path to validation CSV
            
        Returns:
            Tuple of (train_examples, val_examples)
        """
        print(f"Loading training data from {train_file}...")
        train_df = pd.read_csv(train_file)
        
        print(f"Loading validation data from {val_file}...")
        val_df = pd.read_csv(val_file)
        
        train_examples = [
            InputExample(texts=[row['text1'], row['text2']], label=float(row['label']))
            for _, row in train_df.iterrows()
        ]
        
        val_examples = [
            InputExample(texts=[row['text1'], row['text2']], label=float(row['label']))
            for _, row in val_df.iterrows()
        ]
        
        print(f"Loaded {len(train_examples)} training examples")
        print(f"Loaded {len(val_examples)} validation examples")
        
        return train_examples, val_examples
    
    def train(self, 
              train_examples: List[InputExample], 
              val_examples: List[InputExample],
              output_path: str = "models/pubmedbert-finetuned",
              epochs: int = 4,
              batch_size: int = 16,
              warmup_steps: int = 500,
              use_wandb: bool = False):
        """
        Fine-tune model using contrastive learning.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            output_path: Path to save model
            epochs: Number of training epochs
            batch_size: Batch size
            warmup_steps: Number of warmup steps
            use_wandb: Whether to use Weights & Biases for logging
        """
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize W&B if requested
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(project="pubmedbert-finetuning", config={
                "model": self.model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "warmup_steps": warmup_steps,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples)
            })
        
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function (Cosine Similarity Loss for contrastive learning)
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Define evaluator
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples, 
            name='medical-embedding-eval'
        )
        
        print("\n" + "="*50)
        print("Starting training...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Training examples: {len(train_examples)}")
        print(f"Steps per epoch: {len(train_dataloader)}")
        print("="*50 + "\n")
        
        # Training
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            evaluation_steps=500,
            save_best_model=True,
            show_progress_bar=True
        )
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        print(f"\n✓ Training complete! Model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PubMedBERT with contrastive learning")
    parser.add_argument("--train", type=str, required=True, help="Training CSV file")
    parser.add_argument("--val", type=str, required=True, help="Validation CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--model", type=str, 
                       default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                       help="Base model name")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    trainer = MedicalEmbeddingTrainer(model_name=args.model)
    
    train_examples, val_examples = trainer.load_data(
        train_file=args.train,
        val_file=args.val
    )
    
    trainer.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        use_wandb=args.wandb
    )
