"""
CPU-optimized training script for PubMedBERT fine-tuning.
Uses smaller dataset for faster local training.
"""

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

def main():
    print("="*70)
    print("PubMedBERT Fine-tuning - CPU Training (Subset)")
    print("="*70)
    
    # Force CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Configuration
    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    TRAIN_FILE = "data/processed/train_small.csv"
    VAL_FILE = "data/processed/val_small.csv"
    OUTPUT_PATH = "models/pubmedbert-medical-embeddings"
    
    EPOCHS = 2  # Reduced for faster training
    BATCH_SIZE = 8  # Smaller batch for CPU
    WARMUP_STEPS = 100
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: CPU")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    # Load data
    print(f"\n{'='*70}")
    print("Loading data...")
    print(f"{'='*70}")
    
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    
    print(f"Train set: {len(train_df)} pairs")
    print(f"Validation set: {len(val_df)} pairs")
    print(f"Positive pairs (train): {len(train_df[train_df['label'] == 1])}")
    print(f"Negative pairs (train): {len(train_df[train_df['label'] == 0])}")
    
    # Create input examples
    print(f"\n{'='*70}")
    print("Creating training examples...")
    print(f"{'='*70}")
    
    train_examples = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
        train_examples.append(
            InputExample(texts=[row['text1'], row['text2']], label=float(row['label']))
        )
    
    val_examples = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val"):
        val_examples.append(
            InputExample(texts=[row['text1'], row['text2']], label=float(row['label']))
        )
    
    print(f"\nCreated {len(train_examples)} training examples")
    print(f"Created {len(val_examples)} validation examples")
    
    # Load model
    print(f"\n{'='*70}")
    print("Loading PubMedBERT model...")
    print(f"{'='*70}")
    
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model loaded successfully!")
    
    # Setup training
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)
    
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name='medical-embedding-eval'
    )
    
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * EPOCHS
    
    print(f"\nTraining setup:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    
    # Train
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        output_path=OUTPUT_PATH,
        evaluation_steps=50,  # Evaluate more frequently
        save_best_model=True,
        show_progress_bar=True
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n{'='*70}")
    print("✓ Training complete!")
    print(f"{'='*70}")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Model saved to: {OUTPUT_PATH}")
    
    # Quick test
    print(f"\n{'='*70}")
    print("Testing the model...")
    print(f"{'='*70}")
    
    test_texts = [
        "Checkpoint inhibitors have revolutionized cancer immunotherapy.",
        "PD-1 blockade shows promise in melanoma treatment.",
        "Diabetes requires careful blood glucose monitoring."
    ]
    
    embeddings = model.encode(test_texts)
    
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    
    print("\nSample similarities:")
    print(f"Text 1 vs Text 2 (related): {similarities[0][1]:.4f}")
    print(f"Text 1 vs Text 3 (unrelated): {similarities[0][2]:.4f}")
    print(f"Text 2 vs Text 3 (unrelated): {similarities[1][2]:.4f}")
    
    print(f"\n{'='*70}")
    print("✓ All done! Model is ready to use.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
