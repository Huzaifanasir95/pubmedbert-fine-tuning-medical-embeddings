import re
import json
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

class MedicalTextPreprocessor:
    """
    Preprocessor for medical text data.
    Creates training pairs for contrastive learning.
    
    Usage:
        preprocessor = MedicalTextPreprocessor()
        preprocessor.prepare_dataset(
            input_file="data/raw/articles.jsonl",
            output_dir="data/processed"
        )
    """
    
    def __init__(self, min_length: int = 50, max_length: int = 512):
        """
        Initialize preprocessor.
        
        Args:
            min_length: Minimum text length to keep
            max_length: Maximum text length (for tokenization)
        """
        self.min_length = min_length
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean medical text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove figure/table references
        text = re.sub(r'\(Fig\.?\s*\d+[A-Za-z]?\)', '', text)
        text = re.sub(r'\(Table\s*\d+\)', '', text)
        text = re.sub(r'Figure\s*\d+', '', text)
        
        # Remove citation markers
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        text = re.sub(r'\(\d+(?:,\s*\d+)*\)', '', text)
        
        # Normalize common abbreviations
        text = text.replace('e.g.', 'for example')
        text = text.replace('i.e.', 'that is')
        text = text.replace('et al.', 'and others')
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text meets minimum quality requirements."""
        if not text:
            return False
        if len(text) < self.min_length:
            return False
        # Check if text has enough alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.5:
            return False
        return True
    
    def create_training_pairs(self, articles: List[Dict]) -> List[Tuple[str, str, int]]:
        """
        Create positive and negative pairs for contrastive learning.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of (text1, text2, label) tuples
        """
        pairs = []
        
        print("Creating citation-based positive pairs...")
        # Create citation-based positive pairs
        pmcid_to_article = {article['pmcid']: article for article in articles}
        
        for article in tqdm(articles):
            anchor = self.clean_text(article['abstract'])
            if not self.is_valid_text(anchor):
                continue
            
            # Positive pairs from citations
            for cited_pmcid in article.get('citations', []):
                if cited_pmcid in pmcid_to_article:
                    cited_article = pmcid_to_article[cited_pmcid]
                    positive = self.clean_text(cited_article['abstract'])
                    if self.is_valid_text(positive):
                        pairs.append((anchor, positive, 1))  # Label 1 = similar
        
        print(f"Created {len(pairs)} citation-based pairs")
        
        print("Creating MeSH-based positive pairs...")
        # Create MeSH-based positive pairs
        mesh_groups = {}
        for article in articles:
            mesh_terms = article.get('mesh_terms', [])
            for mesh_term in mesh_terms:
                if mesh_term not in mesh_groups:
                    mesh_groups[mesh_term] = []
                mesh_groups[mesh_term].append(article)
        
        mesh_pairs = 0
        for mesh_term, group_articles in tqdm(mesh_groups.items()):
            if len(group_articles) >= 2:
                # Sample pairs from same MeSH group
                for i in range(min(len(group_articles), 10)):
                    for j in range(i+1, min(i+3, len(group_articles))):
                        text1 = self.clean_text(group_articles[i]['abstract'])
                        text2 = self.clean_text(group_articles[j]['abstract'])
                        if self.is_valid_text(text1) and self.is_valid_text(text2):
                            pairs.append((text1, text2, 1))
                            mesh_pairs += 1
        
        print(f"Created {mesh_pairs} MeSH-based pairs")
        
        print("Creating negative pairs...")
        # Create negative pairs (random sampling)
        num_negatives = len(pairs)
        valid_articles = [a for a in articles if self.is_valid_text(self.clean_text(a['abstract']))]
        
        for _ in tqdm(range(num_negatives)):
            if len(valid_articles) < 2:
                break
                
            idx1, idx2 = random.sample(range(len(valid_articles)), 2)
            article1 = valid_articles[idx1]
            article2 = valid_articles[idx2]
            
            text1 = self.clean_text(article1['abstract'])
            text2 = self.clean_text(article2['abstract'])
            
            # Check if they share MeSH terms (if so, skip to avoid false negatives)
            shared_mesh = set(article1.get('mesh_terms', [])) & set(article2.get('mesh_terms', []))
            if not shared_mesh and self.is_valid_text(text1) and self.is_valid_text(text2):
                pairs.append((text1, text2, 0))  # Label 0 = dissimilar
        
        print(f"Total pairs created: {len(pairs)}")
        return pairs
    
    def prepare_dataset(self, input_file: str, output_dir: str, 
                       test_size: float = 0.2, val_size: float = 0.1):
        """
        Prepare final training dataset.
        
        Args:
            input_file: Path to input JSONL file
            output_dir: Directory to save processed data
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Load articles
        print(f"Loading articles from {input_file}...")
        articles = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    articles.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(articles)} articles")
        
        # Create pairs
        pairs = self.create_training_pairs(articles)
        
        if len(pairs) == 0:
            print("ERROR: No valid pairs created!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(pairs, columns=['text1', 'text2', 'label'])
        
        print(f"\nDataset statistics:")
        print(f"Total pairs: {len(df)}")
        print(f"Positive pairs: {len(df[df['label'] == 1])}")
        print(f"Negative pairs: {len(df[df['label'] == 0])}")
        
        # Split into train/val/test
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        train_df, val_df = train_test_split(
            train_df, test_size=val_size, random_state=42, stratify=train_df['label']
        )
        
        # Save
        train_path = f"{output_dir}/train.csv"
        val_path = f"{output_dir}/val.csv"
        test_path = f"{output_dir}/test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\nDataset splits:")
        print(f"Train: {len(train_df)} pairs -> {train_path}")
        print(f"Val: {len(val_df)} pairs -> {val_path}")
        print(f"Test: {len(test_df)} pairs -> {test_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess medical text data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation set proportion")
    
    args = parser.parse_args()
    
    preprocessor = MedicalTextPreprocessor()
    preprocessor.prepare_dataset(
        input_file=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        val_size=args.val_size
    )
