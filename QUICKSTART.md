# Quick Start Guide

This guide will help you get started with the PubMedBERT fine-tuning project.

## Prerequisites

1. Python 3.8 or higher
2. GPU with 8GB+ VRAM (recommended, but CPU works too)
3. NCBI account (free) for PubMed API access

## Step 1: Environment Setup

```bash
# Navigate to project directory
cd pubmedbert-fine-tuning-medical-embeddings

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Data Collection

```bash
# Download medical articles from PubMed Central
python src/data_collection/pubmed_scraper.py \
    --email your.email@example.com \
    --query "cancer immunotherapy" \
    --output data/raw/cancer_immunotherapy.jsonl \
    --max-articles 5000
```

**Tips:**
- Start with 1000-5000 articles for prototyping
- Use specific queries for better results
- Get an NCBI API key for faster downloads (10 req/sec vs 3 req/sec)

## Step 3: Data Preprocessing

```bash
# Preprocess the downloaded articles
python src/data_processing/preprocessor.py \
    --input data/raw/cancer_immunotherapy.jsonl \
    --output data/processed
```

This will create:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## Step 4: Model Training

```bash
# Train the model
python src/training/contrastive_trainer.py \
    --train data/processed/train.csv \
    --val data/processed/val.csv \
    --output models/pubmedbert-medical-embeddings \
    --epochs 4 \
    --batch-size 16
```

**Training time estimates:**
- 5K articles: ~1-2 hours (GPU) / 6-8 hours (CPU)
- 50K articles: ~10-15 hours (GPU) / 2-3 days (CPU)

## Step 5: Run Demo Application

```bash
# Launch Streamlit demo
streamlit run app/streamlit_app.py
```

Open your browser to `http://localhost:8501`

## Common Issues

### Out of Memory (OOM)
- Reduce `--batch-size` to 8 or 4
- Use shorter sequences (modify `max_seq_length` in config)
- Use CPU instead of GPU for small experiments

### Slow Download
- Get NCBI API key: https://www.ncbi.nlm.nih.gov/account/
- Add `--api-key YOUR_KEY` to scraper command

### No GPU Available
- Training will work on CPU, just slower
- Consider using Google Colab (free GPU)
- Or reduce dataset size for faster iteration

## Next Steps

1. Experiment with different medical domains
2. Try different training strategies (triplet loss)
3. Evaluate on medical benchmarks (BIOSSES, MedSTS)
4. Build custom applications with the embeddings

## Getting Help

- Check the main README.md for detailed documentation
- Review the project walkthrough in the brain directory
- Open an issue on GitHub

Happy fine-tuning! 🚀
