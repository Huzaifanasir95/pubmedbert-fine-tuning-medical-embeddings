# 🏥 PubMedBERT Fine-tuning for Medical Embeddings

Fine-tuning PubMedBERT to generate high-quality medical embeddings using the latest research from PubMed Central for improved semantic understanding of medical literature.

## 🎯 Project Overview

This project fine-tunes the PubMedBERT model using contrastive learning on recent medical literature (2020-2024) from PubMed Central to create specialized embeddings for:
- Medical literature semantic search
- Clinical decision support
- Drug discovery research
- Medical document classification
- Research trend analysis

## ✨ Features

- **Advanced Fine-tuning**: Contrastive learning and triplet loss strategies
- **Latest Medical Data**: Automated scraping from PubMed Central
- **Comprehensive Evaluation**: Intrinsic and extrinsic metrics on medical benchmarks
- **Interactive Demo**: Streamlit application for semantic search and visualization
- **Production Ready**: Optimized embeddings with <100ms inference time

## 🏗️ Repository Structure

```
pubmedbert-fine-tuning-medical-embeddings/
├── data/                       # Data directory
│   ├── raw/                   # Raw downloaded articles (JSONL)
│   ├── processed/             # Preprocessed training data (CSV)
│   └── sample_papers.csv      # Sample data for demo
├── src/                       # Source code
│   ├── data_collection/       # Data scraping scripts
│   ├── data_processing/       # Preprocessing pipeline
│   ├── training/              # Model training scripts
│   ├── evaluation/            # Evaluation frameworks
│   └── utils/                 # Utility functions
├── models/                    # Saved models
│   └── pubmedbert-medical-embeddings/
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_analysis.ipynb
├── app/                       # Demo application
│   └── streamlit_app.py
├── tests/                     # Unit tests
├── outputs/                   # Results and visualizations
│   ├── metrics/
│   ├── visualizations/
│   └── logs/
├── configs/                   # Configuration files
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pubmedbert-fine-tuning-medical-embeddings.git
cd pubmedbert-fine-tuning-medical-embeddings

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Data Collection
```bash
python src/data_collection/pubmed_scraper.py --query "cancer immunotherapy" --max-articles 5000
```

#### 2. Data Preprocessing
```bash
python src/data_processing/preprocessor.py --input data/raw/ --output data/processed/
```

#### 3. Model Training
```bash
python src/training/contrastive_trainer.py --config configs/training_config.yaml
```

#### 4. Evaluation
```bash
python src/evaluation/intrinsic_eval.py --model models/pubmedbert-medical-embeddings/
```

#### 5. Run Demo
```bash
streamlit run app/streamlit_app.py
```

## 📊 Results

### Performance Metrics
- **Spearman Correlation (BIOSSES)**: 0.XX
- **Classification Accuracy**: XX%
- **NDCG@10 (Semantic Search)**: 0.XX
- **Inference Time**: <100ms per document

### Comparison with Baselines
| Model | BIOSSES | MedSTS | Classification |
|-------|---------|--------|----------------|
| BERT-base | 0.XX | 0.XX | XX% |
| BioBERT | 0.XX | 0.XX | XX% |
| SciBERT | 0.XX | 0.XX | XX% |
| PubMedBERT (original) | 0.XX | 0.XX | XX% |
| **Ours (fine-tuned)** | **0.XX** | **0.XX** | **XX%** |

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- **Fine-tuning Strategy**: Contrastive learning with cosine similarity loss
- **Training Data**: 50K-100K recent PubMed Central articles (2020-2024)
- **Embedding Dimension**: 768

### Training Configuration
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 4
- **Warmup Steps**: 500
- **Loss Function**: Cosine Similarity Loss / Triplet Loss

## 📚 Applications

1. **Medical Literature Search**: Semantic search over millions of papers
2. **Clinical Decision Support**: Match patient cases to relevant research
3. **Drug Discovery**: Identify drug-target interactions
4. **Medical Education**: Personalized learning recommendations
5. **Research Analysis**: Trend detection and gap identification

## 🧪 Evaluation Benchmarks

- **BIOSSES**: Biomedical semantic similarity
- **MedSTS**: Medical semantic textual similarity
- **BLUE**: Biomedical Language Understanding Evaluation
- **PubMedQA**: Medical question answering
- **BC5CDR**: Chemical-disease relation extraction

## 📖 Documentation

For detailed documentation, see:
- [Project Walkthrough](docs/project_walkthrough.md)
- [Data Collection Guide](docs/data_collection.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PubMedBERT**: Gu et al. (2021)
- **Sentence-BERT**: Reimers & Gurevych (2019)
- **PubMed Central**: NCBI for providing open access to medical literature

## 📧 Contact

For questions or collaboration:
- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 📈 Roadmap

- [x] Initial project setup
- [ ] Data collection pipeline
- [ ] Preprocessing implementation
- [ ] Contrastive learning training
- [ ] Evaluation framework
- [ ] Streamlit demo
- [ ] Documentation
- [ ] Model deployment
- [ ] API development
- [ ] Research paper

---

**⭐ If you find this project useful, please consider giving it a star!**
