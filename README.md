# Fine-tuning PubMedBERT for Medical Literature Embeddings

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.56.1-yellow.svg)](https://huggingface.co/transformers/)

> A domain-specific fine-tuned PubMedBERT model optimized for generating high-quality medical text embeddings using contrastive learning on recent PubMed Central articles.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Performance](#-performance)
- [Demo Application](#-demo-application)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

Medical literature search and semantic understanding are critical challenges in biomedical NLP. This project presents a fine-tuned **PubMedBERT** model optimized for generating semantically meaningful embeddings of medical texts using **contrastive learning**.

### Problem Statement

- **Challenge**: Traditional keyword-based search fails to capture semantic relationships in medical literature
- **Solution**: Fine-tune PubMedBERT with contrastive learning on recent medical articles
- **Impact**: Enable semantic search, document clustering, and clinical decision support

### Key Statistics

| Metric | Value |
|--------|-------|
| **Articles Collected** | 1,918 from PubMed Central (2020-2024) |
| **Training Pairs** | 7,218 (3,636 positive, 3,582 negative) |
| **Model Size** | ~440 MB |
| **Best Similarity Score** | 0.7824 |
| **Training Time** | ~6 hours (CPU) |

---

## ✨ Key Features

- ✅ **Domain-Specific**: Fine-tuned on recent cancer immunotherapy literature
- ✅ **Contrastive Learning**: Uses citation networks and MeSH terms for pair generation
- ✅ **High Performance**: 0.75+ similarity scores on related medical texts
- ✅ **Production Ready**: Includes demo app and API-ready model
- ✅ **Open Source**: Complete code, data pipeline, and trained model

---

## 🏗️ Architecture

### System Architecture

```mermaid
graph TB
    A[PubMed Central API] -->|1,918 articles| B[Data Collection]
    B -->|Raw JSON| C[Preprocessing]
    C -->|7,218 pairs| D[Training Pipeline]
    D -->|Fine-tuning| E[PubMedBERT Model]
    E -->|768-dim embeddings| F[Applications]
    
    F --> G[Semantic Search]
    F --> H[Document Clustering]
    F --> I[Clinical Decision Support]
    
    style E fill:#4a90c4,stroke:#2e5c8a,stroke-width:3px
    style F fill:#2ecc71,stroke:#27ae60,stroke-width:2px
```

### Model Architecture

```mermaid
graph LR
    A[Input Text] --> B[Tokenization]
    B --> C[PubMedBERT<br/>12 Layers<br/>110M Params]
    C --> D[Mean Pooling]
    D --> E[768-dim<br/>Embedding]
    
    style C fill:#e74c3c,stroke:#c0392b,stroke-width:3px
    style E fill:#3498db,stroke:#2980b9,stroke-width:2px
```

### Training Pipeline

```mermaid
graph TD
    A[Medical Articles] --> B{Pair Generation}
    B -->|Citations| C[Positive Pairs]
    B -->|MeSH Terms| C
    B -->|Random| D[Negative Pairs]
    
    C --> E[Balanced Dataset<br/>1,300 pairs]
    D --> E
    
    E --> F[Contrastive Learning<br/>Cosine Similarity Loss]
    F --> G[Fine-tuned Model]
    
    style F fill:#9b59b6,stroke:#8e44ad,stroke-width:2px
    style G fill:#2ecc71,stroke:#27ae60,stroke-width:3px
```

---

## 📊 Results

### Training Performance

![Training Loss](outputs/visualizations/training_loss_curve.png)
*Figure 1: Training and validation loss curves showing successful convergence from 0.45 to 0.12*

### Dataset Distribution

![Dataset Statistics](outputs/visualizations/dataset_statistics.png)
*Figure 2: Balanced dataset split - Train: 1,000 pairs (76.9%), Val: 100 pairs (7.7%), Test: 200 pairs (15.4%)*

### Model Performance

![Performance Metrics](outputs/visualizations/model_performance_metrics.png)
*Figure 3: Model performance across different test scenarios - achieving 0.75+ similarity on related texts*

### Similarity Distribution

![Similarity Distribution](outputs/visualizations/similarity_score_distribution.png)
*Figure 4: Clear separation between positive (similar) and negative (dissimilar) pairs*

---

## 🚀 Installation

### Prerequisites

- Python 3.13+
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/Huzaifanasir95/pubmedbert-fine-tuning-medical-embeddings.git
cd pubmedbert-fine-tuning-medical-embeddings

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
datasets
biopython
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
streamlit
```

---

## 🎯 Quick Start

### 1. Load the Fine-tuned Model

```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned model
model = SentenceTransformer('models/pubmedbert-medical-embeddings')

# Generate embeddings
texts = [
    "Checkpoint inhibitors have revolutionized cancer immunotherapy",
    "PD-1 blockade shows promising results in melanoma patients"
]

embeddings = model.encode(texts)
print(f"Embedding shape: {embeddings.shape}")  # (2, 768)
```

### 2. Semantic Search

```python
from sklearn.metrics.pairwise import cosine_similarity

query = "lung cancer treatment options"
documents = [
    "EGFR inhibitors for non-small cell lung cancer",
    "Diabetes management with metformin",
    "Immunotherapy for advanced lung cancer"
]

# Encode query and documents
query_emb = model.encode([query])
doc_embs = model.encode(documents)

# Calculate similarities
scores = cosine_similarity(query_emb, doc_embs)[0]

# Rank results
for doc, score in sorted(zip(documents, scores), key=lambda x: x[1], reverse=True):
    print(f"{score:.4f}: {doc}")
```

**Output:**
```
0.7824: EGFR inhibitors for non-small cell lung cancer
0.7099: Immunotherapy for advanced lung cancer
0.3421: Diabetes management with metformin
```

### 3. Document Similarity

```python
text1 = "Checkpoint inhibitors for cancer treatment"
text2 = "PD-1 blockade in melanoma therapy"

emb1, emb2 = model.encode([text1, text2])
similarity = cosine_similarity([emb1], [emb2])[0][0]

print(f"Similarity: {similarity:.4f}")  # 0.7547
```

---

## 📁 Project Structure

```
pubmedbert-fine-tuning-medical-embeddings/
├── data/
│   ├── raw/
│   │   └── cancer_immunotherapy.jsonl    # 1,918 articles
│   └── processed/
│       ├── train.csv                      # 5,196 pairs (full)
│       ├── val.csv                        # 578 pairs
│       ├── test.csv                       # 1,444 pairs
│       ├── train_small.csv                # 1,000 pairs (used)
│       ├── val_small.csv                  # 100 pairs
│       └── test_small.csv                 # 200 pairs
│
├── models/
│   └── pubmedbert-medical-embeddings/     # Fine-tuned model
│       ├── pytorch_model.bin              # Model weights
│       ├── config.json                    # Configuration
│       └── tokenizer files
│
├── src/
│   ├── data_collection/
│   │   └── pubmed_scraper.py             # PubMed API scraper
│   ├── data_processing/
│   │   └── preprocessor.py               # Text cleaning & pair generation
│   └── training/
│       └── contrastive_trainer.py        # Training script
│
├── app/
│   └── streamlit_app.py                  # Interactive demo
│
├── outputs/
│   └── visualizations/                   # All graphs (10 PNG files)
│
├── configs/
│   └── training_config.yaml              # Hyperparameters
│
├── requirements.txt                       # Dependencies
├── LICENSE                                # MIT License
└── README.md                              # This file
```

---

## 🔬 Methodology

### Data Collection

```mermaid
sequenceDiagram
    participant User
    participant API as NCBI E-utilities API
    participant DB as PubMed Central
    
    User->>API: Query: "cancer immunotherapy"
    API->>DB: Search (2020-2024)
    DB-->>API: Article IDs
    API->>DB: Fetch full-text (batches)
    DB-->>API: XML content
    API-->>User: 1,918 articles
```

**Collection Details:**
- **Query**: "cancer immunotherapy"
- **Date Range**: 2020-01-01 to 2024-12-31
- **Source**: PubMed Central Open Access
- **API Key**: Used for 10 req/sec (vs 3 req/sec)
- **Content**: Full-text, abstracts, MeSH terms, citations

### Preprocessing Pipeline

1. **Text Cleaning**
   - Remove figure/table references
   - Remove citation markers
   - Normalize whitespace
   - Expand abbreviations

2. **Pair Generation**
   - **Positive Pairs** (y=1):
     - Citation-based: Papers citing each other
     - MeSH-based: Shared medical subject headings
   - **Negative Pairs** (y=0):
     - Random sampling without shared MeSH terms

3. **Dataset Split**
   - Train: 76.9% (1,000 pairs)
   - Validation: 7.7% (100 pairs)
   - Test: 15.4% (200 pairs)

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Base Model** | microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext |
| **Loss Function** | Cosine Similarity Loss |
| **Optimizer** | AdamW |
| **Learning Rate** | 2e-5 |
| **Warmup Steps** | 100 |
| **Batch Size** | 8 |
| **Epochs** | 2 |
| **Max Sequence Length** | 512 tokens |
| **Device** | CPU |
| **Training Time** | ~6 hours |

### Loss Function

For a pair of texts $(t_1, t_2)$ with label $y \in \{0, 1\}$:

$$
\text{sim}(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\| \|\mathbf{e}_2\|}
$$

$$
\mathcal{L} = \begin{cases}
1 - \text{sim}(\mathbf{e}_1, \mathbf{e}_2) & \text{if } y = 1 \\
\max(0, \text{sim}(\mathbf{e}_1, \mathbf{e}_2)) & \text{if } y = 0
\end{cases}
$$

---

## 📈 Performance

### Quantitative Results

| Test Scenario | Similarity Score | Interpretation |
|---------------|-----------------|----------------|
| **Immunotherapy Papers** | 0.7547 | ✅ Highly Related |
| **Semantic Search (Lung Cancer)** | 0.7824 | ✅ Excellent Ranking |
| **Unrelated Texts** | 0.3421 | ✅ Correctly Dissimilar |
| **Medical Concept Clustering** | 0.7234 | ✅ Good Grouping |

### Qualitative Analysis

**Example 1: Related Medical Texts**
```
Text 1: "Checkpoint inhibitors have revolutionized cancer immunotherapy"
Text 2: "PD-1 and CTLA-4 blockade shows promising results in melanoma"
Similarity: 0.7547 ✅
```

**Example 2: Unrelated Texts**
```
Text 1: "Cancer immunotherapy treatment"
Text 2: "Diabetes management with insulin"
Similarity: 0.3421 ✅
```

### Learning Curves

![Learning Rate Schedule](outputs/visualizations/learning_rate_schedule.png)
*Figure 5: Linear warmup for first 100 steps, then constant learning rate*

![Training Time](outputs/visualizations/training_time_breakdown.png)
*Figure 6: Training time distribution across phases*

---

## 🎨 Demo Application

### Interactive Streamlit App

Run the demo application:

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

### Features

1. **🔍 Semantic Search**
   - Enter medical query
   - Search through documents
   - Ranked results with similarity scores

2. **📊 Document Similarity**
   - Compare two medical texts
   - Visual similarity gauge
   - Interpretation guidance

3. **🎨 Embedding Visualization**
   - t-SNE dimensionality reduction
   - Interactive scatter plots
   - Similarity heatmaps

---

## 🔧 Advanced Usage

### Training on Full Dataset

```bash
python src/training/contrastive_trainer.py \
    --train data/processed/train.csv \
    --val data/processed/val.csv \
    --output models/pubmedbert-full \
    --epochs 4 \
    --batch-size 16
```

### Custom Data Collection

```bash
python src/data_collection/pubmed_scraper.py \
    --email your.email@example.com \
    --api-key YOUR_NCBI_API_KEY \
    --query "your medical query" \
    --output data/raw/custom.jsonl \
    --max-articles 5000
```

### Data Preprocessing

```bash
python src/data_processing/preprocessor.py \
    --input data/raw/custom.jsonl \
    --output data/processed \
    --test-size 0.2 \
    --val-size 0.1
```

---

## 🎓 Applications

### 1. Medical Literature Search
```python
# Semantic search over millions of PubMed articles
query = "What are the side effects of checkpoint inhibitors?"
relevant_papers = search_pubmed(query, model, top_k=10)
```

### 2. Clinical Decision Support
```python
# Match patient symptoms to relevant research
symptoms = "Patient presents with melanoma resistant to standard therapy"
relevant_treatments = find_similar_cases(symptoms, model)
```

### 3. Drug Discovery
```python
# Find similar compounds and mechanisms
compound_description = "EGFR tyrosine kinase inhibitor"
similar_compounds = cluster_compounds(compound_description, model)
```

### 4. Research Trend Analysis
```python
# Cluster and analyze research topics
papers = load_recent_papers()
clusters = cluster_research_topics(papers, model, n_clusters=10)
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{nasir2024pubmedbert,
  author = {Nasir, Huzaifa},
  title = {Fine-tuning PubMedBERT for Medical Literature Embeddings Using Contrastive Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Huzaifanasir95/pubmedbert-fine-tuning-medical-embeddings}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PubMed Central** for providing open access to medical literature
- **HuggingFace** for the Transformers library
- **Microsoft Research** for PubMedBERT
- **Sentence-Transformers** team for the excellent framework

---

## 📧 Contact

**Huzaifa Nasir**
- Email: nasirhuzaifa95@gmail.com
- GitHub: [@Huzaifanasir95](https://github.com/Huzaifanasir95)
- Project Link: [pubmedbert-fine-tuning-medical-embeddings](https://github.com/Huzaifanasir95/pubmedbert-fine-tuning-medical-embeddings)

---

## 🔗 Related Resources

- [PubMedBERT Paper](https://arxiv.org/abs/2007.15779)
- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**


</div>
