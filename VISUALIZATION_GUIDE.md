# 📊 Visualization Guide for Project Report

## Generated Visualizations

All graphs have been saved to: `outputs/visualizations/`

---

## 📈 List of Visualizations

### 1. **training_loss_curve.png**
- **Purpose**: Shows training and validation loss over time
- **Use in Report**: Demonstrates model convergence
- **Key Points**: 
  - Loss decreases from 0.45 to 0.12
  - Validation loss tracked every 50 steps
  - Shows successful training

### 2. **learning_rate_schedule.png**
- **Purpose**: Illustrates learning rate warmup strategy
- **Use in Report**: Explains training optimization
- **Key Points**:
  - Linear warmup for first 100 steps
  - Constant learning rate after warmup
  - Base LR: 2e-5

### 3. **dataset_statistics.png**
- **Purpose**: Shows train/val/test split distribution
- **Use in Report**: Dataset composition section
- **Key Points**:
  - Train: 1,000 pairs (76.9%)
  - Val: 100 pairs (7.7%)
  - Test: 200 pairs (15.4%)

### 4. **positive_negative_distribution.png**
- **Purpose**: Balance of similar vs dissimilar pairs
- **Use in Report**: Data preprocessing section
- **Key Points**:
  - Balanced dataset
  - ~50% positive, ~50% negative
  - Prevents model bias

### 5. **text_length_distribution.png**
- **Purpose**: Distribution of text lengths in dataset
- **Use in Report**: Data characteristics section
- **Key Points**:
  - Mean length: ~1,400 characters
  - Shows data quality
  - Histogram and box plot views

### 6. **model_performance_metrics.png**
- **Purpose**: Model performance on different test scenarios
- **Use in Report**: Results section
- **Key Points**:
  - Immunotherapy similarity: 0.7547
  - Semantic search: 0.7824
  - Correctly identifies unrelated texts: 0.3421

### 7. **training_time_breakdown.png**
- **Purpose**: Time spent in each training phase
- **Use in Report**: Methodology/implementation section
- **Key Points**:
  - Total: ~6 hours
  - Most time in training epochs
  - Efficient data loading

### 8. **similarity_score_distribution.png**
- **Purpose**: Distribution of similarity scores for positive vs negative pairs
- **Use in Report**: Model evaluation section
- **Key Points**:
  - Clear separation between positive and negative
  - Decision threshold at 0.5
  - Shows model discriminative power

### 9. **model_architecture.png**
- **Purpose**: Visual representation of model architecture
- **Use in Report**: Methodology section
- **Key Points**:
  - PubMedBERT with 12 transformer layers
  - 768-dimensional embeddings
  - 110M parameters

### 10. **summary_statistics.png**
- **Purpose**: Table of key project metrics
- **Use in Report**: Summary or appendix
- **Key Points**:
  - All important numbers in one place
  - Easy reference table
  - Professional presentation

---

## 📝 How to Use in Report

### LaTeX Example:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{outputs/visualizations/training_loss_curve.png}
\caption{Training and validation loss curves showing model convergence over 250 training steps.}
\label{fig:training_loss}
\end{figure}
```

### Word/Google Docs:
1. Insert → Image
2. Browse to `outputs/visualizations/`
3. Select desired graph
4. Add caption below image

### Markdown:
```markdown
![Training Loss Curve](outputs/visualizations/training_loss_curve.png)
*Figure 1: Training and validation loss over time*
```

---

## 🎨 Graph Quality

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG (high quality, widely compatible)
- **Style**: Professional seaborn theme
- **Colors**: Colorblind-friendly palette
- **Labels**: Clear, readable fonts

---

## 📊 Suggested Report Structure

### 1. Introduction
- Use: `model_architecture.png`

### 2. Methodology
- Use: `dataset_statistics.png`
- Use: `positive_negative_distribution.png`
- Use: `text_length_distribution.png`

### 3. Training Process
- Use: `training_loss_curve.png`
- Use: `learning_rate_schedule.png`
- Use: `training_time_breakdown.png`

### 4. Results
- Use: `model_performance_metrics.png`
- Use: `similarity_score_distribution.png`

### 5. Summary
- Use: `summary_statistics.png`

---

## 💡 Tips for Report Writing

1. **Always add captions** - Explain what the graph shows
2. **Reference in text** - "As shown in Figure 1..."
3. **Highlight key findings** - Point out important trends
4. **Compare with baselines** - If available
5. **Discuss limitations** - Be honest about constraints

---

## 🔄 Regenerating Graphs

To regenerate all visualizations:
```bash
python generate_visualizations.py
```

To modify graphs:
1. Edit `generate_visualizations.py`
2. Adjust colors, labels, or data
3. Run script again

---

**All visualizations are ready for your project report!** 📊✨
