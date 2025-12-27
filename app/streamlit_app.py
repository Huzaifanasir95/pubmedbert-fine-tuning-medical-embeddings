"""
Streamlit Demo App for PubMedBERT Medical Embeddings
Interactive interface for semantic search and similarity analysis
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(
    page_title="Medical Literature Embeddings",
    page_icon="🏥",
    layout="wide"
)

# Load model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer('models/pubmedbert-medical-embeddings')

# Title
st.title("🏥 PubMedBERT Medical Embeddings Explorer")
st.markdown("Fine-tuned embeddings for medical literature semantic search and analysis")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
**Model**: PubMedBERT fine-tuned with contrastive learning

**Training Data**: 1,000 medical literature pairs from PubMed Central (2020-2024)

**Use Cases**:
- Semantic search
- Document similarity
- Medical concept clustering
""")

# Load model
try:
    model = load_model()
    st.sidebar.success("✓ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 Semantic Search", "📊 Document Similarity", "🎨 Embedding Visualization"])

# Tab 1: Semantic Search
with tab1:
    st.header("Semantic Search in Medical Literature")
    
    query = st.text_area(
        "Enter your medical query:",
        "What are the latest treatments for lung cancer?",
        height=100
    )
    
    # Sample documents
    default_docs = """Recent advances in targeted therapy for non-small cell lung cancer include EGFR inhibitors.
Immunotherapy with checkpoint inhibitors has shown efficacy in advanced lung cancer.
Cardiovascular disease prevention focuses on lifestyle modifications and medication.
Combination chemotherapy remains a standard treatment for small cell lung cancer.
Type 2 diabetes treatment includes metformin as first-line therapy.
PD-1 blockade demonstrates promising results in melanoma patients.
Insulin therapy is essential for type 1 diabetes management."""
    
    documents_text = st.text_area(
        "Enter documents (one per line):",
        default_docs,
        height=200
    )
    
    if st.button("🔍 Search", type="primary"):
        documents = [doc.strip() for doc in documents_text.split('\n') if doc.strip()]
        
        if query and documents:
            with st.spinner("Generating embeddings..."):
                # Generate embeddings
                query_embedding = model.encode([query])
                doc_embeddings = model.encode(documents)
                
                # Calculate similarities
                scores = cosine_similarity(query_embedding, doc_embeddings)[0]
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Document': documents,
                    'Similarity Score': scores
                }).sort_values('Similarity Score', ascending=False)
                
                # Display results
                st.subheader("Search Results")
                
                for idx, row in results_df.iterrows():
                    score = row['Similarity Score']
                    color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"
                    
                    with st.expander(f"**Score: {score:.4f}** - {row['Document'][:80]}...", expanded=(idx==results_df.index[0])):
                        st.write(row['Document'])
                        st.progress(float(score))
                
                # Visualization
                fig = px.bar(
                    results_df,
                    x='Similarity Score',
                    y=results_df.index,
                    orientation='h',
                    title='Document Relevance Scores',
                    labels={'y': 'Document Index'}
                )
                st.plotly_chart(fig, use_container_width=True)

# Tab 2: Document Similarity
with tab2:
    st.header("Document Similarity Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text1 = st.text_area(
            "Document 1:",
            "Checkpoint inhibitors have revolutionized cancer immunotherapy treatment.",
            height=150,
            key="doc1"
        )
    
    with col2:
        text2 = st.text_area(
            "Document 2:",
            "PD-1 and CTLA-4 blockade shows promising results in melanoma patients.",
            height=150,
            key="doc2"
        )
    
    if st.button("Calculate Similarity", type="primary", key="sim_btn"):
        if text1 and text2:
            with st.spinner("Calculating..."):
                embeddings = model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                # Display metric
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric("Cosine Similarity", f"{similarity:.4f}")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=similarity * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Similarity Score (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                if similarity > 0.8:
                    st.success("✅ **Highly Similar** - Documents are very closely related")
                elif similarity > 0.6:
                    st.info("ℹ️ **Moderately Similar** - Documents share some common themes")
                else:
                    st.warning("⚠️ **Low Similarity** - Documents are largely unrelated")

# Tab 3: Embedding Visualization
with tab3:
    st.header("Embedding Space Visualization")
    
    # Sample texts
    sample_texts = st.text_area(
        "Enter texts to visualize (one per line):",
        """Checkpoint inhibitors for cancer treatment
PD-1 blockade in melanoma therapy
Diabetes management with metformin
Insulin therapy for type 1 diabetes
Cardiovascular disease prevention
Hypertension treatment with ACE inhibitors
Chemotherapy for breast cancer
Radiation therapy in oncology""",
        height=200
    )
    
    if st.button("Generate Visualization", type="primary", key="viz_btn"):
        texts = [t.strip() for t in sample_texts.split('\n') if t.strip()]
        
        if len(texts) >= 3:
            with st.spinner("Generating embeddings and reducing dimensions..."):
                from sklearn.manifold import TSNE
                
                # Generate embeddings
                embeddings = model.encode(texts)
                
                # Reduce to 2D using t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(texts)-1))
                embeddings_2d = tsne.fit_transform(embeddings)
                
                # Create DataFrame
                viz_df = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'text': texts,
                    'label': [f"Text {i+1}" for i in range(len(texts))]
                })
                
                # Plot
                fig = px.scatter(
                    viz_df,
                    x='x',
                    y='y',
                    text='label',
                    hover_data=['text'],
                    title="Medical Text Embedding Space (t-SNE)",
                    width=800,
                    height=600
                )
                
                fig.update_traces(textposition='top center', marker=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)
                
                # Similarity heatmap
                st.subheader("Similarity Heatmap")
                similarities = cosine_similarity(embeddings)
                
                fig_heatmap = px.imshow(
                    similarities,
                    labels=dict(x="Text", y="Text", color="Similarity"),
                    x=[f"T{i+1}" for i in range(len(texts))],
                    y=[f"T{i+1}" for i in range(len(texts))],
                    color_continuous_scale="RdYlGn",
                    aspect="auto"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Please enter at least 3 texts for visualization")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with PubMedBERT fine-tuned on medical literature from PubMed Central</p>
    <p>Model: <code>microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext</code></p>
</div>
""", unsafe_allow_html=True)
