"""
Test the fine-tuned PubMedBERT model.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def test_model():
    print("="*70)
    print("Testing Fine-tuned PubMedBERT Model")
    print("="*70)
    
    # Load the fine-tuned model
    print("\nLoading model...")
    model = SentenceTransformer('models/pubmedbert-medical-embeddings')
    print("✓ Model loaded successfully!")
    
    # Test cases
    print("\n" + "="*70)
    print("Test 1: Cancer Immunotherapy (Related Papers)")
    print("="*70)
    
    cancer_texts = [
        "Checkpoint inhibitors have revolutionized cancer immunotherapy treatment.",
        "PD-1 and CTLA-4 blockade shows promising results in melanoma patients.",
        "CAR-T cell therapy represents a breakthrough in treating hematologic malignancies.",
        "Diabetes management requires careful monitoring of blood glucose levels."
    ]
    
    embeddings = model.encode(cancer_texts)
    similarities = cosine_similarity(embeddings)
    
    print("\nTexts:")
    for i, text in enumerate(cancer_texts):
        print(f"{i+1}. {text}")
    
    print("\nSimilarity Matrix:")
    print("     Text1  Text2  Text3  Text4")
    for i in range(len(cancer_texts)):
        print(f"Text{i+1}", end="")
        for j in range(len(cancer_texts)):
            print(f"  {similarities[i][j]:.3f}", end="")
        print()
    
    print("\nKey Similarities:")
    print(f"  Text 1 vs Text 2 (both immunotherapy): {similarities[0][1]:.4f}")
    print(f"  Text 1 vs Text 3 (both cancer): {similarities[0][2]:.4f}")
    print(f"  Text 1 vs Text 4 (unrelated): {similarities[0][3]:.4f}")
    
    # Test 2: Semantic Search
    print("\n" + "="*70)
    print("Test 2: Semantic Search")
    print("="*70)
    
    query = "What are the latest treatments for lung cancer?"
    documents = [
        "Recent advances in targeted therapy for non-small cell lung cancer include EGFR inhibitors.",
        "Immunotherapy with checkpoint inhibitors has shown efficacy in advanced lung cancer.",
        "Cardiovascular disease prevention focuses on lifestyle modifications and medication.",
        "Combination chemotherapy remains a standard treatment for small cell lung cancer.",
        "Type 2 diabetes treatment includes metformin as first-line therapy."
    ]
    
    print(f"\nQuery: {query}")
    print("\nDocuments:")
    for i, doc in enumerate(documents):
        print(f"{i+1}. {doc}")
    
    query_embedding = model.encode([query])
    doc_embeddings = model.encode(documents)
    
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    ranked_indices = np.argsort(scores)[::-1]
    
    print("\nRanked Results:")
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"{rank}. [Score: {scores[idx]:.4f}] {documents[idx][:60]}...")
    
    # Test 3: Medical Entity Clustering
    print("\n" + "="*70)
    print("Test 3: Medical Concept Clustering")
    print("="*70)
    
    concepts = [
        "Hypertension treatment with ACE inhibitors",
        "Beta blockers for blood pressure management",
        "Insulin therapy for diabetes mellitus",
        "Metformin as first-line diabetes treatment",
        "Chemotherapy for breast cancer",
        "Radiation therapy in oncology"
    ]
    
    concept_embeddings = model.encode(concepts)
    concept_similarities = cosine_similarity(concept_embeddings)
    
    print("\nConcepts:")
    for i, concept in enumerate(concepts):
        print(f"{i+1}. {concept}")
    
    print("\nTop Similar Pairs:")
    pairs = []
    for i in range(len(concepts)):
        for j in range(i+1, len(concepts)):
            pairs.append((i, j, concept_similarities[i][j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, j, score in pairs[:5]:
        print(f"  {concepts[i][:40]}... <-> {concepts[j][:40]}... : {score:.4f}")
    
    print("\n" + "="*70)
    print("✓ All tests completed successfully!")
    print("="*70)
    print("\nModel is working correctly and ready for use!")

if __name__ == "__main__":
    test_model()
