import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.vectorizer import TextVectorizer

def run_reference_pipeline():
    print("--- Running Reference Pipeline ---")
    
    # 1. Load Data
    loader = DataLoader(split="train")
    df = loader.load_data(n_samples=1) # Load just 1 article for demo
    document = df.iloc[0]['document']
    reference_summary = df.iloc[0]['summary']
    
    print(f"\nOriginal Document Length: {len(document)} chars")
    
    # 2. Preprocess
    preprocessor = TextPreprocessor()
    sentences_data = preprocessor.preprocess_text(document)
    corpus = [s['processed'] for s in sentences_data]
    original_sentences = [s['original'] for s in sentences_data]
    
    print(f"Number of sentences after filtering: {len(corpus)}")
    
    # 3. Vectorization
    vectorizer = TextVectorizer()
    tfidf_matrix, features = vectorizer.fit_transform(corpus)
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    
    # 4. TruncatedSVD (Sklearn Reference)
    # We want to see the singular values to check for 'elbow'
    n_components = min(tfidf_matrix.shape) - 1
    n_components = min(n_components, 50) # Limit for visualization
    
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(tfidf_matrix)
    
    singular_values = svd.singular_values_
    
    # 5. Visualize Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(singular_values) + 1), singular_values, 'bo-')
    plt.title('Scree Plot (Singular Values)')
    plt.xlabel('Component Number')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.savefig('scree_plot_reference.png')
    print("Scree plot saved to 'scree_plot_reference.png'")
    
    # 6. Basic Extraction (Top K based on LSA)
    # Just as a dummy check: pick top sentences from first few components
    # (This is not the full logic, just to verify pipeline works)
    print("\nTop sentences by first component weight:")
    U = svd.transform(tfidf_matrix) # U * Sigma approx
    # U is (n_sentences, n_components)
    
    # Sort sentences by contribution to 1st latent concept
    comp_0_scores = U[:, 0]
    top_indices = np.argsort(comp_0_scores)[::-1][:3]
    
    for idx in top_indices:
        print(f"- {original_sentences[idx]}")

if __name__ == "__main__":
    run_reference_pipeline()

