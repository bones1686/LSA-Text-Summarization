import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.linear_algebra.lanczos_svd import LanczosSVD
from src.summarizer.rank_selector import RankSelector
from src.summarizer.mmr import mmr_selection
from src.vectorizer import TextVectorizer
from src.preprocessor import TextPreprocessor

class LSASummarizer:
    def __init__(self, min_sentence_length=10, max_iter=50):
        self.preprocessor = TextPreprocessor(min_sentence_length=min_sentence_length)
        self.vectorizer = TextVectorizer()
        self.svd_solver = None # Will init per document to adjust max_iter/dim if needed
        self.max_iter = max_iter

    def summarize(self, text, summary_length=3, lambda_param=0.6, fixed_k=None):
        """
        Generates summary for a given text.
        Args:
            text (str): Input text.
            summary_length (int): Number of sentences.
            lambda_param (float): MMR trade-off.
            fixed_k (int): Optional fixed rank. If None, uses Gavish-Donoho.
        Returns:
            str: Summary text.
        """
        # 1. Preprocess
        sentences_data = self.preprocessor.preprocess_text(text)
        if not sentences_data:
            return ""
            
        corpus = [s['processed'] for s in sentences_data]
        original_sentences = [s['original'] for s in sentences_data]
        
        if len(corpus) < summary_length:
             return " ".join(original_sentences)

        # 2. Vectorize
        tfidf_matrix, _ = self.vectorizer.fit_transform(corpus)
        # tfidf_matrix is (n_sentences, n_vocab)
        
        m, n = tfidf_matrix.shape
        
        # 3. SVD
        # Decide n_components buffer.
        # We need enough components to check for rank selection.
        # Let's say min(m, n, 100).
        n_components = min(m, n, 50)
        
        # Initialize Custom SVD
        # For small matrices, max_iter needs to be large enough
        actual_max_iter = max(self.max_iter, n_components * 2)
        
        self.svd_solver = LanczosSVD(n_components=n_components, max_iter=actual_max_iter)
        self.svd_solver.fit(tfidf_matrix)
        
        singular_values = self.svd_solver.singular_values_
        U = self.svd_solver.U_ # (n_sentences, n_components)
        
        if U is None or len(singular_values) == 0:
            # Fallback if SVD failed
            return " ".join(original_sentences[:summary_length])
            
        # 4. Rank Selection
        if fixed_k:
            k = fixed_k
        else:
            k = RankSelector.get_optimal_k(singular_values, m, n)
            
        # Clip k to available components
        k = min(k, len(singular_values))
        
        # 5. Sentence Embeddings in Latent Space
        # U_k * Sigma_k
        U_k = U[:, :k]
        S_k = np.diag(singular_values[:k])
        
        sentence_embeddings = U_k.dot(S_k)
        
        # 6. MMR
        selected_indices = mmr_selection(
            sentence_embeddings, 
            lambda_param=lambda_param, 
            summary_length=summary_length
        )
        
        # Sort indices to preserve original order? 
        # Usually summaries flow better if sentences are in order.
        selected_indices.sort()
        
        summary = " ".join([original_sentences[i] for i in selected_indices])
        return summary

