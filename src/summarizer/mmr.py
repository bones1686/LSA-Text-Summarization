import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr_selection(sentence_vectors, lambda_param=0.6, summary_length=3):
    """
    Selects indices of sentences using Maximal Marginal Relevance.
    Args:
        sentence_vectors (np.array): Matrix of sentence vectors (n_sentences, n_features).
        lambda_param (float): Trade-off between relevance and diversity (0..1).
                              1.0 = Max Relevance, 0.0 = Max Diversity.
        summary_length (int): Number of sentences to select.
    Returns:
        list: Indices of selected sentences.
    """
    n_sentences = sentence_vectors.shape[0]
    
    # Handle edge case where requested length > available sentences
    summary_length = min(summary_length, n_sentences)
    if summary_length == 0:
        return []

    # 1. Compute Centroid (Query)
    # We assume the query is the mean of all sentence vectors (document embedding)
    query_vec = np.mean(sentence_vectors, axis=0).reshape(1, -1)
    
    # 2. Compute similarity of all sentences to Query (Sim1)
    sim_to_query = cosine_similarity(sentence_vectors, query_vec).flatten()
    
    selected_indices = []
    candidate_indices = list(range(n_sentences))
    
    # 3. Iterative Selection
    for _ in range(summary_length):
        best_mmr_score = -np.inf
        best_idx = -1
        
        for idx in candidate_indices:
            # Relevance part
            relevance = sim_to_query[idx]
            
            # Diversity part: max similarity to already selected
            if not selected_indices:
                diversity_penalty = 0
            else:
                # Similarity to selected sentences
                selected_vecs = sentence_vectors[selected_indices]
                current_vec = sentence_vectors[idx].reshape(1, -1)
                sim_to_selected = cosine_similarity(current_vec, selected_vecs).flatten()
                diversity_penalty = np.max(sim_to_selected)
                
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
                
        # Move best_idx from candidates to selected
        if best_idx != -1:
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
            
    return selected_indices

