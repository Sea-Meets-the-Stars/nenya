""" Nenya related analysis methods """

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_closest_latents(latents:np.ndarray, query_index:int):
    """
    Find the indices of the closest latent vectors to a given query vector 
    based on cosine similarity.

    Args:
        latents (np.ndarray): A 2D array of latent vectors where each row 
                                represents a vector.
        query_index (int): The index of the query vector in the `latents` array.

    Returns:
        tuple: A tuple containing:
            - sorted_indices (np.ndarray): Indices of the latent vectors sorted 
                by descending similarity to the query vector, excluding the query 
                vector itself.
            - similarities (np.ndarray): Cosine similarity values corresponding 
                to the sorted indices.
    """
    # --- Compute cosine similarity ---
    query_vector = latents[query_index].reshape(1, -1)
    similarities = cosine_similarity(query_vector, latents)[0]
    sorted_indices = np.argsort(-similarities)[1:]  # Exclude self

    return sorted_indices, similarities[sorted_indices]
