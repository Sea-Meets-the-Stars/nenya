""" Nenya related analysis methods """

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def find_closest_latents(latents:np.ndarray, query_index:int):
    # --- Compute cosine similarity ---
    query_vector = latents[query_index].reshape(1, -1)
    similarities = cosine_similarity(query_vector, latents)[0]
    sorted_indices = np.argsort(-similarities)[1:]  # Exclude self

    return sorted_indices, similarities[sorted_indices]
