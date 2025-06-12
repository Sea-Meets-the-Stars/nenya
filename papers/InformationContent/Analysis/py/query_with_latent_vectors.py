import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
LATENTS_FILE = "Pass_006_latents.h5"
DATA_FILE = "Pass_006.h5"
PARTITION = "valid"
QUERY_INDEX = 6020
TOP_N = 5
OUTPUT_PNG = "query_neighbors_pcolor.png"

# --- Load latent vectors ---
with h5py.File(LATENTS_FILE, "r") as f:
    latents = f[PARTITION][:]
print(f"Loaded latents: {latents.shape}")

# --- Load image data ---
with h5py.File(DATA_FILE, "r") as f:
    images = f[PARTITION][:]
print(f"Loaded images: {images.shape}")

# --- Compute cosine similarity ---
query_vector = latents[QUERY_INDEX].reshape(1, -1)
similarities = cosine_similarity(query_vector, latents)[0]
top_indices = np.argsort(-similarities)[1:TOP_N+1]  # Exclude self

# --- Combine indices ---
all_indices = [QUERY_INDEX] + list(top_indices)

# --- Shared color scale for all plots ---
all_data = np.array([images[i][0] for i in all_indices])
vmin = np.min(all_data)
vmax = np.max(all_data)

# --- Plot using pcolor ---
plt.figure(figsize=(3 * len(all_indices), 3))

for i, idx in enumerate(all_indices):
    arr = images[idx][0]  # Assuming shape (1, H, W)

    plt.subplot(1, len(all_indices), i + 1)
    plt.pcolor(arr, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.gca().set_aspect("auto")

    if i == 0:
        plt.title("Query")
    else:
        plt.title(f"Sim: {similarities[idx]:.2f}")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)
print(f"✅ Saved plot with pcolor to: {OUTPUT_PNG}")

