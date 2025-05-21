import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

from nenya import params, latents_extraction

# --- CONFIG ---
CONFIG_PATH = "opts_nenya_swot_fast.json"
MODEL_PATH = "last.pth"
DATA_FILE = "Pass_006.h5"
LATENTS_NPY = "latents.npy"
PARTITION = "valid"  # assumes all images are under 'train'
TOP_N = 5             # number of nearest neighbors
QUERY_INDEX = 0       # index of the image to compare

# --- STEP 1: Load config ---
opt = params.Params(CONFIG_PATH)

# --- STEP 2: Extract latent vectors ---
print("Extracting latent vectors...")
latent_dict = latents_extraction.model_latents_extract(
    opt=opt,
    data_file=DATA_FILE,
    model_path=MODEL_PATH,
    partitions=(PARTITION,),
    remove_module=True,
    debug=False
)
latents = latent_dict[PARTITION]
np.save(LATENTS_NPY, latents)
print(f"Saved {latents.shape[0]} latent vectors to {LATENTS_NPY}")

# --- STEP 3: Load images from HDF5 ---
with h5py.File(DATA_FILE, "r") as f:
    images = f[PARTITION][:]

# --- STEP 4: Find N nearest neighbors ---
query_vector = latents[QUERY_INDEX].reshape(1, -1)
similarities = cosine_similarity(query_vector, latents)[0]
top_indices = np.argsort(-similarities)[1:TOP_N+1]  # exclude self

# --- STEP 5: Plot results ---
def plot_results(query_idx, imgs, neighbor_idxs, sim_scores, output_file="query_neighbors.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(3 * (len(neighbor_idxs) + 1), 3))
    all_indices = [query_idx] + list(neighbor_idxs)

    for i, idx in enumerate(all_indices):
        arr = imgs[idx]

        # Normalize float32/float64 to uint8 for visibility
        if arr.dtype in (np.float32, np.float64):
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            arr = (arr * 255).astype(np.uint8)

        # Squeeze singleton dimensions
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        # Plot as heatmap (good for 1D/2D narrow signals)
        plt.subplot(1, len(all_indices), i + 1)
        plt.imshow(np.atleast_2d(arr), aspect="auto", cmap="viridis")
        plt.axis("off")
        if i == 0:
            plt.title("Query")
        else:
            plt.title(f"Sim: {sim_scores[i-1]:.2f}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    print(f"✅ Saved plot to: {output_file}")


plot_results(QUERY_INDEX, images, top_indices, similarities[top_indices])

