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
def plot_results(query_idx, imgs, neighbor_idxs, sim_scores):
    plt.figure(figsize=(15, 4))
    all_indices = [query_idx] + list(neighbor_idxs)
    for i, idx in enumerate(all_indices):
        img = Image.fromarray(imgs[idx]).convert("RGB")
        plt.subplot(1, len(all_indices), i + 1)
        plt.imshow(img)
        plt.axis("off")
        if i == 0:
            plt.title("Query")
        else:
            plt.title(f"Sim: {sim_scores[i-1]:.2f}")
    plt.tight_layout()
    plt.show()

print(f"Query index: {QUERY_INDEX}")
print("Top similar indices:", top_indices)
plot_results(QUERY_INDEX, images, top_indices, similarities[top_indices])

