import os, sys
import numpy as np
import h5py

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs

def main(ntrain=150000, nvalid=50000):
    # Load up the existing file
    h5_file = os.path.join(os.getenv('OS_SSH'), 'swot_prototype_train.h5')

    print("Loading images from:", h5_file)
    with h5py.File(h5_file, 'r') as f:
        all_images = [f['valid'][:], f['train'][:]]
    # Combine the two sets of images
    all_images = np.concatenate(all_images, axis=0)[:,0,...]

    opts_file, path, preproc_file, latents_file = info_defs.grab_paths('SWOT_L3')
    # Split by 150000 and 50000

    print("Loaded")
    with h5py.File(preproc_file, 'w') as f:
        f.create_dataset('train', data=all_images[:ntrain])
        f.create_dataset('valid', data=all_images[ntrain:ntrain+nvalid])
        
        # Add metadata
        f.attrs['dataset'] = 'SWOT_L3'
        f.attrs['n_train'] = ntrain
        f.attrs['n_valid'] = nvalid
        f.attrs['image_shape'] = all_images.shape[1:]
    print(f"SWOT_L3 preprocessed and saved to: {preproc_file}")

if __name__ == '__main__':
    main()