""" Grab 64x64 portions of the original VIIRS images """

import os, sys
import numpy as np
import h5py

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs

dataset = 'VIIRS_SST_sub'

def main(ntrain=150000, nvalid=50000, seed=1234, size=64):

    # Load up the full native
    native_dataset = 'VIIRS_SST'
    odict_native = info_defs.grab_paths(native_dataset)

    print("Loading images from:", odict_native['preproc_file'])
    with h5py.File(odict_native['preproc_file'], 'r') as f:
        all_images = [f['train'][:], f['valid'][:]]

    # Combine the two sets of images
    all_images = np.concatenate(all_images, axis=0)#[:,0,...]

    # Take lower 64x64
    all_images = all_images[:, 0:size, 0:size]

    odict = info_defs.grab_paths(dataset)
    with h5py.File(odict['preproc_file'], 'w') as f:
        f.create_dataset('train', data=all_images[:ntrain])
        f.create_dataset('valid', data=all_images[ntrain:ntrain+nvalid])
        
        # Add metadata
        f.attrs['dataset'] = dataset
        f.attrs['n_train'] = ntrain
        f.attrs['n_valid'] = nvalid
        f.attrs['image_shape'] = all_images.shape[1:]
    print(f"{dataset} preprocessed and saved to: {odict['preproc_file']}")

if __name__ == '__main__':
    # 
    main()