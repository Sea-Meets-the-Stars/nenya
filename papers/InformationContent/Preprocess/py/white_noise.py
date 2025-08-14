import os, sys
import numpy as np
import h5py

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import info_defs

dataset = 'WNoise'

def main(ntrain=150000, nvalid=50000, seed=1234, size=64):

    # Init
    nimg = ntrain+nvalid
    all_images = np.zeros((nimg, size, size))

    # Random generator
    rng = np.random.default_rng(seed)

    # Traning
    all_images[:ntrain] = rng.standard_normal(size=(ntrain, size, size))

    # Validation
    all_images[ntrain:] = rng.standard_normal(size=(nvalid, size, size))
    

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
    # Test
    #main(ntrain=1000, nvalid=500)

    # Full
    main()