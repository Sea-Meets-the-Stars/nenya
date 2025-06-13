import os

def grab_paths(dataset:str):

    path = None
    preproc_file = None
    latents_file = None

    if dataset == 'MNIST':
        if 'OS_DATA' in os.environ:
            path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'MNIST', 'Info')
            preproc_file = os.path.join(path, 'PreProc', 'mnist_resampled.h5')
            latents_file = os.path.join(path, 'latents', 
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'mnist_resampled_latents.h5')
        opts_file = 'opts_nenya_mnist.json'
    elif dataset == 'ImageNet':
        if 'OS_DATA' in os.environ:
            path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'ImageNet', 'Info')
            preproc_file = os.path.join(path, 'PreProc', 'imagenet_processed.h5')
            #latents_file = os.path.join(path, 'latents', 
            #                    'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
            #                    'mnist_resampled_latents.h5')
        opts_file = 'opts_nenya_imagenet.json'
    elif dataset in ['MODIS_SST_2km', 'MODIS_SST_2km_sub']:
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Nenya')
            preproc_file = os.path.join(path, 'PreProc', 'MODIS_R2019_2004_95clear_128x128_preproc_std.h5')
            latents_file = os.path.join(path,
                        'latents/MODIS_R2019_v4_REDO',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_5_cosine_warm', 
                        'MODIS_R2019_2004_95clear_128x128_latents_std.h5')
        opts_file = None
    elif dataset == 'SWOT_L3':
        if 'OS_SSH' in os.environ:
            path = os.path.join(os.getenv('OS_SSH'), 'SWOT_L3', 'Info')
            preproc_file = os.path.join(path, 'PreProc', 'SWOT_L3_250m_preproc.h5')
        opts_file = 'opts_nenya_swot_l3.json'
    else:
        raise ValueError(f"Dataset {dataset} not supported for Nenya.")

    # Return
    return opts_file, path, preproc_file, latents_file 