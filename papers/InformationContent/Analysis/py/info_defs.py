import os

def grab_paths(dataset:str):

    out_dict = {}
    out_dict['path'] = None
    out_dict['preproc_file'] = None
    out_dict['latents_file'] = None
    out_dict['model_file'] = None

    if dataset == 'MNIST':
        if 'OS_DATA' in os.environ:
            path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'MNIST', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'mnist_resampled.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'mnist_resampled_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_mnist.json'
        out_dict['pca_file'] = 'pca_latents_MNIST.npz'
    elif dataset == 'ImageNet':
        if 'OS_DATA' in os.environ:
            path = os.path.join(os.getenv('OS_DATA'), 'Natural', 'ImageNet', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'imagenet_processed.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'imagenet',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'imagenet_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_imagenet.json'
        out_dict['pca_file'] = 'pca_latents_ImageNet.npz'
    elif dataset == 'orig_MODIS_SST_2km':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Nenya')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'MODIS_R2019_2004_95clear_128x128_preproc_std.h5')
            out_dict['latents_file'] = os.path.join(path,
                        'latents/MODIS_R2019_v4_REDO',
                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_5_cosine_warm', 
                        'MODIS_R2019_2004_95clear_128x128_latents_std.h5')
        out_dict['opts_file'] = None
        # 200,000
        out_dict['pca_file'] = 'pca_latents_MODIS_SST_2km_sub.npz'
    elif dataset == 'MODIS_SST':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_MODIS_2021_128x128.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'MODIS_2021',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'train_MODIS_2021_128x128_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_modis.json'
        out_dict['pca_file'] = 'pca_latents_MODIS_SST.npz'
    elif dataset == 'MODIS_SST_2km':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_MODIS_2021_64x64.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 'MODIS_2021',
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
                                'train_MODIS_2021_64x64_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_modis_2km.json'
        out_dict['pca_file'] = 'pca_latents_MODIS_SST_2km.npz'
    elif dataset == 'VIIRS':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_VIIRS_N21_2024.h5')
        out_dict['opts_file'] = 'opts_nenya_viirs.json'
    elif dataset == 'VIIRS_2km':
        if 'OS_SST' in os.environ:
            path = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_VIIRS_N21_2024_2km.h5')
        out_dict['opts_file'] = 'opts_nenya_viirs_2km.json'
    elif dataset == 'orig_VIIRS_SST_2km':
            out_dict['pca_file'] = 'pca_latents_orig_VIIRS_SST_2km.npz'
    elif 'LLC_SST' in dataset:
        if 'OS_OGCM' in os.environ:
            path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Info')
            out_dict['path'] = path
            if 'nonoise' in dataset:
                out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'train_llc_nonoise.h5')
        out_dict['opts_file'] = 'opts_nenya_llc.json'
        if 'nonoise' in dataset:
            out_dict['pca_file'] = 'pca_latents_LLC_SST_nonoise.npz'
    elif dataset == 'SWOT_L3':
        if 'OS_SSH' in os.environ:
            path = os.path.join(os.getenv('OS_SSH'), 'SWOT_L3', 'Info')
            out_dict['path'] = path
            out_dict['preproc_file'] = os.path.join(path, 'PreProc', 'SWOT_L3_250m_preproc.h5')
            out_dict['latents_file'] = os.path.join(path, 'latents', 
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm', 
                                'SWOT_L3_250m_latents.h5')
        out_dict['opts_file'] = 'opts_nenya_swot_l3.json'
    else:
        raise ValueError(f"Dataset {dataset} not supported for Nenya.")

    # Return
    return out_dict