""" Run Nenya on the N21 dataset """

from importlib import reload
import os


from nenya import latents_extraction
from nenya import workflow
import info_defs


#opts_file = 'opts_nenya_modis.json'
#
#if 'OS_SST' in os.environ:
#    modis_path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Info')
#    preproc_file = os.path.join(modis_path, 'PreProc', 'train_MODIS_2021_128x128.h5')
#    latents_file = os.path.join(modis_path, 'latents', 'MODIS_2021',
#                        'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
#                        'train_MODIS_2021_128x128_latents.h5')

from IPython import embed

def evaluate():
    # Copy in the model
    #model_file = os.path.join(modis_path, 'models', 'MODIS_2021',
    #                    'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_5_cosine_warm',
    #                    'last.pth')
    #shutil.copy(model_file, './')

    # Evaluate the model for MODIS native
    latents_extraction.evaluate("opts_nenya_modis.json", 
                os.path.join(modis_path, 'PreProc', 'train_MODIS_2021_128x128.h5'),
                local_model_path=modis_path,
                use_gpu=False,
                debug=False, clobber=True)

'''
def chk_latents(query_idx:int, partition:str='train', top_N:int=5):

    # Grab the latents
    with h5py.File(latents_file, 'r') as f:
        latents = f[partition][:]
        print(f"Latents shape: {latents.shape}")


    # Closest
    closest_idx, similarities = analysis.find_closest_latents(latents, query_idx)

    # Grab the images
    with h5py.File(preproc_file, 'r') as f:
        images = [f[partition][idx] for idx in [query_idx]+closest_idx[:top_N].tolist()]
        print(f"Grabbed {len(images)} images for plotting including the query.")

    # Plot
    embed(header='53 of nenya')
    reload(plotting)
    plotting.closest_latents(images,  similarities,
                          output_png=f'nenya_MODIS_{partition}_chk_latents_{query_idx}.png')
'''


def main(task:str):
    dataset = 'MODIS'
    pdict = info_defs.grab_paths(dataset)
    if task == 'train':
        workflow.train(pdict['opts_file'], load_epoch=23, debug=False)
    elif task == 'evaluate':
        # This takes 6 hours on profx with all CPU
        workflow.evaluate(pdict['opts_file'], pdict['preproc_file'], local_model_path=pdict['path'],
                          latents_file=pdict['latents_file'],
                          base_model_name='ckpt_epoch_23.pth')
    elif task == 'chk_latents':
        workflow.chk_latents(dataset, pdict['latents_file'], pdict['preproc_file'], 100)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = workflow.return_task()
    main(task)
