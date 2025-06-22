""" Run Nenya on the N21 dataset """
import os

from nenya import workflow
import info_defs

from IPython import embed

def main(task:str):
    dataset = 'MODIS_SST_2km'
    pdict = info_defs.grab_paths(dataset)
    if task == 'train':
        workflow.train(pdict['opts_file'], debug=False)
    elif task == 'evaluate':
        # This takes 8 hours on profx with all CPU
        workflow.evaluate(pdict['opts_file'], pdict['preproc_file'], local_model_path=pdict['path'],
                          latents_file=pdict['latents_file'], debug=False)
    elif task == 'chk_latents':
        workflow.chk_latents(dataset, pdict['latents_file'], pdict['preproc_file'], 100)
    elif task == 'eigenimages':
        workflow.find_eigenmodes(pdict['opts_file'], pdict['pca_file'], 
                                 (1,128,128), f'{dataset}_eigenimages.npz',
                                 base_model_name='ckpt_epoch_3.pth', 
                                 local_model_path=pdict['path'],
                                 num_iterations=3000)#, show=True)
                                 #num_iterations=500, show=True)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = workflow.return_task()
    main(task)
