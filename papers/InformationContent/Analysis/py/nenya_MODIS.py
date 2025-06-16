""" Run Nenya on the N21 dataset """

from importlib import reload
import os


from nenya import latents_extraction
from nenya import workflow
import info_defs

from IPython import embed



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
