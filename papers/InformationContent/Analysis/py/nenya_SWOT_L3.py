""" Run Nenya on the ImageNet dataset """

from nenya import workflow
import info_defs

from IPython import embed

def main(task:str):
    dataset = 'SWOT_L3'
    pdict = info_defs.grab_paths(dataset)
    if task == 'train':
        workflow.train(pdict['opts_file'], load_epoch=8, debug=False)
    elif task == 'evaluate':
        workflow.evaluate(pdict['opts_file'], pdict['preproc_file'], local_model_path=pdict['path'],
                          latents_file=pdict['latents_file'],
                          base_model_name='ckpt_epoch_28.pth', debug=False)
    elif task == 'chk_latents':
        workflow.chk_latents(dataset, pdict['latents_file'], pdict['preproc_file'], 100)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = workflow.return_task()
    main(task)