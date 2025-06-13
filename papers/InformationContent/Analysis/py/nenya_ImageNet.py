""" Run Nenya on the ImageNet dataset """

from nenya import workflow
import info_defs

from IPython import embed

def main(task:str):
    opts_file, mnist_path, preproc_file, latents_file = info_defs.grab_paths('ImageNet')
    if task == 'train':
        workflow.train(opts_file, debug=False)
    elif task == 'evaluate':
        workflow.evaluate(opts_file, preproc_file, local_model_path=mnist_path,
                          latents_file=latents_file) 
    elif task == 'chk_latents':
        workflow.chk_latents('ImageNet', latents_file, preproc_file, 100)
    else:
        raise ValueError(f"Unknown task: {task}")

# Command line execution
if __name__ == '__main__':
    task = workflow.return_task()
    main(task)