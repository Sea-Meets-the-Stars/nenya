""" Run Nenya on the SWOT ssr dataset """

import os

import h5py

from nenya.train import main as train_main
from nenya import params 
from nenya import latents_extraction

# aws s3 cp s3://odsl/nasa_oceanai_workshop2025/justin/swot_L2unsmoothed_1dayRepeat_ssr_images_unh/Pass_003.parquet . --profile nasa-oceanai
# aws s3 cp s3://odsl/nasa_oceanai_workshop2025/justin/swot_L2unsmoothed_1dayRepeat_ssr_images_unh/Pass_003.h5 . --profile nasa-oceanai

# cp ~/Oceanography/python/nenya/papers/InformationContent/Analysis/opts_nenya_swot_test.json .
# cp ~/Oceanography/python/nenya/papers/InformationContent/Analysis/opts_nenya_swot_fast.json .

# aws s3 sync models s3://odsl/nasa_oceanai_workshop2025/justin/swot_L2unsmoothed_1dayRepeat_ssr_images_unh/ --profile nasa-oceanai 

swot_path = os.getenv('SWOT_PNGs')

from IPython import embed

def evaluate(opt_path, pp_file:str, debug=False, clobber=False, 
             preproc:str='_std'):
    """
    This function is used to obtain the latents of the trained model
    for all of VIIRS

    Args:
        opt_path: (str) option file path.
        model_name: (str) model name 
        clobber: (bool, optional)
            If true, over-write any existing file
    """
    # Parse the model
    opt = params.Params(opt_path)
    params.option_preprocess(opt)

    # Prep
    model_base, existing_files = latents_extraction.prep(opt)

    print(f"Working on {pp_file}")

    # Setup
    latents_file = pp_file.replace('.h5', '_latents.h5')

    # Extract
    print("Extracting latents")
    #embed(header='Check 49')
    latent_dict = latents_extraction.model_latents_extract(
        opt, pp_file, model_base, debug=debug)
    # Save
    latents_hf = h5py.File(latents_file, 'w')
    for partition in latent_dict.keys():
        latents_hf.create_dataset(partition, data=latent_dict[partition])
    latents_hf.close()


def main(flg):

    flg= int(flg)

    # Test
    if flg == 1:
        train_main("opts_nenya_swot_test.json", debug=False)

    # Main
    if flg == 2:
        train_main("opts_nenya_swot_fast.json", debug=False)

    # latents on Pass 006
    if flg == 10:
        evaluate("opts_nenya_swot_fast.json", 
                os.path.join(swot_path,'Pass_006.h5'), 
                debug=False, clobber=True)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]

    main(flg)