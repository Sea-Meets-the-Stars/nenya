""" Run Nenya on the SWOT ssr dataset """

import os
import json

from nenya.train import main as train_main

# aws s3 cp s3://odsl/nasa_oceanai_workshop2025/justin/swot_L2unsmoothed_1dayRepeat_ssr_images_unh/Pass_003.parquet . --profile nasa-oceanai
# aws s3 cp s3://odsl/nasa_oceanai_workshop2025/justin/swot_L2unsmoothed_1dayRepeat_ssr_images_unh/Pass_003.h5 . --profile nasa-oceanai

# cp ~/Oceanography/python/nenya/papers/InformationContent/Analysis/opts_nenya_swot_test.json .
# cp ~/Oceanography/python/nenya/papers/InformationContent/Analysis/opts_nenya_swot_fast.json .

# aws s3 sync models s3://odsl/nasa_oceanai_workshop2025/justin/swot_L2unsmoothed_1dayRepeat_ssr_images_unh/ --profile nasa-oceanai 

def main(flg):

    flg= int(flg)

    # Train the model
    if flg == 1:
        train_main("opts_nenya_swot_test.json", debug=False)
    # Main
    if flg == 2:
        train_main("opts_nenya_swot_fast.json", debug=False)

    #train_main("opts_nenya_viirs.json", debug=False)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]

    main(flg)