""" Run simple tests """

import os
import numpy as np

from nenya.train import main as train_main

# aws3 cp s3://viirs/Tables/VIIRS_2013_tst_train_98.parquet .
# aws3 cp s3://viirs/PreProc/VIIRS_2013_98clear_192x192_preproc_viirs_tst_train.h5 .
train_main("opts_nenya_viirs_test.json", debug=False)