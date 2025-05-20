""" Run simple tests """

import os
import numpy as np

from nenya.train import main as train_main

# s3 get s3://viirs/Tables/VIIRS_2013_tst_train_98.parquet
train_main("opts_nenya_viirs_test.json", debug=True)