""" Module to extract the information content of the 2024 data """

import asyncio

from wrangler.extract.grab_and_go import run as gg_run

tstart = '2024-01-01T00:00:00'
tend = '2024-12-31T23:59:59'

def extract_viirs(dataset:str, eoption_file:str,
                  ex_file:str, tbl_file:str, n_cores:int=15):

    asyncio.run(gg_run(dataset, tstart, tend, eoption_file, 
                       ex_file, tbl_file, n_cores, 
                       verbose=True, debug=False, 
                       save_local_files=False))

# Command line execution
if __name__ == '__main__':

    # VIIRS
    extract_viirs('VIIRS_N21', 'extract_viirs_std.json', 
                  'ex_VIIRS_N21_2024.h5', 'VIIRS_N21_2024.parquet')