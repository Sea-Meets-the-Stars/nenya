""" Module to extract N20 data """

import asyncio

from wrangler.extract.grab_and_go import run as gg_run


def extract_viirs_n20(tstart, tend,
    dataset:str, eoption_file:str,
                  ex_file:str, tbl_file:str, n_cores:int=15,
                  debug=False):

    #if debug:
    #    tstart = '2022-02-28T00:00:00'

    gg_run(dataset, tstart, tend, eoption_file, 
                       ex_file, tbl_file, n_cores, 
                       verbose=True, debug=debug, 
                       save_local_files=True,
                       debug_noasync=debug)

# Command line execution
if __name__ == '__main__':

    # One year at a time
    #for year in range(2020, 2025):
    for year in range(2022, 2025):
        tstart = f'{year}-01-01T00:00:00'
        tend = f'{year}-12-31T23:59:59'


        # VIIRS
        extract_viirs_n20(tstart, tend, 'VIIRS_N20', 'extract_viirs_std.json', 
                  f'ex_VIIRS_N20_{year}.h5', f'VIIRS_N20_{year}.parquet',
                  n_cores=15)#, debug=True)#, debug_async=True, debug=True)
                  #n_cores=15, debug=True)