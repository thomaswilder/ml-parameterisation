#!/usr/bin/env python3

'''
    Description: Compute the feature to add scale-awareness to the parameterisation
    Ld / \Delta s

    Method: Load in Ld and EXP4 mask and compute.
'''

import xarray as xr
from xnemogcm import open_domain_cfg
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import yaml

# ------------ misc parameters ---------------- #

parser = argparse.ArgumentParser(description='Compute the scale-aware feature Ld / grid scale.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--start', required=True,
                    help='first month to process, YYYYMMDD (start_date_init_str)')
parser.add_argument('--end', required=True,
                    help='end of the period, YYYYMMDD, exclusive (end_date_init_str)')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

base_dir = config['paths']['features'] + f'/{region}/'
data_dir = 'coarsened_data/'
mask_path = [base_dir + f'mesh_mask_exp4_{region}.nc']

# -------------------------------------------- #
# Calculate grid scale, Delta s #
# -------------------------------------------- #
domcfg = open_domain_cfg(files = mask_path)

delta_s = np.sqrt(domcfg.e1t*domcfg.e2t)

# -------------------------------------------- #

# Initial date string
start_date_init_str = args.start

# End date string
end_date_init_str = args.end


# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")

# Loop to increment date by one day
current_date_init = start_date_init
while current_date_init < end_date_init:

    # Increment for next date
    next_date_init = current_date_init + relativedelta(months=+1)
    print(next_date_init)

    # Convert dates to string for nemo files
    date_init = (
        f"{current_date_init.year:04d}"\
        f"{current_date_init.month:02d}"\
        f"{current_date_init.day:02d}" # fix for adding leading zeros back in
    )

    print(date_init)

    # the month's last day: the model has a 360-day calendar, so every month ends on
    # day 30, and every monthly file is named MINT_1d_<first day>_<last day>_...
    date_end = f"{date_init[:6]}30"

    # Assign current date to next date for next loop
    current_date_init = next_date_init

    # set nemo filename using dates
    nemo_files = [f'MINT_1d_{date_init}_{date_end}_Ld_cg_{region}.nc']

    # each file is found by its full name, never by a pattern (a pattern such as
    # MINT_1d_00610101_*_Ld_cg also matches the full-period time mean); one list of
    # one path per file, the shape glob gave
    nemo_paths =  [[base_dir + data_dir + f] for f in nemo_files]
    print(nemo_paths)

    data = xr.open_dataset(nemo_paths[0][0])

    # compute scale aware feature
    scale_aware =  (data.Ld*1000) / delta_s

    # convert to dataset
    ds = scale_aware.to_dataset(name='sa')

    # assign attributes
    ds['sa'] = ds.sa.assign_attrs({'standard_name': 'Ld_over_grid_scale',
                    'long_name': 'deformation_radius_over_grid_scale'})

    # save data to filename
    output_filename = f'MINT_1d_{date_init}_{date_end}_sa_cg_{region}.nc'

    ds.to_netcdf(base_dir + data_dir + output_filename)