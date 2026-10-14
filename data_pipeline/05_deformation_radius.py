#!/usr/bin/env python3

'''
    Description: Calculates baroclinic deformation radius from NEMO DINO data.

    Note: This script is modified slightly from the script in ORCA36 directory.
        Also, have made some adjustments to compute Ld from cg data - See #!
'''

import xarray as xr
from xnemogcm import open_domain_cfg
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import glob
import argparse
import yaml

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='compute_deformation_radius.log', level=logging.DEBUG)

logger.info('Begin...')

parser = argparse.ArgumentParser(description='Calculate the baroclinic deformation radius from N2.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--start', required=True,
                    help='first month to process, YYYYMMDD (start_date_init_str)')
parser.add_argument('--end', required=True,
                    help='end of the period, YYYYMMDD, exclusive (end_date_init_str)')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
parser.add_argument('--mode', required=True,
                    help="'cg' for the 1/4 degree coarse-grained grid; any other value for the 1/16 degree fine grid")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

mode = args.mode

if mode == 'cg':
    directory = config['paths']['features'] + f'/{region}/coarsened_data/'
    mask_path = [config['paths']['features'] + f'/{region}/mesh_mask_exp4_{region}.nc']
else:
    directory = config['paths']['features'] + f'/{region}/'
    mask_path = [directory + f'mesh_mask_exp16_{region}.nc']

# open mask dataset
mask = open_domain_cfg(files = mask_path)
logger.info('Mask dataset is: %s', mask)

# grid fields are constant in time; broadcasting supplies the t dimension
e3t = mask.e3t_0
ff_t = mask.ff_t
tmask_surf = mask.tmask.isel(z_c=0)

# Initial date string
start_date_init_str = args.start

# End date string
end_date_init_str = args.end

# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")

# Loop to increment date by one month
current_date_init = start_date_init
while current_date_init < end_date_init:

    logger.info('Processing date: %s', current_date_init)

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

    # Assign current date to next date for next loop
    current_date_init = next_date_init

    # set nemo filename using dates
    #! changes to nemo filename
    if mode == 'cg':
        nemo_files = [f'MINT_1d_{date_init}_*_bn2_cg_{region}.nc']
    else:
        nemo_files = [f'MINT_1d_{date_init}_*_bn2_{region}.nc']

    # get path
    nemo_paths = [glob.glob(directory + f) for f in nemo_files]
    print(nemo_paths)

    # open dataset
    ds = xr.open_dataset(nemo_paths[0][0])

    # rename coordinate
    #! deptht is now z_c
    ds = ds.rename({'deptht': 'z_c',
                    'time_counter': 't',
                    'y': 'y_c',
                    'x': 'x_c'})

    ds['z_c'] = mask['z_c']

    # clamp negative values before taking the square root
    N = np.sqrt(xr.where(ds.vobn2 < 0, 1e-12, ds.vobn2))

    # compute deformation radius in km
    Ld = (N*e3t).sum(dim='z_c')/(np.pi*np.abs(ff_t)*1000.0) #! added missing pi in denominator

    # zero out land values (tmask is 0 over land)
    Ld = (Ld*tmask_surf).transpose('t', 'y_c', 'x_c')

    #! use xnemo format
    ds_out = xr.Dataset(
        data_vars={
            'Ld': (["t", "y_c", "x_c"], Ld.data),
        },
        coords={
            "t": (["t"], ds.t.values,
                             {k: v for k, v in ds.t.attrs.items() if k != 'bounds'}),
            "gphit": (["y_c", "x_c"], mask.gphit.values,
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamt": (["y_c", "x_c"], mask.glamt.values,
                      {"standard_name": "Longitude", "units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": "Deformation radius \
                              -> ocean T grid variables",
        },
    )

    # extract date_end from nemo_paths
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]

    # save data to netcdf
    if mode == 'cg':
        output_file = f'MINT_1d_{date_init}_{date_end}_Ld_cg_{region}.nc'
        save_directory = config['paths']['features'] + f'/{region}/coarsened_data/'
    else:
        output_file = f'MINT_1d_{date_init}_{date_end}_Ld_{region}.nc'
        save_directory = config['paths']['features'] + f'/{region}/'

    logger.info('Saving deformation radius to: %s', save_directory + output_file)

    ds_out.to_netcdf(save_directory + output_file)

    ds.close()
