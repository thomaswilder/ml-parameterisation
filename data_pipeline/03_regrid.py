#!/usr/bin/env python3

'''
    Description: Regrid features and target using xesmf

    Method: Load in input and target grid from 
        get_mask_bounds_for_conservative_regridding.ipynb
        and load in weights precomputed in get_regridding_weights.ipynb
'''

import xarray as xr
import xesmf as xe
import xgcm
import numpy as np
from xnemogcm import open_domain_cfg, get_metrics
import argparse
import yaml
from datetime import datetime
from dateutil.relativedelta import relativedelta

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='regrid_data.log', level=logging.INFO)

logger.info('Begin...')

# ------------ misc parameters ---------------- #

parser = argparse.ArgumentParser(description='Regrid features and target from the 1/16 degree grid to the 1/4 degree grid with xesmf.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--start', required=True,
                    help='first month to process, YYYYMMDD (start_date_init_str)')
parser.add_argument('--end', required=True,
                    help='end of the period, YYYYMMDD, exclusive (end_date_init_str)')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
parser.add_argument('--variable', required=True,
                    help="'thetao', 'so', 'uo', 'vo' or 'eke_Lmin3LdO25'")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

variable = args.variable

# the eke written by filtering/compute_kinetic_energy.py, stored as fine_ke:
# 'eke_Lmin3LdO25' uses the constant filter scale min(<3Ld>, <ds_O25>)
eke_variables = ('eke_Lmin3LdO25',)

# variables stored in the base region directory (e.g. SO_EAST); all other
# variables are filtered features read from filtered_data/
base_dir_variables = eke_variables

directory = config['paths']['features'] + f'/{region}/'

if variable in base_dir_variables:
    data_directory = directory
else:
    data_directory = directory + 'filtered_data/'

save_directory = directory + 'coarsened_data/'
mask_path = [directory + f'mesh_mask_exp16_{region}.nc']
mask_path_cg = [directory + f'mesh_mask_exp4_{region}.nc']

weights_fn = directory + f'weights_{region}.nc'

# -------------------------------------------- #

# load in input and target grid
array = np.load(directory + f'{region}_input_grid_16.npy', allow_pickle=True)
input_grid = array.item()

array = np.load(directory + f'{region}_target_grid_025.npy', allow_pickle=True)
target_grid = array.item()

# load in regridder
regridder = xe.Regridder(input_grid, 
                         target_grid, 
                         "conservative", 
                         weights=weights_fn)

# target grid domcfg, used to interpolate the regridded velocity components
# back onto their native staggered points
domcfg_cg = open_domain_cfg(files = mask_path_cg)
grid_cg = xgcm.Grid(domcfg_cg,
                    metrics=get_metrics(domcfg_cg),
                    )

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
    if variable in base_dir_variables:
        nemo_files = [f'MINT_1d_{date_init}_{date_end}_{variable}_{region}.nc']
    else:
        nemo_files = [f'MINT_1d_{date_init}_{date_end}_{variable}_f_{region}.nc']

    # each file is found by its full name, never by a pattern (a pattern such as
    # MINT_1d_00610101_*_Ld_cg also matches the full-period time mean); one list of
    # one path per file, the shape glob gave
    nemo_paths = [[data_directory + f] for f in nemo_files]

    nemo_files = [nemo_paths[0][0].split('/')[-1]]

    # open dataset using xnemogcm
    nemo_paths = [data_directory + f for f in nemo_files]
    #! issues opening subsetted data with full xnemogcm, just hangs?
    #* use this workaround to open data and domcfg separately
    data = xr.open_dataset(nemo_paths[0])
    domcfg = open_domain_cfg(files = mask_path)

    # get surface domcfg
    domcfg_surface = domcfg.isel(z_c=0, z_f=0)

    # set up xgcm grid
    grid = xgcm.Grid(
            domcfg,
            metrics=get_metrics(domcfg),
            )

    bd = {'boundary': 'extend'}

    if variable == 'uo':
        data[variable] = grid.interp(data[variable], ['X'], **bd)
        data = data.rename({'gphiu': 'lat', 'glamu': 'lon'})
    elif variable == 'vo':
        data[variable] = grid.interp(data[variable], ['Y'], **bd)
        data = data.rename({'gphiv': 'lat', 'glamv': 'lon'})
    else:
    # Rename coordinates to lat and lon for xesmf
        data = data.rename({'gphit': 'lat', 'glamt': 'lon'})

    # perform regridding
    ds_tmp = xr.Dataset()
    if variable in eke_variables:
        # keep_attrs carries the filter lengthscale metadata written by
        # compute_kinetic_energy.py over to the coarse-grained fine_ke
        ds_tmp['fine_ke'] = regridder(data['fine_ke'], keep_attrs=True)
    else:
        ds_tmp[variable] = regridder(data[variable])

    # the conservative regridding is performed on T cells, so the regridded
    # velocity components come back on the target T points; interpolate them
    # onto the native staggered points that they are labelled with
    if variable == 'uo':
        ds_tmp = ds_tmp.rename({'y': 'y_c', 'x': 'x_c'}) \
                       .drop_vars(['lat', 'lon'])
        ds_tmp[variable] = grid_cg.interp(ds_tmp[variable], ['X'], **bd)
    elif variable == 'vo':
        ds_tmp = ds_tmp.rename({'y': 'y_c', 'x': 'x_c'}) \
                       .drop_vars(['lat', 'lon'])
        ds_tmp[variable] = grid_cg.interp(ds_tmp[variable], ['Y'], **bd)

    # Rename coordinates back to gphit and glamt
    if variable == 'uo':
        ds_tmp = ds_tmp.assign_coords(gphiu=domcfg_cg.gphiu,
                                      glamu=domcfg_cg.glamu)
    elif variable == 'vo':
        ds_tmp = ds_tmp.assign_coords(gphiv=domcfg_cg.gphiv,
                                      glamv=domcfg_cg.glamv)
    else:
        ds_tmp = ds_tmp.rename({"lat": "gphit", "lon": "glamt"})

    if variable == 'uo' or variable == 'vo':
        # already on the native staggered dims from the interpolation above
        ds = ds_tmp
    elif variable in eke_variables:
        ds = ds_tmp
        ds = ds.rename({'y': 'y_c', 'x': 'x_c'})
        # record the filter lengthscale in the dataset metadata too
        ds.attrs = {
            'description': 'DINO EXP16 regridded to EXP4 -> ocean T grid variables',
            **ds['fine_ke'].attrs,
        }
    elif variable == 'thetao' or variable == 'so':
        ds = ds_tmp
        ds = ds.rename({'y': 'y_c', 'x': 'x_c'})

    # save data to netcdf; the eke variant is kept in the filename,
    # e.g. MINT_1d_..._eke_Lmin3LdO25_cg_SO_JET.nc
    output_file = f'MINT_1d_{date_init}_{date_end}_{variable}_cg_{region}.nc'

    ds.to_netcdf(save_directory + output_file)

logger.info('End!')