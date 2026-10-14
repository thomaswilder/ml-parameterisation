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
import glob
import argparse
import yaml
from datetime import datetime
from dateutil.relativedelta import relativedelta

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='regrid_data.log', level=logging.INFO)

logger.info('Begin...')

def create_regridded_dataset(ds_regrid, variable):
    """
        Create dataset 
    """
    
    # Create data_vars dictionary dynamically
    data_vars = {}
    data_vars[variable] = (["t", "y_c", "x_c"], 
                        ds_regrid[variable].values)
    
    # Create the dataset
    ds_tmp = xr.Dataset(
        data_vars=data_vars,
        coords={
            "t": (["t"], ds_regrid.t.values),
            "gphit": (["y_c", "x_c"], ds_regrid.gphit.values,
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamt": (["y_c", "x_c"], ds_regrid.glamt.values,
                      {"standard_name": "Longitude","units": "degrees_east"}),
        },
        attrs={
            "description": \
                "DINO EXP16 regridded to EXP4 -> ocean T grid variables",
        }
    )
    
    return ds_tmp

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
parser.add_argument('--mode', required=True,
                    help="'non_geostrophic' (the geostrophic branch is dead)")
parser.add_argument('--variable', required=True,
                    help="e.g. 'thetao', 'so', 'uo', 'vo' or 'eke_Lmin3LdO25'")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

mode = args.mode # 'geostrophic' or 'full' velocity

variable = args.variable

# eke variants written by filtering/compute_kinetic_energy.py, all stored as
# fine_ke: 'eke' uses the spatially varying 3Ld filter, 'eke_Lmin3LdO25' uses
# the constant filter scale min(<3Ld>, <ds_O25>)
eke_variables = ('eke', 'eke_Lmin3LdO25')

# variables stored in the base region directory (e.g. SO_EAST); all other
# variables are filtered features read from filtered_data/
base_dir_variables = ('mke', 'bn2', 'Ld', 'eke_parts') + eke_variables

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
    # date = current_date.strftime("%Y%m%d")
    date_init = (
        f"{current_date_init.year:04d}"\
        f"{current_date_init.month:02d}"\
        f"{current_date_init.day:02d}" # fix for adding leading zeros back in
    )

    print(date_init)

    # Assign current date to next date for next loop
    current_date_init = next_date_init

    # set nemo filename using dates
    # if variable != 'coarse_ke' or variable != 'fine_ke':
    if mode == 'geostrophic':
        nemo_files = [f'MINT_1d_{date_init}_*_{variable}g_f_{region}.nc']
    else:
        if variable in base_dir_variables:
            nemo_files = [f'MINT_1d_{date_init}_*_{variable}_{region}.nc']
        else:
            nemo_files = [f'MINT_1d_{date_init}_*_{variable}_f_{region}.nc']

    nemo_paths = [glob.glob(data_directory + f) for f in nemo_files]

    # extract date_end from nemo_paths
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]

    nemo_files = [nemo_paths[0][0].split('/')[-1]]

    # open dataset using xnemogcm
    nemo_paths = [data_directory + f for f in nemo_files]
    # ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths,
    #                               domcfg_files=mask_path)
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

    if variable == 'uo' or variable == 'ug':
        data[variable] = grid.interp(data[variable], ['X'], **bd)
        data = data.rename({'gphiu': 'lat', 'glamu': 'lon'})
    elif variable == 'vo' or variable == 'vg':
        data[variable] = grid.interp(data[variable], ['Y'], **bd)
        data = data.rename({'gphiv': 'lat', 'glamv': 'lon'})
    else:
    # Rename coordinates to lat and lon for xesmf
        data = data.rename({'gphit': 'lat', 'glamt': 'lon'})

    # perform regridding
    ds_tmp = xr.Dataset()
    if variable == 'coarse_ke' or variable == 'mke':
        ds_tmp['coarse_ke'] = regridder(data['coarse_ke'])
    elif variable == 'fine_ke' or variable in eke_variables:
        # keep_attrs carries the filter lengthscale metadata written by
        # compute_kinetic_energy.py over to the coarse-grained fine_ke
        ds_tmp['fine_ke'] = regridder(data['fine_ke'], keep_attrs=True)
    elif variable == 'eke_parts':
        # the eddy velocity components and their kinetic energy all go through
        # the same regridder, so that the subgrid eke formed from them is a
        # variance of a single coarse-graining operator
        for part in ('up', 'vp', 'kp'):
            ds_tmp[part] = regridder(data[part])
    else:
        ds_tmp[variable] = regridder(data[variable])

    # the conservative regridding is performed on T cells, so the regridded
    # velocity components come back on the target T points; interpolate them
    # onto the native staggered points that they are labelled with
    if variable == 'uo' or variable == 'ug':
        ds_tmp = ds_tmp.rename({'y': 'y_c', 'x': 'x_c'}) \
                       .drop_vars(['lat', 'lon'])
        ds_tmp[variable] = grid_cg.interp(ds_tmp[variable], ['X'], **bd)
    elif variable == 'vo' or variable == 'vg':
        ds_tmp = ds_tmp.rename({'y': 'y_c', 'x': 'x_c'}) \
                       .drop_vars(['lat', 'lon'])
        ds_tmp[variable] = grid_cg.interp(ds_tmp[variable], ['Y'], **bd)

    # Rename coordinates back to gphit and glamt
    if variable == 'uo' or variable == 'ug':
        ds_tmp = ds_tmp.assign_coords(gphiu=domcfg_cg.gphiu,
                                      glamu=domcfg_cg.glamu)
    elif variable == 'vo' or variable == 'vg':
        ds_tmp = ds_tmp.assign_coords(gphiv=domcfg_cg.gphiv,
                                      glamv=domcfg_cg.glamv)
    else:
        ds_tmp = ds_tmp.rename({"lat": "gphit", "lon": "glamt"})

    if variable == 'uo' or variable == 'ug' \
            or variable == 'vo' or variable == 'vg':
        # already on the native staggered dims from the interpolation above
        ds = ds_tmp
    elif variable == 'vobn2':
        ds = ds_tmp
        ds = ds.rename({'y': 'y_c', 'x': 'x_c'})
    elif variable == 'coarse_ke' or variable == 'fine_ke' or variable in eke_variables \
            or variable == 'mke' or variable == 'eke_parts':
        ds = ds_tmp
        ds = ds.rename({'y': 'y_c', 'x': 'x_c'})
        if variable in eke_variables:
            # record the filter lengthscale in the dataset metadata too
            ds.attrs = {
                'description': 'DINO EXP16 regridded to EXP4 -> ocean T grid variables',
                **ds['fine_ke'].attrs,
            }
    elif variable == 'thetao' or variable == 'so':
        ds = ds_tmp
        ds = ds.rename({'y': 'y_c', 'x': 'x_c'})
    else:
        ds = create_regridded_dataset(ds_tmp, variable)

    # save data to netcdf; the eke variant is kept in the filename,
    # e.g. MINT_1d_..._eke_Lmin3LdO25_cg_SO_JET.nc
    if mode == 'geostrophic':
        output_file = f'MINT_1d_{date_init}_{date_end}_{variable}g_cg_{region}.nc'
    else:
        output_file = f'MINT_1d_{date_init}_{date_end}_{variable}_cg_{region}.nc'

    ds.to_netcdf(save_directory + output_file)

logger.info('End!')