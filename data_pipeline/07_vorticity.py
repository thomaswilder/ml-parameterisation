#!/usr/bin/env python3

'''
    Description: Calculates relative vorticity
    
    Method: Uses geostrophic velocity

    #TODO Check why vorticity ends up with incorrect pandas index
'''

import glob
import argparse
import yaml
import xarray as xr
from xnemogcm import open_domain_cfg, get_metrics
import xgcm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='compute_vorticity.log', level=logging.INFO, filemode='w')

logger.info('Begin...')

parser = argparse.ArgumentParser(description='Calculate the relative vorticity of the coarse-grained velocity.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--start', required=True,
                    help='first month to process, YYYYMMDD (start_date_init_str)')
parser.add_argument('--end', required=True,
                    help='end of the period, YYYYMMDD, exclusive (end_date_init_str)')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
parser.add_argument('--mode', required=True,
                    help="'full' (the geostrophic branch is dead)")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

mode = args.mode # 'geostrophic' or 'full' velocity


directory = config['paths']['features'] + f'/{region}/coarsened_data/'
mask_path = [config['paths']['features'] + f'/{region}/mesh_mask_exp4_{region}.nc']

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

    logger.info('Processing date: %s', current_date_init)

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
    if mode == 'geostrophic':
        nemo_files = [f'MINT_1d_{date_init}_*_ug_cg_{region}.nc',
                      f'MINT_1d_{date_init}_*_vg_cg_{region}.nc']
    else:
        nemo_files = [f'MINT_1d_{date_init}_*_uo_cg_{region}.nc',
                      f'MINT_1d_{date_init}_*_vo_cg_{region}.nc']
    print(nemo_files)

    nemo_paths = [glob.glob(directory + f) for f in nemo_files]

    nemo_files = [nemo_paths[0][0].split('/')[-1], nemo_paths[1][0].split('/')[-1]]

    # open dataset using xnemogcm
    nemo_paths = [directory + f for f in nemo_files]
    # ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths,
    #                               domcfg_files=mask_path)
    #! issues opening subsetted data with full xnemogcm, just hangs?
    #* use this workaround to open data and domcfg separately
    dataU = xr.open_dataset(nemo_paths[0])
    dataV = xr.open_dataset(nemo_paths[1])
    domcfg = open_domain_cfg(files = mask_path)

    # get surface domcfg
    domcfg_surface = domcfg.isel(z_c=0, z_f=0)

    grid = xgcm.Grid(domcfg, 
                     metrics=get_metrics(domcfg))
    
    bd = {'boundary': 'extend'}

    # compute relative vorticity
    if mode == 'geostrophic':
        zeta = 1/(domcfg_surface.e1f*domcfg_surface.e2f) * \
                (grid.diff(dataV.vg*domcfg_surface.e2v, 'X', **bd) \
                - grid.diff(dataU.ug*domcfg_surface.e1u, 'Y', **bd)) \
                    * domcfg_surface.fmask
    else:
        zeta = 1/(domcfg_surface.e1f*domcfg_surface.e2f) * \
                (grid.diff(dataV.vo*domcfg_surface.e2v, 'X', **bd) \
                - grid.diff(dataU.uo*domcfg_surface.e1u, 'Y', **bd)) \
                    * domcfg_surface.fmask
    
    logger.info('Relative vorticity calculated is: %s', zeta)

    #! interpolation done to put on cell centre
    # interpolate to cell centre
    zeta_c = grid.interp(zeta, ['X', 'Y'], **bd)

    ds_tmp = xr.Dataset(
        data_vars={
            'vor': (["y_c", "x_c", "t"], 
                        zeta_c.values),
        },
        coords={
            "t": (["t"], zeta_c.t.values,
                        zeta_c.t.attrs),
            "gphit": (["y_c", "x_c"], domcfg_surface.gphit.values, 
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamt": (["y_c", "x_c"], domcfg_surface.glamt.values, 
                      {"standard_name": "Longitude","units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": f"Relative vorticity from coarse_grained \
                  {mode} velocities -> ocean T grid variables",
        },
    )

    logger.info('Dataset to be saved is: %s', ds_tmp)

    # extract date_end from nemo_paths
    filename = nemo_paths[0].split('/')[-1]
    date_end = filename.split('_')[3]
    date_end

    # save data to netcdf
    if mode == 'geostrophic':
        output_file = f'MINT_1d_{date_init}_{date_end}_vorg_cg_{region}.nc'
    else:
        output_file = f'MINT_1d_{date_init}_{date_end}_vor_cg_{region}.nc'

    save_directory = config['paths']['features'] + f'/{region}/coarsened_data/'

    ds_tmp.to_netcdf(save_directory + output_file)