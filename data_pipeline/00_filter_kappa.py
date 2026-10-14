#!/usr/bin/env python3

'''
    Description: Row D. kappa, the field that sets the filter scale of 02_filter.py:
        the grid scale of the 1/4 degree grid over its maximum in the region,
        regridded to the 1/16 degree grid.

    Method: sqrt(e1t*e2t) of the regional 1/4 degree mask, over its maximum, then an
        xesmf conservative regridding from the 1/4 to the 1/16 degree grid (the
        reverse of row C; the weights are made here and not saved).

    Reads:  {paths.features}/<region>/mesh_mask_exp4_<region>.nc
            {paths.features}/<region>/<region>_target_grid_025.npy
            {paths.features}/<region>/<region>_input_grid_16.npy
    Writes: {paths.kappa}/kappa16_<region>.nc

    Needs xesmf. From the notebook coarsening/compute_kappa_for_filtering.ipynb.
'''

import argparse

import numpy as np
import xarray as xr
import xesmf as xe
import yaml

parser = argparse.ArgumentParser(
    description='kappa: the 1/4 degree grid scale over its maximum, on the 1/16 degree grid.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region
directory = config['paths']['features'] + f'/{region}/'

mask025 = xr.open_dataset(directory + f'mesh_mask_exp4_{region}.nc')

target_grid = np.load(directory + f'{region}_target_grid_025.npy', allow_pickle=True)
tar = target_grid.item()

input_grid = np.load(directory + f'{region}_input_grid_16.npy', allow_pickle=True)
inp = input_grid.item()

# from the 1/4 degree grid to the 1/16 degree grid
regridder = xe.Regridder(tar, inp, "conservative")

grid_scale = np.sqrt(mask025.e1t*mask025.e2t)
max_grid_scale = grid_scale.max()

kappa = grid_scale.isel(time_counter=0) / max_grid_scale
kappa = kappa.to_dataset(name='kappa')

kappa16 = regridder(kappa)

kappa16.to_netcdf(config['paths']['kappa'] + f'/kappa16_{region}.nc')
print(f"Saved to {config['paths']['kappa']}/kappa16_{region}.nc")
