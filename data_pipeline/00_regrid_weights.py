#!/usr/bin/env python3

'''
    Description: Row C. The weights of the conservative regridding from the 1/16 degree
        grid to the 1/4 degree grid of a region, for 03_regrid.py.

    Method: xesmf.Regridder(input grid, target grid, "conservative") on the grids of
        row B, saved with Regridder.to_netcdf.

    Reads:  {paths.features}/<region>/<region>_target_grid_025.npy
            {paths.features}/<region>/<region>_input_grid_16.npy
    Writes: {paths.features}/<region>/weights_<region>.nc

    Needs xesmf. From the notebook coarsening/get_regridding_weights.ipynb.
'''

import argparse

import numpy as np
import xesmf as xe
import yaml

parser = argparse.ArgumentParser(
    description='Conservative regridding weights from the 1/16 to the 1/4 degree grid of a region.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

directory = config['paths']['features'] + f'/{region}/'

array = np.load(directory + f'{region}_target_grid_025.npy', allow_pickle=True)
target_grid = array.item()

array = np.load(directory + f'{region}_input_grid_16.npy', allow_pickle=True)
input_grid = array.item()

regridder = xe.Regridder(input_grid, target_grid, "conservative")

regridder.to_netcdf(directory + f'weights_{region}.nc')
print(f'Saved to {directory}weights_{region}.nc')
