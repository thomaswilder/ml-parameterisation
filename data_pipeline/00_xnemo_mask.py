#!/usr/bin/env python3

'''
    Description: Row E. The regional 1/4 degree mesh mask with the dimension names of
        xnemogcm, the mask that goes with the training files.

    Method: The mask of row A with time_counter, x, y renamed to t, x_c, y_c.

    Reads:  {paths.features}/<region>/mesh_mask_exp4_<region>.nc
    Writes: {paths.features}/<region>/coarsened_data/mesh_mask_exp4_<region>_xnemo.nc

    From the first half of the notebook machine_learning/make_mask_xnemo_readable.ipynb.

    The released masks of SO_GYRE and SO_TROP are what this script writes. Those of
    SO_JET and SO_EAST were made earlier from a mask cut with xarray `where` and
    `dropna`: they hold the same values, but their 7 masks are float32 (int8 here),
    mbathy and misf float64 (int32 here), and e3t_1d, e3w_1d, gdept_1d, gdepw_1d
    are broadcast over y_c, x_c.
'''

import argparse

import xarray as xr
import yaml

parser = argparse.ArgumentParser(
    description='The regional 1/4 degree mesh mask with the dimension names of xnemogcm.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region
directory = config['paths']['features'] + f'/{region}/'

mask = xr.open_dataset(directory + f"mesh_mask_exp4_{region}.nc")

mask = mask.rename({'time_counter': 't', 'x': 'x_c', 'y': 'y_c'})

mask.to_netcdf(directory + 'coarsened_data/' + f"mesh_mask_exp4_{region}_xnemo.nc")
print(f"Saved to {directory}coarsened_data/mesh_mask_exp4_{region}_xnemo.nc")
