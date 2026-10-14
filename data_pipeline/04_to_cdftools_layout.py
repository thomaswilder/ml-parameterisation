#!/usr/bin/env python3

'''
    Description: Converts coarse-grained xnemogcm datasets to cdftools-readable format.

    Method: Reformats coarse-grained T and S datasets so they match the native
        NEMO grid_T output structure (dimensions, coordinates, etc.), enabling
        downstream processing with cdftools (e.g. buoyancy frequency computation).
'''

import xarray as xr
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import argparse
import yaml

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='convert_xnemo_to_cdftools_readable.log', level=logging.DEBUG)

logger.info('Begin...')

# ------------ misc parameters ---------------- #

parser = argparse.ArgumentParser(description='Convert coarse-grained T and S to the layout that cdftools reads (cdfbn2).')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--start', required=True,
                    help='first month to process, YYYYMMDD (start_date_init_str)')
parser.add_argument('--end', required=True,
                    help='end of the period, YYYYMMDD, exclusive (end_date_init_str)')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
parser.add_argument('--variable-T', default='grid_T',
                    help='NEMO file type used as the template')
parser.add_argument('--variable-name-T', default='thetao')
parser.add_argument('--variable-name-S', default='so')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

nemo_directory = config['paths']['subsets'] + f'/{region}/'
coarse_directory = config['paths']['features'] + f'/{region}/coarsened_data/'

variable_T = args.variable_T
variable_name_T = args.variable_name_T
variable_name_S = args.variable_name_S

# ------------ helper functions ---------------- #

def reformat_to_template(source_ds, template_ds, var_name):
    """
    Reformat source_ds so that the variable `var_name` has the same
    dimensions and coordinates as template_ds.
    """
    # --- Step 1: Build dimension rename mapping ---
    # Map source dim names to template dim names based on known aliases
    alias_groups = [
        {"x", "x_c", "x_f", "lon", "longitude"},
        {"y", "y_c", "y_f", "lat", "latitude"},
        {"z", "z_c", "depth", "deptht", "lev", "level"},
        {"t", "time", "time_counter"},
    ]

    dim_rename = {}
    for src_dim in source_ds.dims:
        if src_dim in template_ds.dims:
            continue  # already matches
        # Find alias group for this source dim
        for group in alias_groups:
            if src_dim in group:
                # Find the template dim in the same alias group with matching size
                for tgt_dim in template_ds.dims:
                    if tgt_dim in group and template_ds.sizes[tgt_dim] == source_ds.sizes[src_dim]:
                        dim_rename[src_dim] = tgt_dim
                        break
                break

    # Rename dimensions to match template
    renamed_ds = source_ds.rename(dim_rename)

    # --- Step 2: Extract data variable with explicit dim ordering ---
    data = renamed_ds[var_name]
    target_order = ("time_counter", "deptht", "y", "x")
    # Use target dims that exist in data; append any extras at the end
    ordered_dims = [d for d in target_order if d in data.dims]
    extra_dims = [d for d in data.dims if d not in target_order]
    data = data.transpose(*(ordered_dims + extra_dims))

    # --- Step 3: Build output dataset with template structure ---
    ds_out = xr.Dataset(attrs=template_ds.attrs)
    ds_out[var_name] = data

    # --- Step 4: Copy ALL coordinates from template whose dims exist in output ---
    for coord_name, coord_var in template_ds.coords.items():
        if set(coord_var.dims).issubset(set(ds_out.dims)):
            ds_out[coord_name] = coord_var

    # --- Step 5: Copy bounds variables referenced by coordinates ---
    for coord_name in list(ds_out.coords):
        bounds_name = ds_out[coord_name].attrs.get("bounds")
        if isinstance(bounds_name, str) and bounds_name in template_ds:
            ds_out[bounds_name] = template_ds[bounds_name]

    # --- Step 6: Rename x_c/y_c to x/y if needed ---
    xy_rename = {}
    if "x_c" in ds_out.dims and "x" not in ds_out.dims:
        xy_rename["x_c"] = "x"
    if "y_c" in ds_out.dims and "y" not in ds_out.dims:
        xy_rename["y_c"] = "y"
    if xy_rename:
        ds_out = ds_out.rename(xy_rename)

    # --- Step 7: Rename glamt/gphit to nav_lon/nav_lat if needed ---
    coord_rename = {}
    if "glamt" in ds_out.coords and "nav_lon" not in ds_out.coords:
        coord_rename["glamt"] = "nav_lon"
    if "gphit" in ds_out.coords and "nav_lat" not in ds_out.coords:
        coord_rename["gphit"] = "nav_lat"
    if coord_rename:
        ds_out = ds_out.rename(coord_rename)

    return ds_out

# ------------ date loop parameters ------------ #

# Initial date string
start_date_init_str = args.start

# End date string
end_date_init_str = args.end

# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")

# ------------ main loop ----------------------- #

current_date_init = start_date_init
while current_date_init < end_date_init:

    # Increment for next date
    next_date_init = current_date_init + relativedelta(months=+1)
    print(next_date_init)

    # Convert dates to string for nemo files
    date_init = (
        f"{current_date_init.year:04d}"
        f"{current_date_init.month:02d}"
        f"{current_date_init.day:02d}"
    )

    print(date_init)

    # the month's last day: the model has a 360-day calendar, so every month ends on
    # day 30, and every monthly file is named MINT_1d_<first day>_<last day>_...
    date_end = f"{date_init[:6]}30"

    # Assign current date to next date for next loop
    current_date_init = next_date_init

    # set nemo and coarse-grained filenames using dates
    nemo_pattern = f'MINT_1d_{date_init}_{date_end}_{variable_T}_{region}.nc'
    thetao_pattern = f'MINT_1d_{date_init}_{date_end}_{variable_name_T}_cg_{region}.nc'
    so_pattern = f'MINT_1d_{date_init}_{date_end}_{variable_name_S}_cg_{region}.nc'

    # each file is found by its full name, never by a pattern (a pattern such as
    # MINT_1d_00610101_*_Ld_cg also matches the full-period time mean); a list that
    # is empty when the file is missing, as glob gave
    nemo_paths = [p for p in [nemo_directory + nemo_pattern] if os.path.isfile(p)]
    thetao_paths = [p for p in [coarse_directory + thetao_pattern] if os.path.isfile(p)]
    so_paths = [p for p in [coarse_directory + so_pattern] if os.path.isfile(p)]

    print(f'NEMO template: {nemo_paths}')
    print(f'Coarse T: {thetao_paths}')
    print(f'Coarse S: {so_paths}')

    if not nemo_paths or not thetao_paths or not so_paths:
        logger.warning('Missing files for date %s, skipping...', date_init)
        print(f'Missing files for date {date_init}, skipping...')
        continue

    # open datasets
    ds = xr.open_dataset(nemo_paths[0])
    dsT = xr.open_dataset(thetao_paths[0])
    dsS = xr.open_dataset(so_paths[0])

    logger.info('Opened datasets for date %s', date_init)

    # identify variable names
    t_var = variable_name_T if variable_name_T in dsT.data_vars else list(dsT.data_vars)[0]
    s_var = variable_name_S if variable_name_S in dsS.data_vars else list(dsS.data_vars)[0]

    # reformat coarse-grained datasets to match NEMO template
    ds_T_fmt = reformat_to_template(dsT, ds, var_name=t_var)
    ds_S_fmt = reformat_to_template(dsS, ds, var_name=s_var)

    logger.info('Reformatted datasets for date %s', date_init)
    logger.info('T formatted dims: %s', dict(ds_T_fmt.sizes))
    logger.info('S formatted dims: %s', dict(ds_S_fmt.sizes))

    # save to netcdf
    output_T = f'MINT_1d_{date_init}_{date_end}_{variable_name_T}_cg_{region}_cdftools.nc'
    output_S = f'MINT_1d_{date_init}_{date_end}_{variable_name_S}_cg_{region}_cdftools.nc'

    ds_T_fmt.to_netcdf(
        coarse_directory + output_T,
        unlimited_dims=["time_counter"],
    )
    ds_S_fmt.to_netcdf(
        coarse_directory + output_S,
        unlimited_dims=["time_counter"],
    )

    logger.info('Saved %s and %s', output_T, output_S)
    print(f'Saved {output_T} and {output_S}')


