#!/usr/bin/env python3

'''
    Description: Filters input features generated from DINO EXP16

    Method: Scalars use gcm-filters GridType.IRREGULAR_WITH_LAND with spatially
        varying length scale with max scale the max grid scale from DINO EXP4
        (1/4 deg). Velocity vectors (variable = 'grid_UV') are
        filtered jointly on their native staggered grids using the
        viscosity-based GridType.VECTOR_C_GRID Laplacian.
'''

import gcm_filters
import xarray as xr
from xnemogcm import get_metrics, open_nemo_and_domain_cfg
import xgcm
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import yaml

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='filter_input_features.log', level=logging.DEBUG)

logger.info('Begin...')

# ------------ misc parameters ---------------- #

parser = argparse.ArgumentParser(description='Filter input features generated from DINO EXP16.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--start', required=True,
                    help='first month to process, YYYYMMDD (start_date_init_str)')
parser.add_argument('--end', required=True,
                    help='end of the period, YYYYMMDD, exclusive (end_date_init_str)')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
# the scalar option ('grid_T') filters variable_name with the scalar filter;
# 'grid_UV' (uo & vo) filters both velocity components jointly with the
# VECTOR_C_GRID filter on their native grids
parser.add_argument('--variable', required=True,
                    help="'grid_T' (scalar, filters --variable-name) or 'grid_UV' (uo and vo)")
parser.add_argument('--variable-name', default='thetao',
                    help="variable filtered when --variable is a scalar option, e.g. 'thetao' or 'so'")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

variable = args.variable
variable_name = args.variable_name

if variable == 'grid_UV':
    u_name, v_name = 'uo', 'vo'

directory = f"{config['paths']['subsets']}/{region}/"
if variable == 'grid_UV':
    # production grid_U/grid_V files are surface only, so the
    # full-depth mask conflicts on z_c
    mask16_path = [config['paths']['features'] + f'/{region}/mesh_mask_exp16_surface_{region}.nc']
else:
    mask16_path = [config['paths']['features'] + f'/{region}/mesh_mask_exp16_{region}.nc']
mask025_path = config['paths']['features'] + f'/{region}/mesh_mask_exp4_{region}.nc'

#TODO add xnemo option to load in deformation radius

# open spatial filter
kappa = xr.open_dataset(config['paths']['kappa'] + f'/kappa16_{region}.nc')

# rename dimensions for gcm-filters
kappa = kappa.rename({'y': 'y_c', 'x': 'x_c'})

# get max grid scale from EXP4 (1/4 deg)
mask025 = xr.open_dataset(mask025_path)
grid_scale = np.sqrt(mask025.e1t*mask025.e2t)
max_grid_scale = grid_scale.max()

# -------------------------------------------- #

# Initial date string
start_date_init_str = args.start

# End date string
end_date_init_str = args.end

logger.info('Beginning date loop')


# Convert date strings to datetime objects
start_date_init = datetime.strptime(start_date_init_str, "%Y%m%d")
end_date_init = datetime.strptime(end_date_init_str, "%Y%m%d")

# Loop to increment date by one day
current_date_init = start_date_init
while current_date_init < end_date_init:

    # Increment for next date
    next_date_init = current_date_init + relativedelta(months=+1)
    logger.info(f'Next date: {next_date_init}')

    # Convert dates to string for nemo files
    date_init = (
        f"{current_date_init.year:04d}"\
        f"{current_date_init.month:02d}"\
        f"{current_date_init.day:02d}" # fix for adding leading zeros back in
    )

    logger.info(f'Current date: {date_init}')

    # the month's last day: the model has a 360-day calendar, so every month ends on
    # day 30, and every monthly file is named MINT_1d_<first day>_<last day>_...
    date_end = f"{date_init[:6]}30"

    # Assign current date to next date for next loop
    current_date_init = next_date_init

    # set nemo filename using dates
    if variable == 'grid_UV':
        nemo_files = [f'MINT_1d_{date_init}_{date_end}_grid_U_{region}.nc',
                      f'MINT_1d_{date_init}_{date_end}_grid_V_{region}.nc']
    else:
        nemo_files = [f'MINT_1d_{date_init}_{date_end}_{variable}_{region}.nc']
    logger.info(f'Nemo files: {nemo_files}')

    # open dataset using xnemogcm
    # each file is found by its full name, never by a pattern (a pattern such as
    # MINT_1d_00610101_*_Ld_cg also matches the full-period time mean); one list of
    # one path per file, the shape glob gave
    nemo_paths = [[directory + f] for f in nemo_files]
    logger.info(f'Nemo paths: {nemo_paths}')


    ds = open_nemo_and_domain_cfg(
        nemo_files=[p for paths in nemo_paths for p in paths],
        domcfg_files=mask16_path)

    # Rechunk the dataset
    ds=ds.chunk(dict(y_c=-1))
    ds=ds.chunk(dict(x_f=-1))
    ds=ds.chunk(dict(y_f=-1))
    ds=ds.chunk(dict(x_c=-1))

    # Need west and south scale factor convention for irregular filter on tripolar grid
    dxw = ds.e1u.roll(x_f=-1, roll_coords=False)  # x-spacing centered at western T-cell edge in m
    dyw = ds.e2u.roll(x_f=-1, roll_coords=False)  # y-spacing centered at western T-cell edge in m
    dxs = ds.e1v.roll(y_f=-1, roll_coords=False)  # x-spacing centered at southern T-cell edge in m
    dys = ds.e2v.roll(y_f=-1, roll_coords=False)  # y-spacing centered at southern T-cell edge in m

    # Ensure that coordinates are the same
    wet_mask = ds.tmask.isel(z_c=0)
    dxw = dxw.swap_dims({"x_f": "x_c"})
    dyw = dyw.swap_dims({"x_f": "x_c"})
    dxs = dxs.swap_dims({"y_f": "y_c"})
    dys = dys.swap_dims({"y_f": "y_c"})
    area = ds.e1t*ds.e2t

    # find minimum grid spacing
    dx_min = min(ds.e1t.where(ds.tmask.isel(z_c=0)).min(), ds.e2t.where(ds.tmask.isel(z_c=0)).min())
    dx_min = dx_min.values

    # set up the filter
    specs = {
            'filter_scale': max_grid_scale.values,
            'dx_min': dx_min,
            'filter_shape': gcm_filters.FilterShape.GAUSSIAN
        }

    filter_irregular_with_land = gcm_filters.Filter(
        **specs,
        grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
        grid_vars={
            'wet_mask': wet_mask,
            'dxw': dxw, 'dyw': dyw, 'dxs': dxs, 'dys': dys, 'area': area, 
            'kappa_w': kappa.kappa, 'kappa_s': kappa.kappa,
        }
    ) # need to vary kappa using the deformation radius.
    filter_irregular_with_land

    # viscosity-based vector filter for velocity components, following
    # https://gcm-filters.readthedocs.io/en/latest/examples/example_vector_laplacian.html
    # the spatially varying scale goes in kappa_iso; kappa_aniso only adds
    # viscosity aligned with the x-direction, so it is set to zero
    if variable == 'grid_UV':
        # U/V/F-point grid vars on common (y_c, x_c) dims: gcm-filters works
        # positionally and the VECTOR_C_GRID Laplacian handles the C-grid
        # staggering internally
        wet_mask_f = ds.fmask.isel(z_c=0).swap_dims({"x_f": "x_c", "y_f": "y_c"})
        dxCu = ds.e1u.swap_dims({"x_f": "x_c"})  # x-spacing centered at U point in m
        dyCu = ds.e2u.swap_dims({"x_f": "x_c"})  # y-spacing centered at U point in m
        dxCv = ds.e1v.swap_dims({"y_f": "y_c"})  # x-spacing centered at V point in m
        dyCv = ds.e2v.swap_dims({"y_f": "y_c"})  # y-spacing centered at V point in m
        dxBu = ds.e1f.swap_dims({"x_f": "x_c", "y_f": "y_c"})  # x-spacing centered at F point in m
        dyBu = ds.e2f.swap_dims({"x_f": "x_c", "y_f": "y_c"})  # y-spacing centered at F point in m

        filter_vector_c_grid = gcm_filters.Filter(
            **specs,
            grid_type=gcm_filters.GridType.VECTOR_C_GRID,
            grid_vars={
                'wet_mask_t': wet_mask, 'wet_mask_q': wet_mask_f,
                'dxT': ds.e1t, 'dyT': ds.e2t,
                'dxCu': dxCu, 'dyCu': dyCu, 'area_u': dxCu * dyCu,
                'dxCv': dxCv, 'dyCv': dyCv, 'area_v': dxCv * dyCv,
                'dxBu': dxBu, 'dyBu': dyBu,
                'kappa_iso': kappa.kappa, 'kappa_aniso': xr.zeros_like(kappa.kappa),
            }
        )
        filter_vector_c_grid

    # set up xgcm grid
    grid = xgcm.Grid(
                ds,
                metrics=get_metrics(ds),
                )

    bd = {'boundary': 'extend'}
    
    if variable == 'grid_UV':
        uo = ds[u_name].isel(z_c=0)
        vo = ds[v_name].isel(z_c=0)

        # filter u and v jointly on their native staggered grids: swap to
        # common (y_c, x_c) dims for gcm-filters, then swap back for saving
        u_f, v_f = filter_vector_c_grid.apply_to_vector(
            uo.swap_dims({"x_f": "x_c"}),
            vo.swap_dims({"y_f": "y_c"}),
            dims=['y_c', 'x_c'])
        ds[f'{u_name}_f'] = u_f.drop_vars('x_c', errors='ignore') \
                               .swap_dims({"x_c": "x_f"})
        ds[f'{v_name}_f'] = v_f.drop_vars('y_c', errors='ignore') \
                               .swap_dims({"y_c": "y_f"})

    else:
        ds[f'{variable_name}_f'] = filter_irregular_with_land.apply(ds[variable_name],
                                                           dims=['y_c', 'x_c'])
        
    logger.info('Dataset is: %s', ds)

    # create dataset
    if variable == 'grid_UV':
        ds_tmp_u = xr.Dataset(
            data_vars={
                f'{u_name}': (["t", "y_c", "x_f"],
                            ds[f'{u_name}_f'].values),
            },
            coords={
                "t": (["t"], ds[f'{u_name}_f'].t.values,
                            ds[f'{u_name}_f'].t.attrs),
                "gphiu": (["y_c", "x_f"], ds.gphiu.values,
                        {"standard_name": "Latitude", "units": "degrees_north"}),
                "glamu": (["y_c", "x_f"], ds.glamu.values,
                        {"standard_name": "Longitude","units": "degrees_east"}),
            },
            attrs={
                "name": "NEMO dataset",
                "description": f"Filtered {u_name} (VECTOR_C_GRID) \
                                -> ocean U grid variables",
            },
    )
        ds_tmp_v = xr.Dataset(
            data_vars={
                f'{v_name}': (["t", "y_f", "x_c"],
                            ds[f'{v_name}_f'].values),
            },
            coords={
                "t": (["t"], ds[f'{v_name}_f'].t.values,
                            ds[f'{v_name}_f'].t.attrs),
                "gphiv": (["y_f", "x_c"], ds.gphiv.values,
                        {"standard_name": "Latitude", "units": "degrees_north"}),
                "glamv": (["y_f", "x_c"], ds.glamv.values,
                        {"standard_name": "Longitude","units": "degrees_east"}),
            },
            attrs={
                "name": "NEMO dataset",
                "description": f"Filtered {v_name} (VECTOR_C_GRID) \
                                -> ocean V grid variables",
            },
    )
    else:
        ds_tmp = xr.Dataset(
            data_vars={
                f'{variable_name}': (["t", "z_c", "y_c", "x_c"], 
                            ds[f'{variable_name}_f'].values),
            },
            coords={
                "t": (["t"], ds[f'{variable_name}_f'].t.values,
                            ds[f'{variable_name}_f'].t.attrs),
                "gphit": (["y_c", "x_c"], ds[f'{variable_name}_f'].gphit.values, 
                        {"standard_name": "Latitude", "units": "degrees_north"}),
                "glamt": (["y_c", "x_c"], ds[f'{variable_name}_f'].glamt.values, 
                        {"standard_name": "Longitude","units": "degrees_east"}),
            },
            attrs={
                "name": "NEMO dataset",
                "description": f"Filtered {variable_name} at cell centre \
                                -> ocean T grid variables",
            },
        )

    # save data to netcdf
    save_directory = config['paths']['features'] + f'/{region}/filtered_data/'

    if variable == 'grid_UV':
        output_file_u = f'MINT_1d_{date_init}_{date_end}_{u_name}_f_{region}.nc'
        output_file_v = f'MINT_1d_{date_init}_{date_end}_{v_name}_f_{region}.nc'
        logger.info('Saving filtered velocities to: %s and %s',
                    save_directory + output_file_u,
                    save_directory + output_file_v)
        ds_tmp_u.to_netcdf(save_directory + output_file_u)
        ds_tmp_v.to_netcdf(save_directory + output_file_v)
    else:
        output_file = f'MINT_1d_{date_init}_{date_end}_{variable_name}_f_{region}.nc'

        ds_tmp.to_netcdf(save_directory + output_file)