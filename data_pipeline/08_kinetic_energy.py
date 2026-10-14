#!/usr/bin/python3

'''
    Description: Computes eddy kinetic energy using GCM-Filters.
        Velocity components are filtered on their native staggered grids
        using the viscosity-based VECTOR_C_GRID Laplacian; the squared
        velocity (a scalar) is filtered with IRREGULAR_WITH_LAND.
        Currently in CPU mode.

        The filter lengthscale is set by `filter_lengthscale`:
          'ld'          -> spatially varying filter scale 3Ld (original method)
          'min_3ld_o25' -> constant filter scale min(<3Ld>, <ds_O25>), where
                           <.> is a spatial average and ds_O25 is the grid
                           spacing of the 1/4 degree (EXP4) coarse grid.
                           Only implemented for variable = 'eke' (fine_ke).
'''

import glob
import argparse
import yaml
import gcm_filters
import xarray as xr
from xnemogcm import open_domain_cfg, get_metrics, open_nemo_and_domain_cfg
import xgcm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='compute_ke_dino.log', level=logging.INFO, filemode='w')

logger.info('Begin...')

# ------------ set parameters ---------------- #

parser = argparse.ArgumentParser(description='Compute eddy kinetic energy (eke) or mean kinetic energy (mke) with GCM-Filters.')
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
                    help="'eke' (fine grid, 1/16 degree velocity) or 'mke' (coarse-grained velocity)")
parser.add_argument('--depth', required=True,
                    help="'non_full' (surface velocity); 'full' is a dead branch")
parser.add_argument('--ld-scaling', type=int, default=3,
                    help='scale the deformation radius by this factor')
parser.add_argument('--filter-lengthscale', required=True,
                    help="'ld' (spatially varying 3Ld, needed for mke) or 'min_3ld_o25' (eke only)")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

mode = args.mode # 'geostrophic' or 'full' velocity
variable = args.variable # eke and eke_parts use full field, mke and res_eke use cg field
                     # eke_parts writes the eddy velocity and its kinetic
                     # energy on t-grid points, for coarse-graining with a
                     # single set of conservative weights
depth = args.depth # full depth profile or not

if mode == 'geostrophic':
    directory = config['paths']['features'] + f'/{region}/'
    mask_path = [directory + f'mesh_mask_exp16_surface_{region}.nc']
else:
    if variable == 'eke' or variable == 'eke_parts':
        if depth == 'full':
            directory = config['paths']['subsets'] + f'/{region}/full_depth/'
            mask_path = [config['paths']['features'] + f'/{region}/mesh_mask_exp16_{region}.nc']
        else:
            directory = config['paths']['subsets'] + f'/{region}/'
            mask_path = [config['paths']['features'] + f'/{region}/mesh_mask_exp16_surface_{region}.nc']
    elif variable == 'mke' or variable == 'res_eke':
        directory = config['paths']['features'] + f'/{region}/coarsened_data/'
        mask_path = [config['paths']['features'] + f'/{region}/mesh_mask_exp4_surface_{region}.nc']

# Initial date string
start_date_init_str = args.start

# End date string
end_date_init_str = args.end

# scale deformation radius
ld_scaling = args.ld_scaling

# filter lengthscale: 'ld' for the spatially varying 3Ld filter, or
# 'min_3ld_o25' for the constant filter scale min(<3Ld>, <ds_O25>)
filter_lengthscale = args.filter_lengthscale
# -------------------------------------------- #

# the alternative lengthscale is only set up for eke (fine_ke) on the fine grid
if filter_lengthscale == 'min_3ld_o25' and variable != 'eke':
    raise ValueError("filter_lengthscale = 'min_3ld_o25' is only implemented for variable = 'eke'")

# -------------------------------------------- #
# get max deformation radius in metres
#TODO need to time average Ld and load this in
if variable == 'eke' or variable == 'eke_parts':
    directory_ld = config['paths']['features'] + f'/{region}/'
    ld = xr.open_dataset(directory_ld + f'MINT_1d_00610101_00721230_Ld_{region}.nc')
elif variable == 'mke' or variable == 'res_eke':
    directory_ld = config['paths']['features'] + f'/{region}/coarsened_data/'
    ld = xr.open_dataset(directory_ld + f'MINT_1d_00610101_00721230_Ld_cg_{region}.nc')

# scale Ld (Ld is in km)
ld = ld_scaling * ld.Ld

if filter_lengthscale == 'ld':
    # get max Ld (scaling)
    Lmax = 1000*ld.max(dim=['x_c', 'y_c'], skipna=True).values

    # set kappa as a scale of deformation radius
    kappa = (1000*ld)**2 / Lmax**2

    # no tag on the output filename, so the original files are reproduced
    ls_tag = ''
    ls_attrs = {'filter_lengthscale': f'{ld_scaling}Ld (spatially varying)'}

elif filter_lengthscale == 'min_3ld_o25':
    # spatial average of 3Ld over the region, in metres
    ld_mean = 1000*float(ld.mean(dim=['x_c', 'y_c'], skipna=True))

    # grid spacing of the 1/4 degree (EXP4) coarse grid, taken as the square
    # root of the t-cell area so that it is a single length per cell, and
    # spatially averaged over wet points (e1t, e2t are already in metres)
    mesh_o25 = xr.open_dataset(directory_ld + f'mesh_mask_exp4_surface_{region}.nc') \
                 .isel(time_counter=0)
    wet_o25 = mesh_o25.tmask.isel(nav_lev=0)
    dels_o25 = np.sqrt(mesh_o25.e1t * mesh_o25.e2t).where(wet_o25)
    dels_o25_mean = float(dels_o25.mean(skipna=True))

    # constant filter scale: the smaller of the two spatial averages. Lmax is
    # the filter_scale passed to gcm-filters below
    Lmax = min(ld_mean, dels_o25_mean)

    # the filter scale does not vary in space, so kappa = 1 everywhere
    kappa = xr.ones_like(ld)

    logger.info('<%sLd> = %.1f m, <ds_O25> = %.1f m, filter scale = %.1f m',
                ld_scaling, ld_mean, dels_o25_mean, Lmax)

    # tag for the output filename and metadata recording the lengthscale
    ls_tag = f'_Lmin{ld_scaling}LdO25'
    ls_attrs = {
        'filter_lengthscale': f'min(<{ld_scaling}Ld>, <ds_O25>), <.> = spatial average, '
                              'ds_O25 = sqrt(e1t*e2t) of the 1/4 degree (EXP4) grid',
        'filter_scale_m': Lmax,
        f'mean_{ld_scaling}Ld_m': ld_mean,
        'mean_ds_O25_m': dels_o25_mean,
    }

else:
    raise ValueError(f'Unknown filter_lengthscale: {filter_lengthscale}')

# viscosities for the vector filter: isotropic contribution set by the
# filter lengthscale (on t-grid points)
kappa_iso = kappa
kappa_aniso = xr.zeros_like(kappa)
# -------------------------------------------- #

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
        nemo_files = [f'MINT_1d_{date_init}_*_ug_{region}.nc',
                      f'MINT_1d_{date_init}_*_vg_{region}.nc']
    else:
        if variable == 'eke' or variable == 'eke_parts':
            nemo_files = [f'MINT_1d_{date_init}_*_grid_U_{region}.nc',
                          f'MINT_1d_{date_init}_*_grid_V_{region}.nc']
        elif variable == 'mke' or variable == 'res_eke':
            nemo_files = [f'MINT_1d_{date_init}_*_uo_cg_{region}.nc',
                          f'MINT_1d_{date_init}_*_vo_cg_{region}.nc']
    print(nemo_files)

    nemo_paths = [glob.glob(directory + f) for f in nemo_files]

    logger.info('nemo_paths are: %s', nemo_paths)

    # extract date_end from nemo_paths
    filename = nemo_paths[0][0].split('/')[-1]
    date_end = filename.split('_')[3]

    nemo_files = [nemo_paths[0][0].split('/')[-1], nemo_paths[1][0].split('/')[-1]]

    logger.info('nemo_files is: %s', nemo_files)

    # open dataset using xnemogcm
    if variable == 'eke' or variable == 'eke_parts':
        nemo_paths = [directory + f for f in nemo_files]
        ds = open_nemo_and_domain_cfg(nemo_files=nemo_paths,
                                    domcfg_files=mask_path)
        
    elif variable == 'mke' or variable == 'res_eke':
        domcfg = open_domain_cfg(files = mask_path)
        dataU = xr.open_dataset(nemo_paths[0][0])
        dataV = xr.open_dataset(nemo_paths[1][0])

        ds = xr.merge([domcfg, dataU], compat='override')
        ds = xr.merge([ds, dataV], compat='override')
    
    logger.info('Dataset is: %s', ds)
    
    # Rechunk the dataset
    ds = ds.chunk(dict(y_c=-1))
    ds = ds.chunk(dict(x_f=-1))
    ds = ds.chunk(dict(y_f=-1))
    ds = ds.chunk(dict(x_c=-1))

    # Need west and south scale factor convention for irregular filter on tripolar grid
    dxw = ds.e1u.roll(x_f=-1, roll_coords=False)  # x-spacing centered at western T-cell edge in m
    dyw = ds.e2u.roll(x_f=-1, roll_coords=False)  # y-spacing centered at western T-cell edge in m
    dxs = ds.e1v.roll(y_f=-1, roll_coords=False)  # x-spacing centered at southern T-cell edge in m
    dys = ds.e2v.roll(y_f=-1, roll_coords=False)  # y-spacing centered at southern T-cell edge in m

    # Ensure that coordinates are the same
    if depth == 'full':
        wet_mask = ds.tmask
        wet_mask_f = ds.fmask
    else:
        wet_mask = ds.tmask.isel(z_c=0)
        wet_mask_f = ds.fmask.isel(z_c=0)
    dxw = dxw.swap_dims({"x_f": "x_c"})
    dyw = dyw.swap_dims({"x_f": "x_c"})
    dxs = dxs.swap_dims({"y_f": "y_c"})
    dys = dys.swap_dims({"y_f": "y_c"})
    area = ds.e1t * ds.e2t

    # U/V/F-point grid vars for the vector filter, all on common (y_c, x_c)
    # dims: gcm-filters works positionally and the VECTOR_C_GRID Laplacian
    # handles the C-grid staggering internally
    wet_mask_f = wet_mask_f.swap_dims({"x_f": "x_c", "y_f": "y_c"})
    dxCu = ds.e1u.swap_dims({"x_f": "x_c"})  # x-spacing centered at U point in m
    dyCu = ds.e2u.swap_dims({"x_f": "x_c"})  # y-spacing centered at U point in m
    dxCv = ds.e1v.swap_dims({"y_f": "y_c"})  # x-spacing centered at V point in m
    dyCv = ds.e2v.swap_dims({"y_f": "y_c"})  # y-spacing centered at V point in m
    dxBu = ds.e1f.swap_dims({"x_f": "x_c", "y_f": "y_c"})  # x-spacing centered at F point in m
    dyBu = ds.e2f.swap_dims({"x_f": "x_c", "y_f": "y_c"})  # y-spacing centered at F point in m

    # find minimum grid spacing
    dx_min = min(ds.e1t.where(ds.tmask.isel(z_c=0)).min(), \
                  ds.e2t.where(ds.tmask.isel(z_c=0)).min())
    dx_min = dx_min.values

    # set up the filters
    specs = {
            'filter_scale': Lmax,
            'dx_min': dx_min,
            'filter_shape': gcm_filters.FilterShape.GAUSSIAN
        }

    # scalar filter for the squared velocity field
    filter_irregular_with_land = gcm_filters.Filter(
        **specs,
        grid_type=gcm_filters.GridType.IRREGULAR_WITH_LAND,
        grid_vars={
            'wet_mask': wet_mask,
            'dxw': dxw, 'dyw': dyw, 'dxs': dxs, 'dys': dys, 'area': area,
            'kappa_w': kappa, 'kappa_s': kappa,
        }
    ) # need to vary kappa using the deformation radius.
    filter_irregular_with_land

    # viscosity-based vector filter for the velocity components, following
    # https://gcm-filters.readthedocs.io/en/latest/examples/example_vector_laplacian.html
    filter_vector_c_grid = gcm_filters.Filter(
        **specs,
        grid_type=gcm_filters.GridType.VECTOR_C_GRID,
        grid_vars={
            'wet_mask_t': wet_mask, 'wet_mask_q': wet_mask_f,
            'dxT': ds.e1t, 'dyT': ds.e2t,
            'dxCu': dxCu, 'dyCu': dyCu, 'area_u': dxCu * dyCu,
            'dxCv': dxCv, 'dyCv': dyCv, 'area_v': dxCv * dyCv,
            'dxBu': dxBu, 'dyBu': dyBu,
            'kappa_iso': kappa_iso, 'kappa_aniso': kappa_aniso,
        }
    )
    filter_vector_c_grid
    
    grid = xgcm.Grid(
                ds,
                metrics=get_metrics(ds),
    )
    
    # select velocity components on their native staggered grids
    logger.info('Selecting velocity components on native grids')
    if mode == 'geostrophic':
        uo = ds.ug
        vo = ds.vg
    else:
        if variable == 'eke' or variable == 'eke_parts':
            if depth == 'full':
                uo = ds.uo
                vo = ds.vo
            else:
                uo = ds.uo.isel(z_c=0)
                vo = ds.vo.isel(z_c=0)
        elif variable == 'mke' or variable == 'res_eke':
            uo = ds.uo
            vo = ds.vo

    # compute mean components of velocity by filtering the velocity vector
    # on its native staggered grids: swap to common (y_c, x_c) dims for
    # gcm-filters, then swap back to staggered dims for xgcm
    logger.info('Computing mean velocities using GCM-Filters vector filter')
    uoce_mean, voce_mean = filter_vector_c_grid.apply_to_vector(
                                                uo.swap_dims({"x_f": "x_c"}),
                                                vo.swap_dims({"y_f": "y_c"}),
                                                dims=['y_c', 'x_c'])
    uoce_mean = uoce_mean.drop_vars('x_c', errors='ignore') \
                         .swap_dims({"x_c": "x_f"}).persist()
    voce_mean = voce_mean.drop_vars('y_c', errors='ignore') \
                         .swap_dims({"y_c": "y_f"}).persist()

    # average the squared mean velocities onto t-grid points (NEMO KE convention)
    logger.info('Interpolating squared mean velocities to t-grid')
    uoce_mean_sq_c = grid.interp(uoce_mean**2, 'X', boundary='extend').persist()
    voce_mean_sq_c = grid.interp(voce_mean**2, 'Y', boundary='extend').persist()

    # following definition of EKE by Buzzicotti et al., (2023)
    if variable == 'eke' or variable == 'res_eke':
        # squared full velocities averaged onto t-grid points
        u_sq_c = grid.interp(uo**2, 'X', boundary='extend') \
               + grid.interp(vo**2, 'Y', boundary='extend')
        logger.info('Computing filtered velocity squared using GCM-Filters')
        u_sq_c_filt = filter_irregular_with_land.apply(u_sq_c,
                                                  dims=['y_c', 'x_c']).persist()

    # now compute eddy energy
    if variable == 'mke':
        coarse_ke = 0.5 * ( uoce_mean_sq_c
                        + voce_mean_sq_c )
    elif variable == 'eke':
        # coarse_ke = 0.5 * ( uoce_mean_sq_c + voce_mean_sq_c )
        fine_ke = 0.5 * ( u_sq_c_filt
                        - ( uoce_mean_sq_c + voce_mean_sq_c ) )
        fine_ke = fine_ke.where(fine_ke >= 0, 0) # clipping negative values
    elif variable == 'res_eke':
        # eddy energy resolved by the coarse-grained velocity field, used
        # elsewhere to form the subgrid eke as fine_ke - res_ke
        res_ke = 0.5 * ( u_sq_c_filt
                        - ( uoce_mean_sq_c + voce_mean_sq_c ) )
        res_ke = res_ke.where(res_ke >= 0, 0) # clipping negative values
    elif variable == 'eke_parts':
        # eddy velocity relative to the filtered mean flow, on t-grid points,
        # together with its kinetic energy. Squaring is done here on the fine
        # grid; coarse-graining all three with the same conservative weights
        # then gives the subgrid eke as kp_cg - 0.5 * ( up_cg**2 + vp_cg**2 ),
        # which is a variance of the coarse-graining operator and so is
        # positive by construction
        up = grid.interp(uo - uoce_mean, 'X', boundary='extend')
        vp = grid.interp(vo - voce_mean, 'Y', boundary='extend')
        kp = 0.5 * ( up**2 + vp**2 )

    # logger.info('bare_ke is: %s', bare_ke)
    if variable == 'mke':
        logger.info('coarse_ke is: %s', coarse_ke)
    elif variable == 'eke':
        logger.info('fine_ke is: %s', fine_ke)
    elif variable == 'res_eke':
        logger.info('res_ke is: %s', res_ke)
    elif variable == 'eke_parts':
        logger.info('kp is: %s', kp)

    # create dataset for xnemo readable
    data_vars = {}
    if variable == 'mke':
        data_vars['coarse_ke'] = (["t", "y_c", "x_c"], coarse_ke.values)
    elif variable == 'eke':
        # fine_ke keeps its name so downstream scripts (e.g. regrid_data.py)
        # can read it; the filter lengthscale used is recorded in its attrs
        if depth == 'full':
            data_vars['fine_ke'] = (["t", "z_c", "y_c", "x_c"], fine_ke.values, ls_attrs)
        else:
            data_vars['fine_ke'] = (["t", "y_c", "x_c"], fine_ke.values, ls_attrs)
            # data_vars['u_sq_c_filt'] = (["t", "y_c", "x_c"], u_sq_c_filt.values)
            # data_vars['coarse_ke'] = (["t", "y_c", "x_c"], coarse_ke.values)
    elif variable == 'res_eke':
        data_vars['res_eke'] = (["t", "y_c", "x_c"], res_ke.values)
    elif variable == 'eke_parts':
        data_vars['up'] = (["t", "y_c", "x_c"], up.values)
        data_vars['vp'] = (["t", "y_c", "x_c"], vp.values)
        data_vars['kp'] = (["t", "y_c", "x_c"], kp.values)

    ds_tmp = xr.Dataset(
        data_vars=data_vars,
        coords={
            "t": (["t"], ds.t.values,
                        ds.t.attrs),
            "gdept": (["z_c"], ds.gdept_1d.values,
                        ds.gdept_1d.attrs),
            "gphit": (["y_c", "x_c"], ds.gphit.values, 
                      {"standard_name": "Latitude", "units": "degrees_north"}),
            "glamt": (["y_c", "x_c"], ds.glamt.values, 
                      {"standard_name": "Longitude","units": "degrees_east"}),
        },
        attrs={
            "name": "NEMO dataset",
            "description": f"Contains {variable} from {mode} component -> ocean T grid variables",
            # record the filter lengthscale in the dataset metadata
            **ls_attrs,
        },
    )

    logger.info('ds_tmp is: %s', ds_tmp)

    # save to netcdf; ls_tag is empty for the 3Ld filter, and '_Lmin3LdO25'
    # for min(<3Ld>, <ds_O25>), e.g. MINT_1d_..._eke_Lmin3LdO25_SO_JET.nc
    if mode == 'geostrophic':
        output_file = f'MINT_1d_{date_init}_{date_end}_keg{ls_tag}_{region}.nc'
    else:
        if depth == 'full':
            output_file = f'MINT_1d_{date_init}_{date_end}_{variable}{ls_tag}_full_depth_{region}.nc'
        else:
            output_file = f'MINT_1d_{date_init}_{date_end}_{variable}{ls_tag}_{region}.nc'

    if variable == 'eke' or variable == 'eke_parts':
        save_directory = config['paths']['features'] + f'/{region}/'
    elif variable == 'mke' or variable == 'res_eke':
        save_directory = config['paths']['features'] + f'/{region}/coarsened_data/'
    logger.info('Saving kinetic energy to: %s', save_directory + output_file)

    ds_tmp.to_netcdf(save_directory + output_file)

logger.info('End!')