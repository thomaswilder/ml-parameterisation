#!/usr/bin/env python3

'''
    Description: Row B. The cell centres and cell corners of the regional 1/16 and
        1/4 degree grids, in the form xesmf takes for conservative regridding.

    Method: The T points inside the region's lat/lon box (regions.<region>.lat, .lon
        in the config) are found on each global mask. Centres are the T points of
        that index range; corners are the F points around them. The index range of
        each grid is printed: it is the range that rows A and 1 give to `ncks -d`
        (regions.<region>.fine and .coarse in the config).

    Reads:  masks.coarse, masks.fine (the two global masks)
    Writes: {paths.features}/<region>/<region>_target_grid_025.npy
            {paths.features}/<region>/<region>_input_grid_16.npy
            Each holds a dict with the keys 'lon', 'lat' (ny, nx) and
            'lon_b', 'lat_b' (ny+1, nx+1).

    From the notebook coarsening/get_mask_bounds_for_conservative_regridding.ipynb.
'''

import argparse
import sys

import numpy as np
import xarray as xr
import yaml


def get_nemo_grid_for_xesmf(mesh_mask, lat_min, lat_max, lon_min, lon_max):
    """
    Extract an xesmf-compatible grid dict from a NEMO mesh mask.

    Cell centers from T-points; cell corners from F-points,
    giving lon_b/lat_b with shape (ny+1, nx+1) as required
    by xesmf for conservative regridding.

    Parameters
    ----------
    mesh_mask : xr.Dataset
        NEMO mesh mask containing glamt, gphit, glamf, gphif.
    lat_min, lat_max : float
        Latitude bounds for selecting T-points.
    lon_min, lon_max : float
        Longitude bounds for selecting T-points.

    Returns
    -------
    dict with keys 'lon', 'lat' (ny, nx) and 'lon_b', 'lat_b' (ny+1, nx+1),
    and the index ranges {'x': [imin, imax], 'y': [jmin, jmax]} (zero-based, inclusive)
    """
    glamt = mesh_mask.glamt.squeeze()
    gphit = mesh_mask.gphit.squeeze()
    glamf = mesh_mask.glamf.squeeze()
    gphif = mesh_mask.gphif.squeeze()

    # Find T-points within lat/lon bounds
    condition = ((gphit >= lat_min) & (gphit <= lat_max) &
                 (glamt >= lon_min) & (glamt <= lon_max))
    idx = np.argwhere(condition.values)
    jmin, imin = idx[:, 0].min(), idx[:, 1].min()
    jmax, imax = idx[:, 0].max(), idx[:, 1].max()

    # T-point centers: shape (ny, nx)
    lon = glamt.values[jmin:jmax+1, imin:imax+1]
    lat = gphit.values[jmin:jmax+1, imin:imax+1]

    # F-point corners: shape (ny+1, nx+1)
    # F(j,i) is the NE corner of T-cell (j,i), so the SW corner
    # of T-cell (jmin, imin) is F(jmin-1, imin-1)
    lon_b = glamf.values[jmin-1:jmax+1, imin-1:imax+1]
    lat_b = gphif.values[jmin-1:jmax+1, imin-1:imax+1]

    print(f"T-point index range: j=[{jmin},{jmax}], i=[{imin},{imax}]")
    print(f"  lon, lat shape:   {lon.shape}")
    print(f"  lon_b, lat_b shape: {lon_b.shape}")

    index_range = {'x': [int(imin), int(imax)], 'y': [int(jmin), int(jmax)]}
    return {'lon': lon, 'lat': lat, 'lon_b': lon_b, 'lat_b': lat_b}, index_range


parser = argparse.ArgumentParser(
    description='Cell centres and corners of the regional 1/16 and 1/4 degree grids, for xesmf.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region
bounds = config['regions'][region]
lat_min, lat_max = bounds['lat']
lon_min, lon_max = bounds['lon']

directory = config['paths']['features'] + f'/{region}/'

# the 1/4 degree grid: the target of the regridding
mask025 = xr.open_dataset(config['masks']['coarse'])
target_grid, coarse_range = get_nemo_grid_for_xesmf(
    mask025, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
np.save(directory + f'{region}_target_grid_025.npy', target_grid)
print(f'Saved to {directory}{region}_target_grid_025.npy')

# the 1/16 degree grid: the input of the regridding
mask16 = xr.open_dataset(config['masks']['fine'])
input_grid, fine_range = get_nemo_grid_for_xesmf(
    mask16, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
np.save(directory + f'{region}_input_grid_16.npy', input_grid)
print(f'Saved to {directory}{region}_input_grid_16.npy')

# The masks and subsets of rows A and 1 are cut with the index ranges in the config.
# They must be the ranges found here, or the grids and the data do not line up.
mismatch = False
for name, found in (('coarse', coarse_range), ('fine', fine_range)):
    in_config = {k: list(v) for k, v in bounds[name].items()}
    if found == in_config:
        print(f'index range of the {name} grid {found} = regions.{region}.{name} in the config')
    else:
        print(f'ERROR: index range of the {name} grid {found} != regions.{region}.{name} '
              f'in the config {in_config}', file=sys.stderr)
        mismatch = True
if mismatch:
    sys.exit(1)
