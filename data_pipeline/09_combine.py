#!/usr/bin/env python3

'''
    Description: Rows 14 and 15. Joins the monthly files of one training variable
        into one file for the whole period.

    Method: The monthly files from --start to --end are opened together and joined
        along t, in order, and written as they are.

    Reads:  {paths.features}/<region>/coarsened_data/MINT_1d_<d0>_<d1>_<variable>_<region>.nc
            for every month of the period
    Writes: {paths.features}/<region>/coarsened_data/MINT_1d_<first year>-<last year>_<variable>_<region>.nc
            (MINT_1d_0061-0072_... for the released data). For --variable vor_cg the
            name ends in _mod: MINT_1d_<first year>-<last year>_vor_cg_<region>_mod.nc

    --variable  the part of the file name that names the variable:
                vor_cg             relative vorticity (07_vorticity.py)
                mke                kinetic energy of the filtered coarse velocity (08_kinetic_energy.py)
                sa_cg              deformation radius over the grid scale (06_scale_aware_feature.py)
                eke_Lmin3LdO25_cg  the target, eddy kinetic energy (08_kinetic_energy.py, 03_regrid.py)

    The vorticity file keeps the name of the released file, which ends in _mod. That
    file was made in two passes: the monthly vorticity files then had no time axis,
    and a second pass gave the joined file the time axis of another product. The
    monthly files of 07_vorticity.py now carry the time axis of the velocities, so the
    file joined here is right as it is; the name stays because the trained models
    were configured with it.

    Each month's file is found by its first and last day (360-day calendar), never by
    a pattern: a pattern for the monthly files also matches this script's output.

    From the notebooks machine_learning/combine_coarsened_datasets.ipynb and
    machine_learning/process_pandas_indexing.ipynb.
'''

import argparse
from datetime import datetime

import xarray as xr
import yaml
from dateutil.relativedelta import relativedelta

parser = argparse.ArgumentParser(description='Join the monthly files of one training variable.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--start', required=True,
                    help='first month of the period, YYYYMMDD')
parser.add_argument('--end', required=True,
                    help='end of the period, YYYYMMDD, exclusive')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
parser.add_argument('--variable', required=True,
                    choices=('vor_cg', 'mke', 'sa_cg', 'eke_Lmin3LdO25_cg'),
                    help='the part of the file name that names the variable')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region
variable = args.variable

directory = config['paths']['features'] + f'/{region}/coarsened_data/'

# the months of the period, by their first and last day: the model has a 360-day
# calendar, so every month ends on day 30
months = []
current_date_init = datetime.strptime(args.start, "%Y%m%d")
end_date_init = datetime.strptime(args.end, "%Y%m%d")
while current_date_init < end_date_init:
    date_init = (f"{current_date_init.year:04d}"
                 f"{current_date_init.month:02d}"
                 f"{current_date_init.day:02d}")
    months.append((date_init, f"{date_init[:6]}30"))
    current_date_init = current_date_init + relativedelta(months=+1)

files = [directory + f"MINT_1d_{d0}_{d1}_{variable}_{region}.nc" for d0, d1 in months]
print(f'{len(files)} monthly files, {files[0]} to {files[-1]}')

ds = xr.open_mfdataset(files,
                       concat_dim='t',
                       combine='nested')

years = f"{months[0][0][:4]}-{months[-1][1][:4]}"
suffix = '_mod' if variable == 'vor_cg' else ''
output_file = directory + f"MINT_1d_{years}_{variable}_{region}{suffix}.nc"

ds.to_netcdf(output_file)
print(f'Saved to {output_file}')
