#!/usr/bin/env python3

'''
    Description: Row 8. The time mean of the deformation radius over a period, on
        the 1/4 degree grid (--mode cg) or on the 1/16 degree grid (--mode non_cg).
        08_kinetic_energy.py reads both: the coarse one for mke, the fine one for eke.

    Method: The monthly files of 05_deformation_radius.py from --start to --end are
        joined along t and averaged over t.

    Reads:  --mode cg      {paths.features}/<region>/coarsened_data/MINT_1d_<d0>_<d1>_Ld_cg_<region>.nc
            --mode non_cg  {paths.features}/<region>/MINT_1d_<d0>_<d1>_Ld_<region>.nc
            for every month of the period
    Writes: the same directory, MINT_1d_<first day>_<last day>_Ld[_cg]_<region>.nc,
            named by the first day and the last day of the period
            (MINT_1d_00610101_00721230_... for the released data)

    Each month's file is found by its first and last day (360-day calendar), never by
    a pattern: a pattern for the monthly files also matches this script's output.

    From the notebook misc/time_average_deformation_radius.ipynb.
'''

import argparse
from datetime import datetime

import xarray as xr
import yaml
from dateutil.relativedelta import relativedelta

parser = argparse.ArgumentParser(description='Time mean of the deformation radius over a period.')
parser.add_argument('--region', required=True,
                    help='region name, a key of `regions` in the config, e.g. SO_JET')
parser.add_argument('--start', required=True,
                    help='first month of the period, YYYYMMDD')
parser.add_argument('--end', required=True,
                    help='end of the period, YYYYMMDD, exclusive')
parser.add_argument('--config', required=True,
                    help='paths config (see config.example.yml)')
parser.add_argument('--mode', required=True, choices=('cg', 'non_cg'),
                    help="'cg': the 1/4 degree grid; 'non_cg': the 1/16 degree grid")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

region = args.region

if args.mode == 'cg':
    directory = config['paths']['features'] + f'/{region}/coarsened_data/'
    name = f'Ld_cg_{region}'
else:
    directory = config['paths']['features'] + f'/{region}/'
    name = f'Ld_{region}'

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

files = [directory + f"MINT_1d_{d0}_{d1}_{name}.nc" for d0, d1 in months]
print(f'{len(files)} monthly files, {files[0]} to {files[-1]}')

ds = xr.open_mfdataset(files,
                       concat_dim='t',
                       combine='nested',
                       chunks={'t': 100})

Ld_mean = ds.Ld.mean(dim='t')

output_file = directory + f"MINT_1d_{months[0][0]}_{months[-1][1]}_{name}.nc"
Ld_mean.to_netcdf(output_file)
print(f'Saved to {output_file}')
