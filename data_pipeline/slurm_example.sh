#!/bin/bash
#
# An example batch job: one step of the data pipeline, for one region, over the whole
# period. Copy it, set the placeholders, pick the step at the end, and submit it with
#
#     sbatch slurm_example.sh
#
# The variables and the command are those of README.md, "The commands for one region".
# Steps in environment X (00_regrid_weights.py, 00_filter_kappa.py, 03_regrid.py) need
# ENV_X in place of ENV_P; 00_regional_masks.sh and 01_subset.sh need ncks (environment N).
#
# What the saved job scripts of the original runs asked for. The saved script of a step
# is the one of its last run, not necessarily the one that made a given file, and the
# python job scripts do not say how many months a job ran:
#   08_kinetic_energy.py:      2 hours, 8 GB
#   03_regrid.py:              2 hours, 5 GB
#   04_to_cdftools_layout.py:  4 hours, 10 GB
#   05_deformation_radius.py:  30 minutes, 20 GB, on the debug partition and qos
#   07_vorticity.py:           two saved scripts: the older (15 January 2026) asks for
#                              12 hours and 20 GB, the newer (14 September 2026) for
#                              1 hour and 10 GB
# The saved job script of the filtering directory has its call of the script of
# 02_filter.py commented out, so no request for that step is recorded.
#
# cdfbn2 (row 6) is the one step whose jobs are on record, from the scheduler
# (sacct), all three asking for 5 GB and one cdfbn2 call per month for SO_JET:
#   fine grid, 144 months:   18 hours asked, 8 h 19 min used (about 3.5 min a month)
#   coarse grid, 144 months: 4 hours asked, stopped by the time limit after 134 months
#                            (about 1.8 min a month); the last 10 months were run in a
#                            second job, 2 hours asked, 21 min used (about 2 min a month)
# The 2 hours of the saved cdfbn2 script belong to that second job (10 months); it was
# written after the two long jobs and is no guide for 144 months. For 144 months ask
# for about 6 hours for the coarse grid and 12 hours for the fine grid, or run the
# months in pieces with --start and --end.
#
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --job-name=eke_SO_JET
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH -o %j.out
#SBATCH -e %j.err

set -eo pipefail

R=SO_JET                              # the region: SO_JET, SO_EAST, SO_GYRE or SO_TROP
START=00610101                        # the first month (day 01)
END=00730101                          # the end of the period, exclusive: 00730101 gives all 144 months
CONFIG=/path/to/config.yml            # your copy of config.example.yml
PIPE=/path/to/data_pipeline           # this directory
WORK=/path/to/work                    # the working directory: each script writes its .log here
ENV_P=/path/to/pyCDFTOOLS             # conda environment P (see envs/environment.yml)

source /path/to/miniconda3/bin/activate
conda activate "$ENV_P"
cd "$WORK"

# row 12, the target on the 1/16 degree grid
python "$PIPE/08_kinetic_energy.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable eke
