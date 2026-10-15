# Data pipeline: reference

The detail behind the [front page](../README.md): the regions, the full command sequence, what each
step reads and writes, the options, the oddities kept on purpose, and how the scripts were
checked. File names such as `envs/…`, `cdftools/…` and `config.example.yml` are relative to
`data_pipeline/`, the directory above this one (the commands call it `$PIPE`).

The scripts that made the training data of the models in this repository, from the
DINO model output to the five files per region that the models read. They are
published so that a reader can see, and repeat, how those files were made.

The training data themselves are on Zenodo, five files per region for the four regions
`SO_JET`, `SO_EAST`, `SO_GYRE` and `SO_TROP`. **The files on Zenodo are the reference.**
They were not made again for the release: the scripts here were checked against them
(see "How this was checked").

| File | Variable | What it holds |
|---|---|---|
| `MINT_1d_0061-0072_vor_cg_{region}_mod.nc` | `vor` | relative vorticity of the coarse-grained velocity |
| `MINT_1d_0061-0072_mke_{region}.nc` | `coarse_ke` | kinetic energy of the filtered coarse-grained velocity |
| `MINT_1d_0061-0072_sa_cg_{region}.nc` | `sa` | deformation radius over the grid scale |
| `MINT_1d_0061-0072_eke_Lmin3LdO25_cg_{region}.nc` | `fine_ke` | the target: eddy kinetic energy below the filter scale |
| `mesh_mask_exp4_{region}_xnemo.nc` | mask, grid | the 1/4 degree mesh mask with xnemogcm dimension names |

Each of the four combined files holds 4320 daily fields: 144 months of 30 days, model
years 0061 to 0072.

## What is needed, and is not on Zenodo

- **The global model output** of DINO at 1/16 degree (experiment EXP16): daily means,
  one file per month and grid, `MINT_1d_<first day>_<last day>_grid_{T,U,V}.nc`, for the
  144 months from `00610101` to `00721230`. That is 432 files, about 5.9 TB
  (`grid_T` 22 GB a month, `grid_U` and `grid_V` 11 GB each).
- **The two global mesh masks**: `mesh_mask_exp16.nc` (1/16 degree, 7.7 GB) and
  `mesh_mask_025.nc` (the 1/4 degree grid of DINO experiment EXP4, 0.5 GB).
- **Software**: the two Python environments under "Environments", `ncks` from NCO, and
  `cdfbn2` from CDFTOOLS.

## Environments

Each step was run in one of four environments. The checks used two conda environments, P
and X, because neither has everything (P has no xesmf, X has no gcm_filters). Their
`conda env export` files, `envs/pyCDFTOOLS.yml` and `envs/regrid_xesmf_cond.yml`, were later
replaced by the single, untested [`envs/environment.yml`](../envs/environment.yml) and are in
the git history at commit `f9bf780`. The versions the checks ran with are in the table.

| | Environment | Used by |
|---|---|---|
| P | `pyCDFTOOLS`: Python 3.12.1, xarray 2023.12.0, numpy 1.26.3, dask 2023.12.1, netCDF4 1.6.5, xgcm 0.8.1, xnemogcm 0.4.2, gcm_filters 0.5.1 | `00_regrid_grids.py`, `00_xnemo_mask.py`, `02_filter.py`, `04_to_cdftools_layout.py`, `05_deformation_radius.py`, `05_time_mean_ld.py`, `06_scale_aware_feature.py`, `07_vorticity.py`, `08_kinetic_energy.py`, `09_combine.py` |
| X | `regrid_xesmf_cond`: Python 3.10.13, xarray 2023.12.0, numpy 1.26.2, dask 2023.12.0, netCDF4 1.6.5, xgcm 0.8.1, xnemogcm 0.4.2, xesmf 0.8.2 with ESMF 8.4.2 | `00_regrid_weights.py`, `00_filter_kappa.py`, `03_regrid.py` (everything that imports xesmf) |
| N | `ncks` (NCO 5.3.3 made the released files) and a `python3` with PyYAML, which the two shell scripts use to read the config. On JASMIN both come with `module load jaspy` | `00_regional_masks.sh`, `01_subset.sh` |
| C | `cdfbn2` of CDFTOOLS v4.0.0-61-g9498c03, built with gfortran against netCDF; see [`envs/other_tools.md`](../envs/other_tools.md) | row 6 |

xesmf imports only after a full `conda activate` of environment X; calling
`<env>/bin/python` directly is not enough. xarray 2025.12.0 was found to fail on
`ds.chunk` for these datasets.

## Configuration

Every path comes from one YAML file. Copy [`config.example.yml`](../config.example.yml), set the four `paths` and
the two `masks`, and pass the copy as `--config`. The directory layout below each root,
and every file name, are fixed by the scripts:

```
{paths.model_output}/MINT_1d_<d0>_<d1>_grid_{T,U,V}.nc               the global model output
{paths.subsets}/<region>/MINT_1d_<d0>_<d1>_grid_{T,U,V}_<region>.nc  regional subsets (row 1)
{paths.kappa}/kappa16_<region>.nc                                    the filter-scale field (row D)
{paths.features}/<region>/                                           masks, grids, weights, fine-grid bn2, Ld and eke
{paths.features}/<region>/filtered_data/                             filtered fields (rows 2, 3)
{paths.features}/<region>/coarsened_data/                            everything on the 1/4 degree grid, and the training files
```

`masks.fine` and `masks.coarse` are the two global masks. `regions` holds, for each
region, the lat/lon box that selects its T points (`lat`, `lon`) and the zero-based,
inclusive index ranges of that box on each grid (`fine`, `coarse`), as `ncks -d` takes
them.

The scripts do not create directories. The commands below make the ones they write to;
`paths.kappa` must exist.

**Do not point `paths.features` (or `paths.subsets`) at a directory that holds the
Zenodo files.** `09_combine.py` writes the same names, `MINT_1d_0061-0072_*` in
`{paths.features}/<region>/coarsened_data/`, and `00_xnemo_mask.py` writes
`mesh_mask_exp4_<region>_xnemo.nc` there too. The Python scripts overwrite without
asking, so a run over the Zenodo files replaces them with files that are close to them
and not identical (see "How this was checked"). Give a run its own empty directories.

## Calendar and file names

The model has a 360-day calendar: every month runs from day 01 to day 30, and every
monthly file is named `MINT_1d_<first day>_<last day>_…` (`MINT_1d_00610101_00610130_…`
for January 0061). Each script finds a month's file by those two dates, never by a
pattern, because a pattern such as `MINT_1d_00610101_*_Ld_cg_…` also matches the
full-period time mean `MINT_1d_00610101_00721230_Ld_cg_…`. So `--start` must be day 01
of a month. `--end` is exclusive, and it too must be day 01 of a month: `--start
00610101 --end 00730101` is all 144 months, `--start 00610101 --end 00610201` is January
0061 alone. The Python scripts and `01_subset.sh` compare the whole date, so they also
process the month that contains an `--end` after day 01 (`--end 00610215` gives January
and February), while the `months` function of the `cdfbn2` loop (row 6) compares only
the year and the month and gives January alone, so the rows would not run over the same
months. In the combined files `t` is in days since 0061-01-02.

## Regions

| Region | lat, lon box | Fine grid `x`, `y` | Fine size (y × x) | Coarse grid `x`, `y` | Coarse size |
|---|---|---|---|---|---|
| `SO_JET` | −47.5 to −37.5, 2.5 to 17.5 | 41–280, 726–942 | 217 × 240 | 11–70, 182–235 | 54 × 60 |
| `SO_EAST` | −50 to −40, 32.5 to 47.5 | 521–760, 665–891 | 227 × 240 | 131–190, 167–223 | 57 × 60 |
| `SO_GYRE` | −65 to −58, 18 to 32 | 289–512, 210–445 | 236 × 224 | 73–128, 53–111 | 59 × 56 |
| `SO_TROP` | −25 to −12.5, 18 to 32 | 289–512, 1178–1389 | 212 × 224 | 73–128, 295–347 | 53 × 56 |

`00_regrid_grids.py` finds the index ranges from the lat/lon box and exits with status 1
if they are not the ranges in the config, because the masks and subsets of rows A and 1
are cut with the config's ranges and would then not line up with the regridding grids.

## The flow

Each item is one step (a "row"); its letter or number is used in the rest of this file.

Once per region:

- **A** the two global masks → the regional masks of both grids (`ncks`)
- **B** the two global masks → the cell centres and corners of both regional grids, for xesmf
- **C** B → the conservative regridding weights from the 1/16 to the 1/4 degree grid
- **D** A, B → kappa, the field that sets the spatially varying filter scale of rows 2 and 3
- **E** A → the 1/4 degree mask with xnemogcm dimension names: **a training file**

For every month:

- **1** the global output → the regional subsets of `grid_T` (every level) and `grid_U`, `grid_V` (the first level) (`ncks`)
- **2** 1, A, D → temperature and salinity filtered on the 1/16 degree grid
- **3** 1, A, D → the surface velocity filtered, as a vector, on the 1/16 degree grid
- **4** 2, 3, A, B, C → those four fields coarse-grained to the 1/4 degree grid
- **5** 4, 1 → the coarse temperature and salinity in the layout `cdfbn2` reads
- **6** 5 → N² on the 1/4 degree grid; 1 → N² on the 1/16 degree grid (`cdfbn2`)
- **7** 6, A → the deformation radius Ld on both grids

Over all 144 months:

- **8** every month of 7 → the time mean of Ld on each grid

For every month:

- **9** 7 (1/4 degree), A → sa, Ld over the grid scale
- **10** 4 (velocity), A → the relative vorticity
- **11** 4 (velocity), 8 (1/4 degree), A → mke, the kinetic energy of the velocity filtered at 3Ld
- **12** 1 (velocity), 8 (1/16 degree), A → eke on the 1/16 degree grid, below the filter scale min(<3Ld>, <Δs_O25>)
- **13** 12, A, B, C → eke coarse-grained to the 1/4 degree grid

Over the whole period:

- **14** every month of 9, 10, 11 and 13 → **the four other training files**

So for the full period rows 1 to 7 must have run for every month before rows 11 and 12,
which read the time means of row 8.

## The commands for one region

Set these first. Every command below is run from `$WORK`, in one bash shell. `START` and
`END` are both the first day of a month (see "Calendar and file names"): run exactly the
commands below with `END` at day 01, as the `months` loop of row 6 reads only the year
and the month of it.
`02_filter.py`, `03_regrid.py`, `04_to_cdftools_layout.py`, `05_deformation_radius.py`,
`07_vorticity.py` and `08_kinetic_energy.py` each write a `.log` file into the current
directory.

<!-- chain: setup -->
```bash
R=SO_JET                              # the region: SO_JET, SO_EAST, SO_GYRE or SO_TROP
START=00610101                        # the first month (day 01)
END=00730101                          # the end of the period, exclusive: 00730101 gives all 144 months
CONFIG=/path/to/config.yml            # your copy of config.example.yml
PIPE=/path/to/data_pipeline           # this directory
WORK=/path/to/work                    # an empty directory: the logs and the cdfbn2 namelists
CDFBN2=/path/to/CDFTOOLS/bin/cdfbn2
ENV_P=/path/to/pyCDFTOOLS             # conda environment P (envs/pyCDFTOOLS.yml)
ENV_X=/path/to/regrid_xesmf_cond      # conda environment X (envs/regrid_xesmf_cond.yml)
HANG_LIMIT=900                        # seconds; see again_if_hung below
source /path/to/miniconda3/bin/activate   # so that `conda activate` works in this shell
```

Then the environments, as shell functions that run one command in a subshell, and
`cfg`, which reads one value of the config (it is the function that
`00_regional_masks.sh` and `01_subset.sh` use). `SUBSETS` and `FEATURES`, which the
commands below need, are `paths.subsets` and `paths.features` of the config, read with it:

<!-- chain: run -->
```bash
in_N() { ( module load jaspy && "$@" ); }          # N; elsewhere: any shell with ncks and python3 + PyYAML
in_P() { ( conda activate "$ENV_P" && "$@" ); }     # P
in_X() { ( conda activate "$ENV_X" && "$@" ); }     # X

# 05_time_mean_ld.py and 09_combine.py can hang now and then (all threads waiting on a
# lock while xarray reads and writes netCDF at the same time). Run them with a time
# limit, and again if it is reached; a run that ends gives the same file every time.
again_if_hung() {
    local attempt status
    for attempt in 1 2 3; do
        status=0
        timeout "$HANG_LIMIT" "$@" || status=$?
        [ "$status" -ne 124 ] && return "$status"
        echo "again_if_hung: no exit after $HANG_LIMIT s, killed; running it again: $*" >&2
    done
    return 124
}

# the first day of every month from $1 to $2 (exclusive), YYYYMMDD
months() {
    local y=$((10#${1:0:4})) m=$((10#${1:4:2}))
    while (( 10#$(printf '%04d%02d' "$y" "$m") < 10#${2:0:6} )); do
        printf '%04d%02d01\n' "$y" "$m"
        m=$((m + 1))
        if (( m == 13 )); then m=1; y=$((y + 1)); fi
    done
}

# one value of the config, e.g. `cfg paths.features`; a list is printed as "a,b"
cfg() {
    python3 - "$CONFIG" "$1" <<'EOF'
import sys
import yaml
with open(sys.argv[1]) as f:
    value = yaml.safe_load(f)
for key in sys.argv[2].split('.'):
    value = value[key]
print(','.join(map(str, value)) if isinstance(value, list) else value)
EOF
}

SUBSETS="$(in_N cfg paths.subsets)"       # paths.subsets of $CONFIG
FEATURES="$(in_N cfg paths.features)"     # paths.features of $CONFIG
```

The output directories (the scripts do not create them):

<!-- chain: run -->
```bash
cd "$WORK"
mkdir -p "$SUBSETS/$R" "$FEATURES/$R/filtered_data" "$FEATURES/$R/coarsened_data" "$WORK/bn2_cg" "$WORK/bn2_fine"
```

**Once per region** (rows A to E):

<!-- chain: run -->
```bash
# A  the regional masks, cut from the two global masks with ncks                     [N]
in_N bash "$PIPE/00_regional_masks.sh" --region "$R" --config "$CONFIG"
# B  cell centres and corners of both grids for xesmf, and the check of the ranges   [P]
in_P python "$PIPE/00_regrid_grids.py" --region "$R" --config "$CONFIG"
# C  the conservative regridding weights, 1/16 to 1/4 degree                         [X]
in_X python "$PIPE/00_regrid_weights.py" --region "$R" --config "$CONFIG"
# D  kappa, the field that sets the filter scale of rows 2 and 3                     [X]
in_X python "$PIPE/00_filter_kappa.py" --region "$R" --config "$CONFIG"
# E  the 1/4 degree mask with xnemogcm dimension names (a training file)             [P]
in_P python "$PIPE/00_xnemo_mask.py" --region "$R" --config "$CONFIG"
```

**Month by month** (rows 1 to 7):

<!-- chain: run -->
```bash
# 1  the regional subsets of the global output, with ncks                            [N]
in_N bash "$PIPE/01_subset.sh" --region "$R" --start "$START" --end "$END" --config "$CONFIG"
# 2  filter temperature and salinity on the 1/16 degree grid                         [P]
in_P python "$PIPE/02_filter.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable grid_T --variable-name thetao
in_P python "$PIPE/02_filter.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable grid_T --variable-name so
# 3  filter the surface velocity, as a vector, on the 1/16 degree grid               [P]
in_P python "$PIPE/02_filter.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable grid_UV
# 4  coarse-grain the filtered fields to the 1/4 degree grid                         [X]
in_X python "$PIPE/03_regrid.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable thetao
in_X python "$PIPE/03_regrid.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable so
in_X python "$PIPE/03_regrid.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable uo
in_X python "$PIPE/03_regrid.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable vo
# 5  the coarse temperature and salinity in the layout cdfbn2 reads                  [P]
in_P python "$PIPE/04_to_cdftools_layout.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG"
```

Row 6 is CDFTOOLS' `cdfbn2`, once on the coarse-grained T and S and once on the
unfiltered fine-grid `grid_T` subset. `cdfbn2` reads its namelist, `nam_cdf_names`,
from the current directory, so each grid has its own directory. The two namelists in
[`cdftools/`](../cdftools/) are the ones that were used; the only change is that `CN_FZGR` (the file
`cdfbn2` takes the vertical grid from) is a placeholder, set here to the region's mask.
The two differ from each other only in `CN_Z` (`deptht` for the coarse grid, `depth` for
the fine one). With `-full` only the 1-D vertical grid is read from the `CN_FZGR` file;
the fine-grid N² came out bit-identical with either of the two namelists.

<!-- chain: run -->
```bash
# 6  N2 with cdfbn2, on the 1/4 and the 1/16 degree grid                             [C]
sed "s|@CN_FZGR@|$FEATURES/$R/mesh_mask_exp4_$R.nc|" "$PIPE/cdftools/nam_cdf_names_cg" > "$WORK/bn2_cg/nam_cdf_names"
sed "s|@CN_FZGR@|$FEATURES/$R/mesh_mask_exp16_$R.nc|" "$PIPE/cdftools/nam_cdf_names_fine" > "$WORK/bn2_fine/nam_cdf_names"
for d0 in $(months "$START" "$END"); do
    m="MINT_1d_${d0}_${d0:0:6}30"
    ( cd "$WORK/bn2_cg" && "$CDFBN2" -t "$FEATURES/$R/coarsened_data/${m}_thetao_cg_${R}_cdftools.nc" -s "$FEATURES/$R/coarsened_data/${m}_so_cg_${R}_cdftools.nc" -nc4 -teos10 -full -o "$FEATURES/$R/coarsened_data/${m}_bn2_cg_${R}.nc" )
    ( cd "$WORK/bn2_fine" && "$CDFBN2" -t "$SUBSETS/$R/${m}_grid_T_${R}.nc" -nc4 -teos10 -full -o "$FEATURES/$R/${m}_bn2_${R}.nc" )
done
# 7  the deformation radius on the 1/4 and the 1/16 degree grid                      [P]
in_P python "$PIPE/05_deformation_radius.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --mode cg
in_P python "$PIPE/05_deformation_radius.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --mode non_cg
```

**Over all 144 months** (row 8), once rows 1 to 7 have run for every month. The period
is written out, not `$START` and `$END`, because `08_kinetic_energy.py` opens the two
time means by the fixed names `MINT_1d_00610101_00721230_Ld_cg_<region>.nc` and
`MINT_1d_00610101_00721230_Ld_<region>.nc`, and `05_time_mean_ld.py` names its output by
the first and last day of its period:

<!-- chain: full-period only -->
```bash
# 8  the time means of the deformation radius over 0061 to 0072                      [P]
in_P again_if_hung python "$PIPE/05_time_mean_ld.py" --region "$R" --start 00610101 --end 00730101 --config "$CONFIG" --mode cg
in_P again_if_hung python "$PIPE/05_time_mean_ld.py" --region "$R" --start 00610101 --end 00730101 --config "$CONFIG" --mode non_cg
```

**Month by month** again (rows 9 to 13):

<!-- chain: run -->
```bash
# 9  the scale-aware feature sa = Ld / grid scale                                    [P]
in_P python "$PIPE/06_scale_aware_feature.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG"
# 10 the relative vorticity of the coarse-grained velocity                           [P]
in_P python "$PIPE/07_vorticity.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG"
# 11 mke: kinetic energy of the coarse-grained velocity filtered at 3Ld              [P]
in_P python "$PIPE/08_kinetic_energy.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable mke
# 12 eke: the eddy kinetic energy on the 1/16 degree grid, the target                [P]
in_P python "$PIPE/08_kinetic_energy.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable eke
# 13 coarse-grain the target                                                         [X]
in_X python "$PIPE/03_regrid.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable eke_Lmin3LdO25
```

**The training files** (row 14). For all 144 months they are
`MINT_1d_0061-0072_…`; the name gives the first and the last year of the period:

<!-- chain: run -->
```bash
# 14 join the monthly files                                                          [P]
in_P again_if_hung python "$PIPE/09_combine.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable vor_cg
in_P again_if_hung python "$PIPE/09_combine.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable mke
in_P again_if_hung python "$PIPE/09_combine.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable sa_cg
in_P again_if_hung python "$PIPE/09_combine.py" --region "$R" --start "$START" --end "$END" --config "$CONFIG" --variable eke_Lmin3LdO25_cg
```

For scale: one month of one region takes seconds to a few minutes per step, and `cdfbn2`
is the slowest. For one month of `SO_JET` the check runs of row 6 took 135 s on the coarse
grid and 225 s on the fine grid, run one after the other with other jobs on the same host,
and in the two chain runs rows 6 and 7 together (both `cdfbn2` calls and the deformation
radius) took 169 s and 153 s. For 144 months the batch jobs of the original runs took,
for `cdfbn2`, 8 h 19 min on the fine grid (about 3.5 min a month), and on the coarse grid
about 2 min a month (see [`slurm_example.sh`](../slurm_example.sh)); the `eke` of one region took about
9 minutes and the filtering of `thetao` about 30. [`slurm_example.sh`](../slurm_example.sh) runs one step as a
batch job.

## The `ncks` commands of rows A and 1

Rows A and 1 are `ncks` commands (NCO). `00_regional_masks.sh` and `01_subset.sh` run
exactly the commands below, and print each one before they run it. They are written out
here so that they can be read, and repeated by hand. They take the index ranges of the
region from the config: `fine_x`, `fine_y`, `coarse_x` and `coarse_y` are the "Fine grid"
and "Coarse grid" ranges of the table under "Regions", written `first,last` (for `SO_JET`,
`fine_x` is `41,280`). The ranges are zero-based and inclusive. The commands have no
`-O`, so `ncks` asks before it overwrites a file.

Row A, once per region: the 1/16 degree mask and the 1/4 degree mask of the region, and
the first level of each:

<!-- chain: doc ncks-A -->
```bash
mask16="$(in_N cfg masks.fine)"
mask025="$(in_N cfg masks.coarse)"
directory="$FEATURES/${R}"
fine_x="$(in_N cfg "regions.${R}.fine.x")"
fine_y="$(in_N cfg "regions.${R}.fine.y")"
coarse_x="$(in_N cfg "regions.${R}.coarse.x")"
coarse_y="$(in_N cfg "regions.${R}.coarse.y")"

ncks -d "x,${fine_x}" -d "y,${fine_y}" "${mask16}" "${directory}/mesh_mask_exp16_${R}.nc"
ncks -d nav_lev,0 "${directory}/mesh_mask_exp16_${R}.nc" "${directory}/mesh_mask_exp16_surface_${R}.nc"

ncks -d "x,${coarse_x}" -d "y,${coarse_y}" "${mask025}" "${directory}/mesh_mask_exp4_${R}.nc"
ncks -d nav_lev,0 "${directory}/mesh_mask_exp4_${R}.nc" "${directory}/mesh_mask_exp4_surface_${R}.nc"
```

Row 1, for every month: the region cut out of the global `grid_T` (every level) and of
`grid_U` and `grid_V` (their first level only):

<!-- chain: doc ncks-1 -->
```bash
input_directory="$(in_N cfg paths.model_output)"
output_directory="$SUBSETS/${R}"
fine_x="$(in_N cfg "regions.${R}.fine.x")"
fine_y="$(in_N cfg "regions.${R}.fine.y")"

for d0 in $(months "$START" "$END"); do
    stem="MINT_1d_${d0}_${d0:0:6}30"
    ncks -d "x,${fine_x}" -d "y,${fine_y}" \
        "${input_directory}/${stem}_grid_T.nc" "${output_directory}/${stem}_grid_T_${R}.nc"
    ncks -d "x,${fine_x}" -d "y,${fine_y}" -d depthu,0 \
        "${input_directory}/${stem}_grid_U.nc" "${output_directory}/${stem}_grid_U_${R}.nc"
    ncks -d "x,${fine_x}" -d "y,${fine_y}" -d depthv,0 \
        "${input_directory}/${stem}_grid_V.nc" "${output_directory}/${stem}_grid_V_${R}.nc"
done
```

Do not run these in addition to the two scripts: they write the same files.

## What each step reads and writes

`O` is `paths.model_output`, `S` is `paths.subsets`, `K` is `paths.kappa`, `F` is
`paths.features`, `R` the region, `m` a month's `MINT_1d_<d0>_<d1>`.

| Row | Script | Reads | Writes |
|---|---|---|---|
| A | `00_regional_masks.sh` | `masks.fine`, `masks.coarse` | `F/R/mesh_mask_exp16_R.nc`, `mesh_mask_exp16_surface_R.nc`, `mesh_mask_exp4_R.nc`, `mesh_mask_exp4_surface_R.nc` (`_surface_`: the first level) |
| B | `00_regrid_grids.py` | `masks.fine`, `masks.coarse` | `F/R/R_input_grid_16.npy`, `F/R/R_target_grid_025.npy` (cell centres and corners) |
| C | `00_regrid_weights.py` | B | `F/R/weights_R.nc` |
| D | `00_filter_kappa.py` | `F/R/mesh_mask_exp4_R.nc`, B | `K/kappa16_R.nc` |
| E | `00_xnemo_mask.py` | `F/R/mesh_mask_exp4_R.nc` | `F/R/coarsened_data/mesh_mask_exp4_R_xnemo.nc` |
| 1 | `01_subset.sh` | `O/m_grid_{T,U,V}.nc` | `S/R/m_grid_{T,U,V}_R.nc`; T keeps every level, U and V the first only |
| 2 | `02_filter.py --variable grid_T` | `S/R/m_grid_T_R.nc`, `F/R/mesh_mask_exp16_R.nc`, `F/R/mesh_mask_exp4_R.nc`, `K/kappa16_R.nc` | `F/R/filtered_data/m_{thetao,so}_f_R.nc` |
| 3 | `02_filter.py --variable grid_UV` | `S/R/m_grid_{U,V}_R.nc`, `F/R/mesh_mask_exp16_surface_R.nc`, `F/R/mesh_mask_exp4_R.nc`, `K/kappa16_R.nc` | `F/R/filtered_data/m_{uo,vo}_f_R.nc` |
| 4 | `03_regrid.py` | `F/R/filtered_data/m_<v>_f_R.nc`, both regional masks, B, C | `F/R/coarsened_data/m_<v>_cg_R.nc` |
| 5 | `04_to_cdftools_layout.py` | `F/R/coarsened_data/m_{thetao,so}_cg_R.nc`, and `S/R/m_grid_T_R.nc` as the template | `F/R/coarsened_data/m_{thetao,so}_cg_R_cdftools.nc` |
| 6 | `cdfbn2` | 5, or `S/R/m_grid_T_R.nc`; the regional mask named in the namelist | `F/R/coarsened_data/m_bn2_cg_R.nc`; `F/R/m_bn2_R.nc` |
| 7 | `05_deformation_radius.py` | 6, the regional mask of the grid | `F/R/coarsened_data/m_Ld_cg_R.nc` (`--mode cg`); `F/R/m_Ld_R.nc` (`--mode non_cg`) |
| 8 | `05_time_mean_ld.py` | every month of 7 | `F/R/coarsened_data/MINT_1d_00610101_00721230_Ld_cg_R.nc`; `F/R/MINT_1d_00610101_00721230_Ld_R.nc` |
| 9 | `06_scale_aware_feature.py` | `F/R/coarsened_data/m_Ld_cg_R.nc`, `F/R/mesh_mask_exp4_R.nc` | `F/R/coarsened_data/m_sa_cg_R.nc` |
| 10 | `07_vorticity.py` | `F/R/coarsened_data/m_{uo,vo}_cg_R.nc`, `F/R/mesh_mask_exp4_R.nc` | `F/R/coarsened_data/m_vor_cg_R.nc` |
| 11 | `08_kinetic_energy.py --variable mke` | `F/R/coarsened_data/m_{uo,vo}_cg_R.nc`, the coarse time mean of 8, `F/R/mesh_mask_exp4_surface_R.nc` | `F/R/coarsened_data/m_mke_R.nc` |
| 12 | `08_kinetic_energy.py --variable eke` | `S/R/m_grid_{U,V}_R.nc`, the fine time mean of 8, `F/R/mesh_mask_exp16_surface_R.nc`, `F/R/mesh_mask_exp4_surface_R.nc` | `F/R/m_eke_Lmin3LdO25_R.nc` |
| 13 | `03_regrid.py --variable eke_Lmin3LdO25` | 12, both regional masks, B, C | `F/R/coarsened_data/m_eke_Lmin3LdO25_cg_R.nc` |
| 14 | `09_combine.py` | every month of 9, 10, 11 or 13 | `F/R/coarsened_data/MINT_1d_<y0>-<y1>_{vor_cg_R_mod,mke_R,sa_cg_R,eke_Lmin3LdO25_cg_R}.nc` |

## Options

Every script takes `--region` and `--config`; every script except the five `00_*` takes
`--start` and `--end`. The other options, and the values they take:

| Script | Option | Values |
|---|---|---|
| `02_filter.py` | `--variable` | `grid_T` (a scalar on T points; filters `--variable-name`) or `grid_UV` (the surface velocity `uo`, `vo`, as a vector) |
| | `--variable-name` | `thetao` or `so`; read only with `--variable grid_T` (default `thetao`) |
| `03_regrid.py` | `--variable` | `thetao`, `so`, `uo`, `vo` or `eke_Lmin3LdO25` |
| `05_deformation_radius.py` | `--mode` | `cg` for the 1/4 degree grid; any other value, such as `non_cg`, for the 1/16 degree grid |
| `05_time_mean_ld.py` | `--mode` | `cg` or `non_cg` |
| `08_kinetic_energy.py` | `--variable` | `mke` or `eke` |
| `09_combine.py` | `--variable` | `vor_cg`, `mke`, `sa_cg` or `eke_Lmin3LdO25_cg` |

Only `05_time_mean_ld.py` and `09_combine.py` check their values on the command line.
The others accept any text, and an unknown value fails only once the script has started,
not with a usage message: `08_kinetic_energy.py` with a `NameError`; `02_filter.py` and
`03_regrid.py` with a `FileNotFoundError`, for want of an input file named after the
value; and `02_filter.py --variable grid_T` with a `KeyError` for an unknown
`--variable-name`.

Some values that were options in the private scripts are fixed in the code, because the
released files were made with one value only: the filter scale of `mke` is the
spatially varying 3Ld, and that of `eke` is min(<3Ld>, <Δs_O25>) (`ld_scaling = 3` in
`08_kinetic_energy.py`); `04_to_cdftools_layout.py` reads `grid_T` as its template and
the variables `thetao` and `so`.

## Kept on purpose

These look odd. They are kept on purpose, because they are how the released files were made:

- The combined vorticity file is named `…_vor_cg_<region>_mod.nc`, with `_mod`, because
  every trained model was configured with that name. The released file was made in two
  passes (the monthly vorticity files then had no proper time axis, and the joined file
  was given one afterwards); `07_vorticity.py` now writes each month with the time axis
  of the velocities, so `09_combine.py` makes the same file in one pass.
- `vor` is stored as (`y_c`, `x_c`, `t`), the other three variables as (`t`, `y_c`, `x_c`).
- The `mke` files carry an extra `gdept` coordinate on a `z_c` dimension of length 1.
- `08_kinetic_energy.py` writes a global attribute `filter_lengthscale` on `mke` that the
  released `mke` files do not have; the values are the same.
- Negative eddy kinetic energy is clipped to zero, and N² below zero is set to `1e-12`
  before its square root is taken for the deformation radius.
- `cdfbn2` is given `-teos10`, so `thetao` and `so` are read as conservative temperature
  and absolute salinity.
- The `description` attribute of the `mke` files, and of the fine-grid file
  `…_eke_Lmin3LdO25_<region>.nc` that `08_kinetic_energy.py` writes, says
  `non_geostrophic`, and that of the vorticity files `full velocities`: the names of
  options of the private scripts that are gone. The coarse-grained `eke` training file,
  which `03_regrid.py` writes, has `DINO EXP16 regridded to EXP4 -> ocean T grid
  variables` instead.

## Things to know when running it

- **`05_time_mean_ld.py` and `09_combine.py` can hang.** In a few runs out of some tens
  all threads ended up waiting on a lock, with no data written; the cause appears to be
  that xarray's netCDF read and write locks can take the same two global locks in
  opposite orders. A run that ends gives the same file every time. The commands above
  run these two scripts with a time limit (`again_if_hung`) and run them again if it is
  reached.
- **Run `05_time_mean_ld.py` only over a period of more than one month.** It names its
  output by the first and last day of the period, so for a single month the output
  would have the name of its own input.
- **Overwriting.** `ncks` asks before it overwrites a file (the commands have no `-O`).
  The Python scripts overwrite without asking: see the warning under "Configuration"
  about the Zenodo files.
- **`07_vorticity.py`** takes `t` from the `uo` file of the month; it does not check that
  the `vo` file has the same.
- **Rows C and D are close to the released files, not identical.** Made again on the
  machine where this was checked, the regridding weights differed from the released
  ones in 538 of 64560 values (largest relative difference 1.7e-13) and kappa in 277 of
  52080 (1.3e-13), the same in every run; every row that reads the released weights or
  kappa reproduced its file bit for bit. So a run from the global output gives files
  that are close to the released ones, not bit-identical: see "How this was checked".
- **The xnemo masks of `SO_JET` and `SO_EAST`.** The released masks of `SO_GYRE` and
  `SO_TROP` are what `00_xnemo_mask.py` writes. Those of `SO_JET` and `SO_EAST` were made
  earlier from a mask cut with xarray `where` and `dropna`: they hold the same values,
  but their 7 masks are float32 (int8 here), `mbathy` and `misf` float64 (int32 here),
  and `e3t_1d`, `e3w_1d`, `gdept_1d`, `gdepw_1d` are broadcast over `y_c`, `x_c`.
- **CDFTOOLS.** The `cdfbn2` used was built from CDFTOOLS v4.0.0-61-g9498c03 with local
  changes, which are in [`envs/cdftools_local_changes.diff`](../envs/cdftools_local_changes.diff). One of them declares
  `CN_SOZOTAUX` in the namelist group `&NAMVARS`, where the two namelists here set it; a
  CDFTOOLS without that change may refuse the namelists (not tested).

## How this was checked

The checks were made on JASMIN in October 2026, with the environments in [`envs/`](../envs/).

**Step by step.** First the private scripts and notebooks the released files were made
with were run, one step at a time, each given the released files it reads as input. Their
arithmetic was not changed, but they were not run exactly as saved: each was run from a
copy, with the block of values at the top (region, dates, mode) and the path strings
rewritten; the notebooks were run as selected code cells, with the same rewriting; and
`ncks` and `cdfbn2` were run with the commands recovered from the `history` attributes
of the files and from the saved job scripts. Two cases are reconstructions of something
that is not saved. The coarse-grid time mean of Ld (row 8) exists only as the
fine-grid notebook, so the coarse form is that notebook with three strings changed (the
directory, the input pattern and the output name). The other is the route to the
released `SO_JET` and `SO_EAST` xnemo masks, a coarse mask cut with xarray `where`: it
was run from a notebook that is saved with the box of another region, with the box of
the region put in. It gives both released files bit for bit, but only for `SO_JET` is
there a file and a commit that corroborate it. Every step reproduced its released output
with `xr.testing.assert_allclose`, and all but two bit for bit: the regridding weights (row C) and kappa (row D) were close,
not identical (largest relative difference 1.7e-13 and 1.3e-13). The xnemo masks of
`SO_JET` and `SO_EAST` (row E) are the exception described above. Then the scripts were
moved here, given a command line and stripped of the options the released files did not
use, and each version was checked to give the same arrays as that private code. The
exception is the time axis of the monthly vorticity files, which was put right on
purpose. Every step of the private code was run twice and gave the same arrays both
times.

**The whole chain.** The commands of "The commands for one region" were run as they
stand, for `SO_JET` and January 0061 (`START=00610101`, `END=00610201`), in an empty
directory that held only the global output of that month, the two global masks and the
two released time means of row 8. Rows A to E were made again. Row 8 needs all 144
months, so its block was not run, and the two released time means were used in its
place; run on its own over the 144 released monthly files, `05_time_mean_ld.py`
reproduces both of them bit for bit. The chain was run twice, and both runs gave the same arrays in all 35 files they wrote.
`SUBSETS` and `FEATURES` were read from the config by the commands themselves. The
`ncks` commands under "The `ncks` commands of rows A and 1" are not run in addition to the
scripts. They were checked instead: with `ncks` replaced by a function that prints its
command line, they print, line for line, the commands that the two scripts printed in the
chain's log, and their `cfg` function is the one in the scripts. The chain's
five training files, against the released ones cut to January 0061:

| File | Against the released file | Largest absolute difference | Largest relative difference |
|---|---|---|---|
| `MINT_1d_0061-0061_vor_cg_SO_JET_mod.nc` | close; `t` identical | 8.1e-20 | 2.0e-12 |
| `MINT_1d_0061-0061_mke_SO_JET.nc` | close | 4.2e-15 | 1.5e-14 |
| `MINT_1d_0061-0061_sa_cg_SO_JET.nc` | identical | 0 | 0 |
| `MINT_1d_0061-0061_eke_Lmin3LdO25_cg_SO_JET.nc` | close | 5.1e-16 | 1.8e-14 |
| `mesh_mask_exp4_SO_JET_xnemo.nc` | every value equal; the types and shapes described above | 0 | 0 |

"Close" means that `xr.testing.assert_allclose` passes and some values differ in their
last bits. That comes from rows C and D. Every file of the chain that does not depend on
the weights or kappa is bit-identical to the released file: the regional masks and
subsets, the regridding grids, the eddy kinetic energy on the 1/16 degree grid, N² and
Ld on both grids, and `sa`. N² on the 1/4 degree grid is bit-identical although its
inputs are not: `cdfbn2` reads temperature and salinity as 4-byte reals, and differences
of order 1e-14 vanish in that rounding, at least for this month. Every file that does
depend on the weights or kappa (the filtered and coarse-grained fields, the vorticity,
`mke` and the coarse-grained `eke`) passes `xr.testing.assert_allclose`, with relative
differences of at most 3.4e-10; the monthly vorticity file passes once its `t` is set
aside (next paragraph).

The monthly vorticity files differ from the released monthly files in `t` alone: the
released ones hold the bare day numbers 0 to 29, these the dates of the velocity they
were computed from.

**Where the chain run differed from the commands as written.** The setup block (the
first block under "The commands for one region", the variables and `source .../activate`)
was not run as written: a prelude took its place. It sets the same variables, pointed at
a scratch directory, and also sources `/etc/profile.d/modules.sh` (so that `module load
jaspy` works in a script) and runs the whole chain under `set -eo pipefail`, which the
setup block does not. It sets `HANG_LIMIT=300`; the setup block says 900. In the first of
the two runs `again_if_hung` fired once: the `09_combine.py` of `sa_cg` was killed after
300 s and run again, and the second attempt ended normally; the second run had no retry.
The files of the first run are, array for array, the files of the second run, and the
`sa` file is identical to the released file, so the retry changed nothing. Every other
block of commands was run as written; the `ncks` blocks of rows A and 1 were checked, not
run, as said above.

**The other side of the filter scale.** For `SO_GYRE`, the one region where the filter
scale of `eke` is <3Ld> and not <Δs_O25>, rows 12 and 13 for January 0061, given the
released inputs, reproduce the released files bit for bit.
