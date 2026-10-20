# The tools outside the two conda environments

The versions the checks of 2026-10-13 and 2026-10-14 ran with (see `../README.md`).

## N: `ncks`, and `python3` with PyYAML

Used by `00_regional_masks.sh` and `01_subset.sh`. On JASMIN both come from
`module load jaspy` (jaspy 3.12, `jaspy3.12-mf3-25.3.0-3-v20250704`):

- NCO 5.3.3 ("Sea Shanty", built on Mar 29 2025), the version named in the `NCO`
  attribute of the released regional masks and subsets;
- Python 3.12.11 with PyYAML 6.0.2, which the two scripts use to read the config.

## C: `cdfbn2` from CDFTOOLS

Used by row 6, with the namelists in `../cdftools/`.

- CDFTOOLS from https://github.com/meom-group/CDFTOOLS at commit
  `9498c03cbca0908c7f707ba7508901973c63ef1c` (`git describe`: `v4.0.0-61-g9498c03`).
- With the local changes in `cdftools_local_changes.diff` (`git diff` of the working
  tree). The changed files are older than the compiled modules (2024-11-05) and the
  `cdfbn2` binary (2025-07-30), so the binary was built with them. The ones that reach
  `cdfbn2` are:
  - `src/modcdfnames.F90`: declares `cn_sozotaux` in the namelist group `namvars`. The
    two namelists in `../cdftools/` set `CN_SOZOTAUX` in `&NAMVARS`, and CDFTOOLS reads
    the groups without `IOSTAT`, so a build without this change may stop on them (not
    tested). Removing the `CN_SOZOTAUX` line from the namelists would then be needed.
  - `src/modutils.f90`: `SetFileName` looks for `<conf>_<tag>_grid-<grid>.nc` before
    `<conf>_<tag>_grid_<grid>.nc`. `cdfbn2` is given its file names, so this is not
    expected to matter.
  - `Macrolib/macro.conda` (the `make.macro` used): gfortran with `-Og` in place of `-O`,
    and without `-lmpich`.
- Linked against the netCDF and HDF5 libraries of the same jaspy environment
  (`libnetcdf.so.19`, `libnetcdff.so.7`, `libhdf5.so.310`).
- `bin/cdfbn2`: 271504 bytes, sha256
  `5fff06006f5106b8654a98510660026812483839a5ea288cf9f49881def6b881`.
