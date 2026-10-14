#!/bin/bash
#
# Row A: the regional mesh masks, cut from the two global masks with ncks.
#
#     bash 00_regional_masks.sh --region SO_JET --config config.yml
#
# Reads   masks.fine, masks.coarse                    the global 1/16 and 1/4 degree masks
#         regions.<region>.fine.{x,y}, .coarse.{x,y}  zero-based, inclusive index ranges
# Writes  in {paths.features}/<region>/:
#         mesh_mask_exp16_<region>.nc           the 1/16 degree mask of the region
#         mesh_mask_exp16_surface_<region>.nc   its first level
#         mesh_mask_exp4_<region>.nc            the 1/4 degree mask of the region
#         mesh_mask_exp4_surface_<region>.nc    its first level
#
# Needs ncks (NCO; version 5.3.3 made the released files) and a python3 with PyYAML,
# which reads the config. On JASMIN both come with `module load jaspy`.
# ncks asks before it overwrites an existing file.

set -euo pipefail

usage() {
    echo "usage: $(basename "$0") --region REGION --config CONFIG"
}

REGION=""
CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --region) REGION="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n "$REGION" && -n "$CONFIG" ]] || { usage >&2; exit 2; }

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

# print a command, then run it
run() {
    echo "$*"
    "$@"
}

directory="$(cfg paths.features)/${REGION}"
mask16="$(cfg masks.fine)"
mask025="$(cfg masks.coarse)"
fine_x="$(cfg "regions.${REGION}.fine.x")"
fine_y="$(cfg "regions.${REGION}.fine.y")"
coarse_x="$(cfg "regions.${REGION}.coarse.x")"
coarse_y="$(cfg "regions.${REGION}.coarse.y")"

# 1/16 degree, then its first level
run ncks -d "x,${fine_x}" -d "y,${fine_y}" "${mask16}" "${directory}/mesh_mask_exp16_${REGION}.nc"
run ncks -d nav_lev,0 "${directory}/mesh_mask_exp16_${REGION}.nc" "${directory}/mesh_mask_exp16_surface_${REGION}.nc"

# 1/4 degree, then its first level
run ncks -d "x,${coarse_x}" -d "y,${coarse_y}" "${mask025}" "${directory}/mesh_mask_exp4_${REGION}.nc"
run ncks -d nav_lev,0 "${directory}/mesh_mask_exp4_${REGION}.nc" "${directory}/mesh_mask_exp4_surface_${REGION}.nc"
