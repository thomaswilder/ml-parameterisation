#!/bin/bash
#
# Row 1: the regional subsets of the global model output, one month at a time, with ncks.
#
#     bash 01_subset.sh --region SO_JET --start 00610101 --end 00730101 --config config.yml
#
# --start  first month, YYYYMMDD
# --end    end of the period, YYYYMMDD, exclusive
#
# Reads   {paths.model_output}/MINT_1d_<first day>_<last day>_grid_{T,U,V}.nc
#         regions.<region>.fine.{x,y}   zero-based, inclusive index ranges
# Writes  {paths.subsets}/<region>/MINT_1d_<first day>_<last day>_grid_{T,U,V}_<region>.nc
#         T keeps every level; U and V keep the first level only.
#
# The model has a 360-day calendar: every month runs from day 01 to day 30, and each
# month's files are found by those two dates.
#
# Needs ncks (NCO; version 5.3.3 made the released files) and a python3 with PyYAML,
# which reads the config. On JASMIN both come with `module load jaspy`.
# ncks asks before it overwrites an existing file.

set -euo pipefail

usage() {
    echo "usage: $(basename "$0") --region REGION --start YYYYMMDD --end YYYYMMDD --config CONFIG"
}

REGION=""
START=""
END=""
CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --region) REGION="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --end) END="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n "$REGION" && -n "$CONFIG" ]] || { usage >&2; exit 2; }
[[ "$START" =~ ^[0-9]{8}$ && "$END" =~ ^[0-9]{8}$ ]] || { usage >&2; exit 2; }

# one value of the config, e.g. `cfg paths.subsets`; a list is printed as "a,b"
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

input_directory="$(cfg paths.model_output)"
output_directory="$(cfg paths.subsets)/${REGION}"
fine_x="$(cfg "regions.${REGION}.fine.x")"
fine_y="$(cfg "regions.${REGION}.fine.y")"

current="$START"
while (( 10#$current < 10#$END )); do
    date_init="$current"
    date_end="${current:0:6}30"           # the month's last day (360-day calendar)
    stem="MINT_1d_${date_init}_${date_end}"

    run ncks -d "x,${fine_x}" -d "y,${fine_y}" \
        "${input_directory}/${stem}_grid_T.nc" "${output_directory}/${stem}_grid_T_${REGION}.nc"
    run ncks -d "x,${fine_x}" -d "y,${fine_y}" -d depthu,0 \
        "${input_directory}/${stem}_grid_U.nc" "${output_directory}/${stem}_grid_U_${REGION}.nc"
    run ncks -d "x,${fine_x}" -d "y,${fine_y}" -d depthv,0 \
        "${input_directory}/${stem}_grid_V.nc" "${output_directory}/${stem}_grid_V_${REGION}.nc"

    # the same day of the next month
    year=$((10#${current:0:4}))
    month=$((10#${current:4:2}))
    day="${current:6:2}"
    month=$((month + 1))
    if (( month == 13 )); then
        month=1
        year=$((year + 1))
    fi
    current="$(printf '%04d%02d%s' "$year" "$month" "$day")"
done
