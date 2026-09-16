#!/usr/bin/env bash
set -e

SCRIPT="INP_Former_MVTec_AD2_test_public_tiff.py"
DATA_PATH="./mvtec_ad_2"

CLASSES=(
    can
    fabric
    fruit_jelly
    rice
    sheet_metal
    vial
    wallplugs
    walnuts
)

for cls in "${CLASSES[@]}"; do
    echo "========================================"
    echo "Training: ${cls}"
    echo "========================================"

    python "${SCRIPT}" \
        --data_path "${DATA_PATH}" \
        --phase train \
        --item "${cls}" \
        --export_validation

    echo "Finished: ${cls}"
    echo
done

echo "All classes finished."