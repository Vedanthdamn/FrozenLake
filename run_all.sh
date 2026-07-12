#!/bin/bash
set -e

configs=(
  "4x4 --slippery"
  "8x8 --slippery"
  "4x4 --no-slippery"
  "8x8 --no-slippery"
)

for config in "${configs[@]}"; do
  read -r map_name slippery_flag <<< "$config"
  echo "=== map=$map_name $slippery_flag ==="
  python3 train.py --map "$map_name" $slippery_flag
  python3 evaluate.py --map "$map_name" $slippery_flag
  python3 plot_training.py --map "$map_name" $slippery_flag
  python3 demo.py --map "$map_name" $slippery_flag
done

python3 plot_comparison.py
