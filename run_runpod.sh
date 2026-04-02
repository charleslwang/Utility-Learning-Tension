#!/bin/bash
set -e

H_SEEDS=64
H_M_TRAIN=512
A_SEEDS=48
A_M_TRAIN=512
OUTPUT_DIR="experiments/outputs_runpod"
DEVICE="cuda"

mkdir -p "$OUTPUT_DIR"

echo "--- Running neural M_H experiments on RunPod ---"
python3 experiments/run_h.py --policy twogate        --seeds $H_SEEDS --m $H_M_TRAIN --n_v 384 --n_test 4096 --n_shift 4096 --n_correction 192 --widths 4,8,16,32,64,96 --epochs 45 --device $DEVICE --output_dir $OUTPUT_DIR
python3 experiments/run_h.py --policy dest_val_nocap --seeds $H_SEEDS --m $H_M_TRAIN --n_v 384 --n_test 4096 --n_shift 4096 --n_correction 192 --widths 4,8,16,32,64,96 --epochs 45 --device $DEVICE --output_dir $OUTPUT_DIR
python3 experiments/run_h.py --policy dest_val       --seeds $H_SEEDS --m $H_M_TRAIN --n_v 384 --n_test 4096 --n_shift 4096 --n_correction 192 --widths 4,8,16,32,64,96 --epochs 45 --device $DEVICE --output_dir $OUTPUT_DIR
python3 experiments/run_h.py --policy dest_train     --seeds $H_SEEDS --m $H_M_TRAIN --n_v 384 --n_test 4096 --n_shift 4096 --n_correction 192 --widths 4,8,16,32,64,96 --epochs 45 --device $DEVICE --output_dir $OUTPUT_DIR

echo -e "\n--- Running neural M_A experiments on RunPod ---"
python3 experiments/run_a.py --policy no_cap --seeds $A_SEEDS --m $A_M_TRAIN --n_v 384 --n_test 4096 --n_shift 4096 --n_correction 160 --stages 36 --width 48 --device $DEVICE --output_dir $OUTPUT_DIR
python3 experiments/run_a.py --policy cap    --seeds $A_SEEDS --m $A_M_TRAIN --n_v 384 --n_test 4096 --n_shift 4096 --n_correction 160 --stages 36 --width 48 --B 2.5 --device $DEVICE --output_dir $OUTPUT_DIR

echo -e "\n--- Rendering figures ---"
python3 experiments/visualize.py --axis h --input_dir $OUTPUT_DIR --output_dir $OUTPUT_DIR
python3 experiments/visualize.py --axis a --input_dir $OUTPUT_DIR --output_dir $OUTPUT_DIR

echo -e "\n--- RunPod sweep complete ---"
