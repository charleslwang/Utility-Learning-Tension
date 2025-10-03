#!/bin/bash
set -e

H_SEEDS=10
H_M_TRAIN=150
A_SEEDS=15  # reduced
A_M_TRAIN=500
OUTPUT_DIR="experiments/outputs"

# echo "--- Running M_H Experiments ---"
# python experiments/run_h.py --policy twogate        --seeds $H_SEEDS --m $H_M_TRAIN --output_dir $OUTPUT_DIR
# python experiments/run_h.py --policy dest_val_nocap --seeds $H_SEEDS --m $H_M_TRAIN --output_dir $OUTPUT_DIR
# python experiments/run_h.py --policy dest_val       --seeds $H_SEEDS --m $H_M_TRAIN --output_dir $OUTPUT_DIR
# python experiments/run_h.py --policy dest_train     --seeds $H_SEEDS --m $H_M_TRAIN --output_dir $OUTPUT_DIR

# echo -e "\n--- Running M_A Experiments ---"
# python experiments/run_a.py --policy no_cap --seeds $A_SEEDS --m $A_M_TRAIN --output_dir $OUTPUT_DIR
# python experiments/run_a.py --policy cap    --seeds $A_SEEDS --m $A_M_TRAIN --B 2.5 --output_dir $OUTPUT_DIR

echo -e "\n--- Generating Visualizations ---"
python experiments/visualize.py --axis h --input_dir $OUTPUT_DIR
# python experiments/visualize.py --axis a --input_dir $OUTPUT_DIR

echo -e "\n--- ✅ Complete! ---"
