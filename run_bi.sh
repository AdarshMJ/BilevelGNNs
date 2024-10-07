#!/bin/bash

layers=(2 4 8 16 32 64 128)

echo "Training for layers: ${layers[@]}"

for num_layers in "${layers[@]}"
do
    python main.py \
    --dataset chameleon_filtered.npz \
    --splits 1 \
    --inner_iterations 100 \
    --inner_lr 0.001 \
    --LR 0.001 \
    --gamma 0.0 \
    --dropout 0.4130296 \
    --hidden_dimension 128 \
    --num_layers $num_layers \
    --out "bilevel_layers_${num_layers}.csv"
done

# Run the Python script to plot the results
python plot_dirichlet_energy.py "${layers[@]}"
