#!/bin/bash
source activate torch
train_size=20000
gpu=0

echo "=======>"
echo "=======> Please ensure you have finished training the shadow models for the Undefended models"
echo "=======> As MemGuard is applied to the trained (undefended) models." 
mkdir lira-memguard-fullMember-$train_size
cp lira-undefended-fullMember-$train_size/*keep.npy ./lira-memguard-fullMember-$train_size
cp lira-undefended-fullMember-$train_size/shadow*.npy ./lira-memguard-fullMember-$train_size


declare -i total_models=128
declare -i index=0
index=$total_models-1
for tag in $(seq 0 $index)
do 
    python lira-inference-memguard.py --org_path shadow-undefended-trainSize-$train_size-fullMember/unprotected_model_best-$tag.pth.tar  \
            --train_size $train_size  \
                      --memguard_path ./final-all-models/purchase_${train_size}_MIA_model.h5 \
                      --save_tag $tag --res_folder lira-memguard-fullMember-$train_size --gpu $gpu
done 

python lira-inference-memguard-defense.py --org_path ./final-all-models/undefended-trainSize-$train_size.pth.tar  --train_size $train_size  \
                      --memguard_path ./final-all-models/purchase_${train_size}_MIA_model.h5 \
                      --res_folder lira-memguard-defense-fullMember-$train_size --gpu $gpu 

python lira-score.py   --res_folder lira-memguard-defense-fullMember-$train_size
python lira-score.py   --res_folder lira-memguard-fullMember-$train_size
python lira-plot.py --shadow_data_path ./lira-memguard-fullMember-$train_size \
            --test_data_path lira-memguard-defense-fullMember-$train_size 
