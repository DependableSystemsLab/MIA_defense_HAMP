#!/bin/bash
source activate torch
train_size=15000
defense=ls
folder_tag=$defense
gpu=0
res_folder=lira-$folder_tag-fullMember-$train_size
declare -i total_models=128
declare -i index=0
index=$total_models-1
for tag in $(seq 0 $index)
do
    python lira-train-hamp.py --isTrain 1 --train_size $train_size --entropy_percentile 0.15 \
            --entropy_penalty 0  --distill_lr 0.5 --save_tag $tag --total_models $total_models \
            --folder_tag $folder_tag --gpu $gpu --res_folder $res_folder
            
     python lira-inference.py --org_path shadow-$folder_tag-trainSize-$train_size-fullMember/protected_model_best-$tag.pth.tar \
    --save_tag $tag --train_size $train_size  --res_folder $res_folder --isModifyOutput 0 --gpu $gpu
done

python lira-score.py --res_folder $res_folder
python lira-inference-defense.py --org_path ./final-all-models/$defense-trainSize-$train_size.pth.tar \
     --train_size $train_size  --res_folder lira-$folder_tag-defense-fullMember-$train_size  --isModifyOutput 0

python lira-score.py --res_folder lira-$folder_tag-defense-fullMember-$train_size
python lira-plot.py --shadow_data_path $res_folder --test_data_path lira-$folder_tag-defense-fullMember-$train_size
