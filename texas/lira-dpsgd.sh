#!/bin/bash 
train_size=15000
defense=dpsgd
folder_tag=$defense
gpu=0
res_folder=lira-$folder_tag-fullMember-$train_size
declare -i total_models=128
declare -i index=0
index=$total_models-1
for tag in $(seq 0 $index)
do
    python lira-train-dpsgd.py --train_size $train_size   --dp_batchsize 64 --lr 0.005 --dp_norm_clip 1.0 \
    --dp_noise_multiplier 1.44  --epochs 200\
     --save_tag $tag --total_models $total_models \
    --folder_tag $folder_tag --res_folder $res_folder 
    
     python lira-inference.py --org_path shadow-$folder_tag-trainSize-$train_size-fullMember/protected_model_best-$tag.pth.tar \
    --save_tag $tag --train_size $train_size  --res_folder $res_folder 
done
python lira-score.py   --res_folder $res_folder

python lira-inference-defense.py --org_path ./final-all-models/$defense-trainSize-$train_size.pth.tar \
     --train_size $train_size  --res_folder lira-$folder_tag-defense-fullMember-$train_size  --isModifyOutput 0

python lira-score.py --res_folder lira-$folder_tag-defense-fullMember-$train_size
python lira-plot.py --shadow_data_path $res_folder --test_data_path lira-$folder_tag-defense-fullMember-$train_size


