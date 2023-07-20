#!/bin/bash


# Given the time limit for the AE, we will evaluate only the undefended model, and HAMP model
# However, you can always choose to evaluate other additional models if time permits. 
echo "Training Undefended model"
python train-org-dmp.py --train_size 1500  --train_org 1   --lr 0.5 --model_save_tag 0

echo "Training HAMP model"
python train-hamp.py  --isTrain 1 --train_size 1500 \
    --entropy_percentile 0.5 --entropy_penalty 1 --alpha 0.001  --distill_lr 0.5 \
    --model_save_tag 0


: '''
echo "Training SELENA model"
python train-selena.py --train_size 1500   --train_selena 1 --train_org 1 --distill_lr 1.0 --lr 0.5 --model_save_tag 0

echo "Training AdvReg model"
python train-advreg.py --train_size 1500 --alpha 10 --lr 0.001


echo "Training DMP model"
python train-org-dmp.py --train_size 1500  --train_dmp 1 --train_org 0 --distill_lr 0.1 \
    --synt_data_path final-all-models/location_synt_from_1500.npy --num_synt_for_train 10000  \
     --org_path ./final-all-models/undefended-trainSize-1500.pth.tar

echo "Training DPSGD model"
python train-dpsgd.py --train_size 1500  --dp_batchsize 128 --lr 0.005 --dp_norm_clip 3.0 --dp_noise_multiplier 2.91  --epochs 50
'''
