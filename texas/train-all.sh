#!/bin/bash


echo "Training Undefended model"
python texas-train-org-dmp.py --train_size 15000  --train_dmp 0 --train_org 1 --distill_lr 1.0 --train_lr 0.0005 

echo "Training DMP model"
python texas-train-org-dmp.py --train_size 15000  --train_dmp 1 --train_org 0 --distill_lr 1.0 --train_lr 0.0005  \
	--synt_data_path final-all-models/texas_synt_from_15000.npy --num_synt_for_train 100000 --random_synt 0  --org_path ./final-all-models/undefended-trainSize-15000.pth.tar

echo "Training SELENA model"
python texas-train-selena.py --train_size 15000  --train_selena 1 --train_org 1 --distill_lr 1.0 --train_lr 0.0005

echo "Training HAMP model"
python texas-train-hamp.py  --isTrain 1 --train_size 15000 --entropy_percentile 0.6 --entropy_penalty 1 --alpha 0.01  --distill_lr 0.5

echo "Training AdvReg model"
python texas-train-advreg.py --train_size 15000 --alpha 10 --lr 0.001

echo "Training DPSGD model"
python texas-train-dpsgd.py --train_size 15000  --dp_batchsize 64 --lr 0.005 --dp_norm_clip 1.0 --dp_noise_multiplier 1.44  --epochs 200

echo "Training Label Smoothing model"
# label smoothing uses low-entropy soft labels, entropy_percentile 0.15 amounts to a smoothing intensity of 0.09
python texas-train-hamp.py  --isTrain 1 --train_size 15000 --entropy_percentile 0.15 --entropy_penalty 0   --distill_lr 0.5



