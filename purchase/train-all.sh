#!/bin/bash

echo "Training Undefended model"
python purchase-train-org.py --train_size 20000   --train_org 1 --train_lr 0.00005 

echo "Training SELENA model"
python purchase-train-selena.py --train_size 20000  --train_selena 1 --train_org 1 --distill_lr 1.0 --train_lr 0.00005 

echo "Training HAMP model"
python purchase-train-hamp.py --isTrain 1 --train_size 20000 --entropy_percentile 0.8 --entropy_penalty 1 --alpha 0.01 --distill_lr 0.5

echo "Training AdvReg model"
python purchase-train-advreg.py --train_size 20000 --alpha 3 --lr 0.0005

echo "Training DPSGD model"
python purchase-train-dpsgd.py --train_size 20000  --dp_batchsize 128 --lr 0.005 --dp_norm_clip 1.0 --dp_noise_multiplier 1.7  --epochs 200

echo "Training Label Smoothing model"
# label smoothing uses low-entropy soft labels, entropy_percentile 0.05 amounts to a smoothing intensity of 0.03
python purchase-train-hamp.py --isTrain 1 --train_size 20000 --entropy_percentile 0.05 --entropy_penalty 0  --distill_lr 0.5 
