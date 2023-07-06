#!/bin/bash


echo "Training Undefended model"
python cifar-train-org.py --train_size 25000  --train_org 1 --lr 0.5 

echo "Training DMP model"
python cifar-train-dmp.py --train_size 25000 --org_model_path ./final-all-models/undefended-trainSize-25000.pth.tar \
        --gan_path ./final-all-models/cifar10_netG_epoch_99-25000.pth --train_dmp 1 --num_synthetic 150000  --distill_lr 0.1 

echo "Training SELENA model"
python cifar-train-selena.py --train_size 25000  --train_selena 1 --train_org 1 --distill_lr 0.5 --lr 0.5

echo "Training HAMP model"
python cifar-train-hamp.py --isTrain 1 --train_size 25000 \
        --entropy_percentile 0.95 --entropy_penalty 1 --alpha 0.001  --distill_lr 0.5
        
echo "Training AdvReg model"
python cifar-train-advreg.py --train_size 25000 --alpha 6 --lr 1.0


 
