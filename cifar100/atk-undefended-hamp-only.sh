#!/bin/bash
size=25000 # specify the size of the training set
gpu=0


# the following parameters specify the paths to different evaluated models
undefended=./final-all-models/undefended-trainSize-$size.pth.tar 
selena=./final-all-models/selena-trainSize-$size.pth.tar
hamp=./final-all-models/hamp-trainSize-$size.pth.tar
dmp=./final-all-models/dmp-trainSize-$size.pth.tar
advreg=./final-all-models/advreg-trainSize-$size.pth.tar  
memguard=./final-all-models/undefended-trainSize-$size.pth.tar 
ls=./final-all-models/ls-trainSize-$size.pth.tar 


# Given the time limit for the AE, we will evaluate only the undefended model, and HAMP model
# However, you can always choose to evaluate other additional models if time permits. 
tag_list=(undefended-$size hamp-$size)
declare -i i=0
for target_model in $undefended $hamp; do

	# isModifyOutput indicates the output modification defense by HAMP
	isModifyOutput=0
	if [ $target_model = $hamp ]
	then
	isModifyOutput=1
	fi

	isMemGuard=0
	if [ ${tag_list[$i]} = memguard-$size ]
	then
		# compute the obfuscated scores by MemGuard
		# and save these scores to be evaluated later 
		python attack.py --path $target_model  --train_size $size  \
		                 --isModifyOutput 0 --attack entropy --save_tag test  --getModelAcy 0 --isMemGuard 0 --prepMemGuard 1 \
		                 --memguard_path ./final-all-models/cifar100\_$size\_MIA_model.h5 --gpu $gpu
		isMemGuard=1              
	fi

	# evaluate each defense models with different attacks, e.g., entropy-based attack, NN-based attack
	python attack.py --gpu $gpu  --isMemGuard $isMemGuard --path $target_model  --train_size $size   --isModifyOutput $isModifyOutput --attack entropy --save_tag ${tag_list[$i]}  --getModelAcy 1
	python attack.py --gpu $gpu  --isMemGuard $isMemGuard --path $target_model  --train_size $size   --isModifyOutput $isModifyOutput --attack loss --save_tag ${tag_list[$i]} 
	python attack.py --gpu $gpu  --isMemGuard $isMemGuard --path $target_model  --train_size $size   --isModifyOutput $isModifyOutput --attack nn --save_tag ${tag_list[$i]}

	i=$i+1
done










