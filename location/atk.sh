#!/bin/bash
#source activate torch

size=1500

gpu=0

undefended=./final-all-models/undefended-trainSize-$size.pth.tar 

# The original SELENA model we used for reporting the results in the paper was lost,
# so we provide a newly trained one
selena=./final-all-models/selena-trainSize-$size.pth.tar
hamp=./final-all-models/hamp-trainSize-$size.pth.tar
dmp=./final-all-models/dmp-trainSize-$size.pth.tar
advreg=./final-all-models/advreg-trainSize-$size.pth.tar 
memguard=./final-all-models/undefended-trainSize-$size.pth.tar 
dpsgd=./final-all-models/dpsgd-trainSize-$size.pth.tar 



tag_list=(undefended-$size selena-$size hamp-$size dmp-$size advreg-$size dpsgd-$size memguard-$size)

declare -i i=0

for target_model in $undefended $selena $hamp $dmp $advreg $dpsgd $memguard; do

	isModifyOutput=0
	if [ $target_model = $hamp ]
	then
	isModifyOutput=1
	fi

	isMemGuard=0
	if [ ${tag_list[$i]} = memguard-$size ]
	then
		# save the modified output by MemGuard
		isMemGuard=1
		python attack.py --path $target_model  --train_size $size  \
		                 --isModifyOutput 0 --attack entropy --save_tag test  --getModelAcy 0 --isMemGuard 0 --prepMemGuard 1 \
		                 --memguard_path ./final-all-models/location\_$size\_MIA_model.h5 --gpu $gpu
	fi


	python attack.py --gpu $gpu  --isMemGuard $isMemGuard --path $target_model  --train_size $size   --isModifyOutput $isModifyOutput --attack entropy --save_tag ${tag_list[$i]}  --getModelAcy 1
	python attack.py --gpu $gpu  --isMemGuard $isMemGuard --path $target_model  --train_size $size   --isModifyOutput $isModifyOutput --attack loss --save_tag ${tag_list[$i]} 
	python attack.py --gpu $gpu  --isMemGuard $isMemGuard --path $target_model  --train_size $size   --isModifyOutput $isModifyOutput --attack nn --save_tag ${tag_list[$i]}




	i=$i+1
done

 
