#!/bin/bash

echo "====> Evaluation on [ Purchase100 ] dataset"
cd ./purchase
# ./atk.sh evaluates each model with multiple score-based attacks, except LiRA attack
bash ./atk-undefended-hamp-only.sh &> R-atk
# the following two commands perform LiRA evaluation (using the pre-computed scores we provided) 
# on the undefended and HAMP models, respectively
python lira-plot.py --shadow_data_path lira-undefended-fullMember-20000 --test_data_path lira-undefended-defense-fullMember-20000 &> R-undefended-lira
python lira-plot.py --shadow_data_path lira-hamp-fullMember-20000 --test_data_path lira-hamp-defense-fullMember-20000 &> R-hamp-lira
# the following command re-trains different defense models 
bash ./train-undefended-hamp-only.sh &> R-train

echo "====> Evaluation on [ Texas100 ] dataset"
cd ../texas
bash ./atk-undefended-hamp-only.sh &> R-atk
python lira-plot.py --shadow_data_path lira-undefended-fullMember-15000 --test_data_path lira-undefended-defense-fullMember-15000 &> R-undefended-lira
python lira-plot.py --shadow_data_path lira-hamp-fullMember-15000 --test_data_path lira-hamp-defense-fullMember-15000 &> R-hamp-lira
bash ./train-undefended-hamp-only.sh &> R-train

echo "====> Evaluation on [ Location30 ] dataset"
cd ../location
bash ./atk-undefended-hamp-only.sh &> R-atk
python lira-plot.py --shadow_data_path lira-undefended-fullMember-1500 --test_data_path lira-undefended-defense-fullMember-1500 &> R-undefended-lira
python lira-plot.py --shadow_data_path lira-hamp-fullMember-1500 --test_data_path lira-hamp-defense-fullMember-1500 &> R-hamp-lira
bash ./train-undefended-hamp-only.sh &> R-train

echo "====> Evaluation on [ CIFAR10 ] dataset"
cd ../cifar10
bash ./atk-undefended-hamp-only.sh &> R-atk
python lira-plot.py --shadow_data_path lira-undefended-fullMember-25000 --test_data_path lira-undefended-defense-fullMember-25000 &> R-undefended-lira
python lira-plot.py --shadow_data_path lira-hamp-fullMember-25000 --test_data_path lira-hamp-defense-fullMember-25000 &> R-hamp-lira
bash ./train-undefended-hamp-only.sh &> R-train

echo "====> Evaluation on [ CIFAR100 ] dataset"
cd ../cifar100
bash ./atk-undefended-hamp-only.sh &> R-atk
python lira-plot.py --shadow_data_path lira-undefended-fullMember-25000 --test_data_path lira-undefended-defense-fullMember-25000 &> R-undefended-lira
python lira-plot.py --shadow_data_path lira-hamp-fullMember-25000 --test_data_path lira-hamp-defense-fullMember-25000 &> R-hamp-lira
bash ./train-undefended-hamp-only.sh &> R-train



 
