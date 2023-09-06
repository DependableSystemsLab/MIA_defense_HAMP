# MIA defense - HAMP

Code for the paper "**Overconfidence is a Dangerous Thing: Mitigating Membership Inference Attacks by Enforcing Less Confident Prediction**" in NDSS'24.

09/2023: We earned all badges (available, functional, reproduced) in the NDSS artifact evaluation.

## Getting started

Download the data and trained models for each dataset: [[Purchase](https://drive.google.com/file/d/1agznlDEFZKxFgHh9EkGup9U61BEOVDlD/view?usp=sharing)] [[Texas](https://drive.google.com/file/d/1BLmnrg4qSNgDE5DWGPWoKd27wmnX8sQ6/view?usp=sharing)] [[Cifar100](https://drive.google.com/file/d/1qenhMyoGiSU0V5xKzfRGCaiUWQ-D0VPD/view?usp=share_link)] [[Cifar10](https://drive.google.com/file/d/1lsLAKOJsd61YaM32_B3fECiBDksmlHrU/view?usp=share_link)] [[Location](https://drive.google.com/file/d/1sHP7DZya35flax6fqc_YI0VyavlrO6rD/view?usp=sharing)]

```
unzip purchase-data.zip 
unzip texas-data.zip 
unzip cifar100-data.zip 
unzip cifar10-data.zip 
unzip location-data.zip 
mv ./purchase-data/* ./purchase/
mv ./texas-data/* ./texas/
mv ./location-data/* ./location/
mv ./cifar10-data/* ./cifar10/
mv ./cifar100-data/* ./cifar100/
```


### Install the dependencies

We tested on Debian 10 with Python 3.8.17. We use torch-gpu and you can install it based on your own cuda version. We also recommend installing the dependencies via virtual env management tool, e.g., anaconda.

```
conda create -n hamp python=3.8
conda activate hamp 

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pandas==1.5.0 scikit-learn==1.0.2 scipy==1.7 tensorflow==2.12.0 rdt==0.6.4 tqdm numba matplotlib numpy==1.22.4
```

## Evaluation options

- Option 1: for reproducing the key results in a time-sensitive manner. 

Run ```run-all.sh```, which consolidates all experimental steps on all datasets. Note that this option only evaluates the performance on the undefended model and the HAMP model. 

Please refer to Step 2 - Evaluation option 1 below before running ```run-all.sh```, as you will need to download additional data for this evaluation. 


- Option 2: for comprehensive evaluation (this option requires more evaluation time)

Run the three experimental steps for each dataset separately, each of which is explained below. In this case, you will be evaluating the performance on all models (including other defense models) and this will take more time. 


### Step 1: Performing Membership Inference

Go to each dataset directory and run ```atk.sh &> R-atk```, which evaluates each model with multiple score-based attacks (except LiRA, which will be executed separately as it needs to train multiple shadow models). We use different tags in differentiating different defenses (e.g., *undefended* means the undefended model, *ls* means label smoothing). 


The output reports the model accuracy, the attack true positive rate (TPR) @ 0.1% false positive rate (FPR), as well as true negative rate (TNR) @ 0.1% false negative rate (FNR). 

We exclude the label-only attacks (e.g., boundary-based attack) as they are unsuccessful when controlled at low false positive/negative regime. 


### Step 2: Evaluating the Likelihood-ratio attack (LiRA)

- **Evaluation option 1** (using the scores from the pre-trained shadow models)

Shadow model training in the LiRA evaluation is a very time-consuming process, and we provide the pre-computed logit-scaled scores from the shadow models to reproduce our results. You can download the scores here [[Purchase](https://drive.google.com/file/d/10-T5S1k5jJ0zHEHi-S8-IpseRyi9KBjF/view?usp=share_link)] [[Texas](https://drive.google.com/file/d/1YRO6EeBiBJrjNWHwuIHZHG1QtQFz8T56/view?usp=share_link)] [[Cifar100](https://drive.google.com/file/d/1yHfELNLEEurBfWy5gCNiiiU8f6WfPTsy/view?usp=share_link)] [[Cifar10](https://drive.google.com/file/d/1VtOuFVL497BiiM7shr3LuMlgB-bv8xe7/view?usp=share_link)] [[Location](https://drive.google.com/file/d/157uMira4I9MycPuCYOUEGT8d-IYzAtjs/view?usp=share_link)], and unzip them in the respecitve dataset directory. Then you can go to the respective dataset directory and run the evaluation as follows. 


```
# For LiRA evaluation on the [ Purchase100 ] dataset
python lira-plot.py --shadow_data_path lira-undefended-fullMember-20000 --test_data_path lira-undefended-defense-fullMember-20000
python lira-plot.py --shadow_data_path lira-hamp-fullMember-20000 --test_data_path lira-hamp-defense-fullMember-20000

# For LiRA evaluation on the [ Texas100 ] dataset
python lira-plot.py --shadow_data_path lira-undefended-fullMember-15000 --test_data_path lira-undefended-defense-fullMember-15000
python lira-plot.py --shadow_data_path lira-hamp-fullMember-15000 --test_data_path lira-hamp-defense-fullMember-15000

# For LiRA evaluation on the [ Location30 ] dataset
python lira-plot.py --shadow_data_path lira-undefended-fullMember-1500 --test_data_path lira-undefended-defense-fullMember-1500
python lira-plot.py --shadow_data_path lira-hamp-fullMember-1500 --test_data_path lira-hamp-defense-fullMember-1500

# For LiRA evaluation on the [ CIFAR10 ] dataset
python lira-plot.py --shadow_data_path lira-undefended-fullMember-25000 --test_data_path lira-undefended-defense-fullMember-25000
python lira-plot.py --shadow_data_path lira-hamp-fullMember-25000 --test_data_path lira-hamp-defense-fullMember-25000

# For LiRA evaluation on the [ CIFAR100 ] dataset
python lira-plot.py --shadow_data_path lira-undefended-fullMember-25000 --test_data_path lira-undefended-defense-fullMember-25000
python lira-plot.py --shadow_data_path lira-hamp-fullMember-25000 --test_data_path lira-hamp-defense-fullMember-25000
```

- **Evaluation option 2** (re-training the shadow models on your own)

Go to each dataset directory and run ```lira-[defense_name].sh```, e.g., ```lira-hamp.sh &> R-lira-hamp```. For each defense, it first trains 128 shadow models, then computes the logit-scaled scores on both the shadow and defense models, and finally performs the hypothesis test for membership inference. 

Please be aware that shadow model training is a very time-consuming process, and some of the defense techniques (e.g., SELENA) are particularly so. You can consider the following options to accelerate the evaluation process: 

1. Distribute the training across multiple GPUs. 
2. Reduce the number of shadow models (default 128). 

### Interpreting the results

Step 1 and step 2 above evaluate the MIA risks posed by different attacks, and you can manually determine the *highest* attack TPR\@0.1\%FPR and attack TNR\@0.1\%FNR from the overall results. 


### Step 3: Training the model from scratch

Run ```./train-all.sh``` on each dataset directory. 


## Citation
If you find this code useful, please consider citing our paper

```
@inproceedings{chen2023overconfidence,
      title={Overconfidence is a Dangerous Thing: Mitigating Membership Inference Attacks by Enforcing Less Confident Prediction}, 
      author={Chen, Zitao and Pattabiraman, Karthik},
      booktitle = {Network and Distributed System Security (NDSS) Symposium},
      year={2024}
}
```
