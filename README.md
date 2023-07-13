# MIA defense - HAMP

Code for the paper "**Overconfidence is a Dangerous Thing: Mitigating Membership Inference Attacks by Enforcing Less Confident Prediction**" in NDSS'24.

## Getting started

Download the data and trained models for each dataset: [[Purchase](https://drive.google.com/file/d/1agznlDEFZKxFgHh9EkGup9U61BEOVDlD/view?usp=sharing)] [[Texas](https://drive.google.com/file/d/1BLmnrg4qSNgDE5DWGPWoKd27wmnX8sQ6/view?usp=sharing)] [[Cifar100](https://drive.google.com/file/d/1qenhMyoGiSU0V5xKzfRGCaiUWQ-D0VPD/view?usp=share_link)] [[Cifar10](https://drive.google.com/file/d/1lsLAKOJsd61YaM32_B3fECiBDksmlHrU/view?usp=share_link)] [[Location](https://drive.google.com/file/d/1sHP7DZya35flax6fqc_YI0VyavlrO6rD/view?usp=sharing)]

```
unzip purchase-data.zip -d ./purchase
unzip texas-data.zip -d ./texas
unzip cifar100-data.zip -d ./cifar100
unzip cifar10-data.zip -d ./cifar10
unzip location-data.zip -d ./location
```


Install the dependencies. 

```
# We install torch-gpu with cuda v1.12.0, and you may change to a different download version depending on your driver version
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pandas==1.5.0 scikit-learn==1.0.2 scipy==1.7 tensorflow==2.12.0 rdt==0.6.4 tqdm numba matplotlib numpy==1.22.4
```


## Performing Membership Inference

Go to each dataset directory and run ```atk.sh &> R-atk```, which evaluates each model with multiple score-based attacks (except LiRA, which will be executed separately as it needs to train multiple shadow models). We use different tags in differentiating different defenses (e.g., *undefended* means the undefended model, *ls* means label smoothing). 


The output reports the model accuracy, the attack true positive rate (TPR) @ 0.1% false positive rate (FPR), as well as true negative rate (TNR) @ 0.1% false negative rate (FNR). 

We exclude the label-only attacks (e.g., boundary-based attack) as they are unsuccessful when controlled at low false positive/negative regime. 


## Evaluating the Likelihood-ratio attack (LiRA)

Go to each dataset directory and run ```lira-[defense_name].sh```, e.g., ```lira-hamp.sh &> R-lira-hamp```. This trains 128 shadow models for each defense. 

Please be aware that shadow model training is a very time-consuming process, and some of the defense techniques (e.g., SELENA) are particularly so. You can consider the following options to accelerate the evaluation process: 

1. Distribute the training across multiple GPUs. 
2. Reduce the number of shadow models (default 128). 

### Pre-trained shadow models

We provide the pre-trained shadow models for HAMP on the CIFAR10 and CIFAR100 datasets for a speedy evaluation ([download here](https://drive.google.com/file/d/1b5feUBr6vlhVzhxD-gSCAMHCM01_cObr/view?usp=share_link)). In this case, you'll only need to get the logits from these shadow models, and then to initiate the inference process. 

Each folder (cifar10 or cifar100) contains two sub-folders: ```shadow-hamp-trainSize-25000-fullMember```, which contains 128 shadow models. ```lira-hamp-fullMember-25000```, which contains the index to the shadow training data for each model (this is for indexing the member and non-member samples for each shadow model). 

Place these two folders in the respective dataset directory, and remove the the shadow-training part by  removing ```python lira-train-hamp.py ...``` in ```lira-hamp.sh```, and then run the modified script to perform the evaluation. 

## Training the model from scratch

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
