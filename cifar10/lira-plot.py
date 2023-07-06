# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pyformat: disable

import os
import scipy.stats

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import functools

# Look at me being proactive!
import matplotlib
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42

import argparse
parser = argparse.ArgumentParser()  
parser.add_argument('--shadow_data_path', type=str, default='0')
parser.add_argument('--test_data_path', type=str, default='00')
parser.add_argument('--fpr', type=float, default=0.001)
parser.add_argument('--save_tag', type=str, default=None)
parser.add_argument('--output_folder', type=str, default='attack_outputs', help='output folder for storing the attack outputs')
args = parser.parse_args()


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc

def load_data():
    """
    Load our saved scores and then put them into a big matrix.
    """
    global shadow_scores, shadow_labels, test_scores, test_labels

    shadow_scores = []  # 
    shadow_labels = []  # 

    test_scores = []    # contain both member and non-member on the target model 
    test_labels = []
 
    look_for_string = 'score'
    for r, d, f in os.walk(args.shadow_data_path):

        for file in f:
            if(look_for_string in file): 
                loaded_scores = np.load(os.path.join(r, file))
                shadow_scores.append( loaded_scores )

                shadow_labels.append( np.load(os.path.join(r, file.replace(look_for_string, 'keep'))) )
 


    for r, d, f in os.walk(args.test_data_path):
        for file in f: 
            if(look_for_string in file):
                loaded_scores = np.load(os.path.join(r, file))
                test_scores.append(loaded_scores)
                
                test_labels.append( np.load(os.path.join(r, file.replace(look_for_string, 'keep'))) )
 
    print('===============')
    print('===> %s %s '%(args.shadow_data_path, args.test_data_path))
    print('===============\n')
    shadow_scores = np.array(shadow_scores)
    shadow_labels = np.array(shadow_labels)
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)
  
 

    shadow_scores = shadow_scores[:, :, np.newaxis]
    test_scores = test_scores[:, :, np.newaxis]
 
    #print(shadow_scores.shape, test_scores.shape)

    test_labels = test_labels.astype(bool)
    shadow_labels = shadow_labels.astype(bool) 

  
    return shadow_scores, shadow_labels, test_scores, test_labels



def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):  

        dat_in.append(scores[keep[:,j],j,:])
        dat_out.append(scores[~keep[:,j],j,:])

 

    in_size = min(min(map(len,dat_in)), in_size)
    out_size = min(min(map(len,dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
 

    #print("Online attack: Num of IN models {}, and OUT models {}".format(dat_out.shape[1], dat_in.shape[1])) 
 

    for ans, sc in zip(check_keep, check_scores): 
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in+1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)
        score = pr_in-pr_out #"this is like take the log of the likelihood ratio test"

        prediction.extend(score.mean(1))
        answers.extend(ans)


    return prediction, answers


def generate_ours_offline( keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,
                          fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []
 

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :]) # take all the in_model for that particular sample (keep is used as a bool to identify in_model)
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len,dat_out)), out_size)
    dat_out = np.array([x[:out_size] for x in dat_out])


    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []

    cnt = 1
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)
        prediction.extend(score.mean(1)) 
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers

def do_plot(fn, legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    prediction, answers = fn(shadow_labels, 
                             shadow_scores,  # scores of the non-member data
                             test_labels,     # labels of the testing data, including both members and non-members (for evaluating the attack)
                             test_scores)   # scores of the testing data, including both members and non-members (for evaluating the attack)

    prediction = np.array(prediction)
    answers = np.array(answers, dtype=bool)
    if(args.save_tag!=None):

        if(not os.path.exists(args.output_folder)):
            os.mkdir(args.output_folder)
        np.save( os.path.join( args.output_folder, 'lira-%s-%s.npy'%(legend,args.save_tag) ), np.r_[ answers, prediction*-1] )


    fpr, tpr, auc, acc = sweep_fn(prediction, answers)
    highest_tpr = tpr[np.where(fpr<args.fpr)[0][-1]]
    fnr = 1 - tpr 
    tnr = 1 - fpr
    highest_tnr = tnr[np.where(fnr<args.fpr)[0][0]]
    print('Attack type ==> %s\n %.2f%% TPR @ %.2f%% FPR |  %.2f%% TNR @ %.2f%% FNR | AUC %.4f\n'%(legend, highest_tpr*100, args.fpr*100, highest_tnr*100, args.fpr*100, auc))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    #plt.plot(fpr, tpr, label=legend+metric_text, **plot_kwargs)
    return (acc,auc)


def fig_fpr_tpr():

    plt.figure(figsize=(4,3))


    do_plot(generate_ours,
            "online",
            metric='auc'
    )

    do_plot(functools.partial(generate_ours, fix_variance=True),
            "online-fixed-variance",
            metric='auc'
    )
    
 
    do_plot(functools.partial(generate_ours_offline),
            "offline",
            metric='auc'
    )

    do_plot(functools.partial(generate_ours_offline, fix_variance=True),
            "offline-fixed-variance",
            metric='auc'
    )

    do_plot(generate_global,
            "global-threshold",
            metric='auc'
    ) 
    
    '''
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-3,1)
    plt.ylim(1e-3,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig("./fprtpr.png")      
    '''

import sys
if __name__ == '__main__': 
    load_data()
    fig_fpr_tpr()
