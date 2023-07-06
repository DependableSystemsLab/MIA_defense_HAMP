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

import sys
import numpy as np
import os
import multiprocessing as mp
import re
import argparse
parser = argparse.ArgumentParser()   
parser.add_argument('--res_folder', type=str, required=True)
args = parser.parse_args()
lira_folder = args.res_folder

for r, d, f in os.walk(lira_folder):
    for file in f: 
        if("logit" in file):

            opredictions = np.load( os.path.join(r, file) )

            #print(opredictions.shape)

            labels = np.load( os.path.join(lira_folder, 'shadow_label.npy' ) )
            ## Be exceptionally careful.
            ## Numerically stable everything, as described in the paper.
            predictions = opredictions - np.max(opredictions, axis=2, keepdims=True)
            predictions = np.array(np.exp(predictions), dtype=np.float64)
            predictions = predictions/np.sum(predictions,axis=2,keepdims=True)
            COUNT = predictions.shape[0]
            y_true = predictions[np.arange(COUNT),:,labels[:COUNT]]

            print('mean acc',np.mean(predictions[:,0,:].argmax(1)==labels[:COUNT]), flush=True)
            print()

            predictions[np.arange(COUNT),:,labels[:COUNT]] = 0
            y_wrong = np.sum(predictions, axis=2)
 
            logit = (np.log(y_true.mean((1))+1e-45) - np.log(y_wrong.mean((1))+1e-45)) 

            np.save(os.path.join(lira_folder, '%s'%file.replace('logit', 'score')), logit)



