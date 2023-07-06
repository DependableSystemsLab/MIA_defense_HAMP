# We provide the shadow attack classifier used by MemGuard in the artifact
# You can also train a new shadow attack classifier for evaluation
# Below is an example
'''
python memguard-train-attack-classifier.py --dataset purchase --train_size 20000 \
        --model_path final-all-models/undefended-trainSize-20000.pth.tar  \
        --save_tag purchase_shadow_attack_classifier

'''

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='cifar10')
parser.add_argument('--model_path',default=None, help='target model')
parser.add_argument('--train_size', type=int, default=10000)
parser.add_argument('--save_tag', type=str, default='memguard_attack_classifier')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

import keras
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, concatenate
import numpy as np
np.random.seed(10000)
def model_defense(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(inputs_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=1000),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model

mia_num_classes=1
defense_epochs=400 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator

def _model_predictions(model, x, y, batch_size=256):
    return_outputs, return_labels = [], []
    return_labels = y.numpy() 
    len_t = len(x)//batch_size
    if len(x)%batch_size:
        len_t += 1

    first = True
    for ind in range(len_t):
        outputs = model.forward(x[ind*batch_size:(ind+1)*batch_size].cuda())
        outputs = outputs.data.cpu().numpy()
        if(first):
            outs = outputs
            first=False
        else:
            outs = np.r_[outs, outputs]
    outputs = outs
    return_outputs.append( softmax_by_row(outputs ) )
    return_outputs = np.concatenate(return_outputs)
    return (return_outputs, return_labels)

if(args.dataset=='cifar10'):
    import sys 
    sys.path.insert(0,'../cifar10/util/')
    from purchase_normal_train import *
    from purchase_private_train import *
    from purchase_attack_train import *
    from purchase_util import *
    from densenet import densenet
    pred_class =10
    best_model=densenet(num_classes=pred_class,depth=100,growthRate=12,compressionRate=2,dropRate=0).cuda()
    private_data_len= int(args.train_size * 0.9) 
    import random
    random.seed(1)
    transform_train = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR10
    trainset = dataloader(root='../cifar10/data', train=True, download=True, transform=transform_train)
    testset = dataloader(root='../cifar10/data', train=False, download=True, transform=transform_train)
    X = []
    Y = []
    for item in trainset: 
        X.append( item[0].numpy() )
        Y.append( item[1]  )
    for item in testset:
        X.append( item[0].numpy() )
        Y.append( item[1]  )
    X = np.asarray(X)
    Y = np.asarray(Y)
    all_indices=pickle.load(open('../cifar10/cifar_shuffle.pkl','rb'))


elif(args.dataset=='cifar100'):
    import sys 
    sys.path.insert(0,'../cifar100/util/')
    from purchase_normal_train import *
    from purchase_private_train import *
    from purchase_attack_train import *
    from purchase_util import *
    from densenet import densenet
    pred_class =100
    best_model=densenet(num_classes=pred_class,depth=100,growthRate=12,compressionRate=2,dropRate=0).cuda()
    private_data_len= int(args.train_size * 0.9) 
    import random
    random.seed(1)
    transform_train = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR100
    trainset = dataloader(root='../cifar100/data', train=True, download=True, transform=transform_train)
    testset = dataloader(root='../cifar100/data', train=False, download=True, transform=transform_train)
    X = []
    Y = []
    for item in trainset: 
        X.append( item[0].numpy() )
        Y.append( item[1]  )
    for item in testset:
        X.append( item[0].numpy() )
        Y.append( item[1]  )
    X = np.asarray(X)
    Y = np.asarray(Y)
    all_indices=pickle.load(open('../cifar100/cifar_shuffle.pkl','rb'))


elif(args.dataset=='purchase'):
    import sys
    sys.path.insert(0,'../purchase/util/')
    from purchase_normal_train import *
    from purchase_private_train import *
    from purchase_attack_train import *
    from purchase_util import *
    pred_class =100
    private_data_len= int(args.train_size * 0.9) 
    data_set= np.load('../purchase/purchase.npy')
    X = data_set[:,1:].astype(np.float64)
    Y = (data_set[:,0]).astype(np.int32)-1
    print('total data len: ',len(X), flush=True)

    all_indices=pickle.load(open('../purchase/purchase_shuffle.pkl','rb'))

    class PurchaseClassifier(nn.Module):
        def __init__(self,num_classes=100):
            super(PurchaseClassifier, self).__init__()
            self.features = nn.Sequential(
                nn.Linear(600,1024),
                nn.Tanh(),
                nn.Linear(1024,512),
                nn.Tanh(),
                nn.Linear(512,256),
                nn.Tanh(),
                nn.Linear(256,128),
                nn.Tanh(),
            )
            self.classifier = nn.Linear(128,num_classes)
            
        def forward(self,inp):
            
            outputs=[]
            x=inp
            module_list =list(self.features.modules())[1:]
            for l in module_list:
                x = l(x)
                outputs.append(x)
            y = x.view(inp.size(0), -1)
            o = self.classifier(y)
            return o #, outputs[-1].view(inp.size(0), -1), outputs[-4].view(inp.size(0), -1)

    best_model=PurchaseClassifier().cuda()

elif(args.dataset=='texas'):
    import sys
    sys.path.insert(0,'../texas/util/')
    from purchase_normal_train import *
    from purchase_private_train import *
    from purchase_attack_train import *
    from purchase_util import *
    private_data_len= int(args.train_size * 0.9) 
    data_set_features= np.load('../texas/texas100-features.npy') 
    data_set_label= np.load('../texas/texas100-labels.npy') 
    pred_class =100
    X =data_set_features.astype(np.float64)
    Y = data_set_label.astype(np.int32)-1
    all_indices=pickle.load(open('../texas/texas_shuffle.pkl','rb'))
    class PurchaseClassifier(nn.Module):
        def __init__(self,num_classes=100):
            super(PurchaseClassifier, self).__init__()

            self.features = nn.Sequential(
                nn.Linear(6169,1024),
                nn.Tanh(),
                nn.Linear(1024,512),
                nn.Tanh(),
                nn.Linear(512,256),
                nn.Tanh(),
                nn.Linear(256,128),
                nn.Tanh(),
            )
            self.classifier = nn.Linear(128,num_classes)
        def forward(self,x):
            hidden_out = self.features(x)
            return self.classifier(hidden_out) #,hidden_out, hidden_out
    best_model=PurchaseClassifier().to(device)

elif(args.dataset=='location'):
    import sys
    sys.path.insert(0,'../location/util/')
    from purchase_normal_train import *
    from purchase_private_train import *
    from purchase_attack_train import *
    from purchase_util import *
    private_data_len= int(args.train_size * 0.9) 
    data_set_features= np.load('../location/location-features.npy') 
    data_set_label= np.load('../location/location-labels.npy') 
    pred_class =30
    X =data_set_features.astype(np.float64)
    Y = data_set_label.astype(np.int32)
    all_indices=pickle.load(open('../location/location_shuffle.pkl','rb'))

    class LocationClassifier(nn.Module):
        def __init__(self,num_classes=30):
            super(LocationClassifier, self).__init__()
            self.features = nn.Sequential(
                nn.Linear(446,1024),
                nn.Tanh(),
                nn.Linear(1024,512),
                nn.Tanh(),
                nn.Linear(512,256),
                nn.Tanh(),
                nn.Linear(256,128),
                nn.Tanh(),
            )
            self.classifier = nn.Linear(128,num_classes)
            
        def forward(self,x):
            hidden_out = self.features(x)
            return self.classifier(hidden_out) 
    best_model=LocationClassifier()
 



resume_best=args.model_path
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model %s'%resume_best
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
best_model=best_model.cuda()


private_data=X[all_indices[:private_data_len]]
private_label=Y[all_indices[:private_data_len]]
# get private data and label tensors required to train the unprotected model
private_data_tensor=torch.from_numpy(private_data).type(torch.FloatTensor)
private_label_tensor=torch.from_numpy(private_label).type(torch.LongTensor)


test_data = X[all_indices[private_data_len:private_data_len*2]]
test_label = Y[all_indices[private_data_len:private_data_len*2]]
test_data_tensor=torch.from_numpy(test_data).type(torch.FloatTensor)
test_label_tensor=torch.from_numpy(test_label).type(torch.LongTensor)


criterion=nn.CrossEntropyLoss()
_,best_test = test(test_data_tensor, test_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda, device=device)

print('\t====> %s | train acy %.4f test acy %.4f'%(args.model_path, best_train, best_test))

f_train, _ = _model_predictions(best_model, torch.cat( (private_data_tensor, test_data_tensor )), 
                                    torch.cat((private_label_tensor, test_label_tensor)) )

l_train = np.r_[ np.ones(len(private_data_tensor)), np.zeros(len(test_label_tensor)) ]

import numpy as np
np.random.seed(1000)
import keras
from keras.models import Model
from keras.backend import set_session
from keras import backend as K
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
set_session(tf.Session())


batch_size=256
#######sort the confidence score
f_train=np.sort(f_train,axis=1)
input_shape=pred_class

model=model_defense(input_shape=input_shape,labels_dim=mia_num_classes)
model.compile(loss=keras.losses.binary_crossentropy,optimizer=tf.keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
model.summary()

b_train=f_train[:,:]
label_train=l_train[:]


index_array=np.arange(b_train.shape[0])
batch_num=np.int(np.ceil(b_train.shape[0]/batch_size))
for i in np.arange(defense_epochs):
    np.random.shuffle(index_array)
    for j in np.arange(batch_num):
        b_batch=b_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,b_train.shape[0])],:]
        y_batch=label_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,label_train.shape[0])]]
        model.train_on_batch(b_batch,y_batch)   

    if (i+1)%100==0:
        print("Epochs: {}".format(i))
        scores_train = model.evaluate(b_train, label_train, verbose=0)
        print('Train loss:', scores_train[0])
        print('Train accuracy:', scores_train[1])    

model.save('%s_MIA_model.h5'%(args.save_tag) ) 



