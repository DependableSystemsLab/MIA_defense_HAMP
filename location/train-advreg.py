import argparse
import os
import shutil
import time
import random
import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import numpy as np
import argparse
parser = argparse.ArgumentParser()   
parser.add_argument('--train_size', type=int, default=5000)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--alpha', type=int, default=3)
args = parser.parse_args()
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
private_data_len= int(args.train_size * 0.9)  
attack_tr_len=private_data_len
attack_te_len=0  
tr_frac=0.45 # we use 45% of private data as the members to train MIA model
val_frac=0.05 # we use 5% of private data as the members to validate MIA model
te_frac=0.5 # we use 50% of private data as the members to test MIA model
print("loading data", flush=True)
data_set_features= np.load('./location-features.npy') 
data_set_label= np.load('./location-labels.npy') 
X =data_set_features.astype(np.float64)
Y = data_set_label.astype(np.int32)
num_classes = 30
print(Y)
print('total data len: ',len(X), flush=True)
print(X.shape, Y.shape)
if not os.path.isfile('./location_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open('./location_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./location_shuffle.pkl','rb'))

tr_len= int(args.train_size * 0.9) # private data for training the defended model
ref_len = tr_len # reference set as non-member to train the defended model 
te_len= 1000
val_len=int(args.train_size * 0.2)
print('generating data for adversarial tuning...')
tr_data=X[all_indices[:tr_len]]
tr_label=Y[all_indices[:tr_len]]


ref_data=X[all_indices[tr_len:(tr_len+ref_len)]]
ref_label=Y[all_indices[tr_len:(tr_len+ref_len)]]

val_data=X[all_indices[(tr_len+ref_len):(tr_len+ref_len+val_len)]]
val_label=Y[all_indices[(tr_len+ref_len):(tr_len+ref_len+val_len)]]

te_data=X[all_indices[(tr_len+ref_len+val_len):(tr_len+ref_len+val_len+te_len)]]
te_label=Y[all_indices[(tr_len+ref_len+val_len):(tr_len+ref_len+val_len+te_len)]]




tr_cls_data_tensor=torch.from_numpy(tr_data).type(torch.FloatTensor)
tr_cls_label_tensor=torch.from_numpy(tr_label).type(torch.LongTensor)

tr_len=len(tr_cls_data_tensor)



tr_cls_tr_at_data_tensor=tr_cls_data_tensor[:int(tr_frac*tr_len)]
tr_cls_tr_at_label_tensor=tr_cls_label_tensor[:int(tr_frac*tr_len)]


tr_cls_val_at_data_tensor=tr_cls_data_tensor[int(tr_frac*tr_len):int((tr_frac+val_frac)*tr_len)]
tr_cls_val_at_label_tensor=tr_cls_label_tensor[int(tr_frac*tr_len):int((tr_frac+val_frac)*tr_len)]


tr_cls_te_at_data_tensor=tr_cls_data_tensor[int((tr_frac+val_frac)*tr_len):]
tr_cls_te_at_label_tensor=tr_cls_label_tensor[int((tr_frac+val_frac)*tr_len):]



ref_data_tensor=torch.from_numpy(ref_data).type(torch.FloatTensor)
ref_label_tensor=torch.from_numpy(ref_label).type(torch.LongTensor)


val_data_tensor=torch.from_numpy(val_data).type(torch.FloatTensor)
val_label_tensor=torch.from_numpy(val_label).type(torch.LongTensor)    
te_data_tensor=torch.from_numpy(te_data).type(torch.FloatTensor)
te_label_tensor=torch.from_numpy(te_label).type(torch.LongTensor)

print('tr len %d at_tr len %d at_val len %d at_te len %d ref len %d val len %d test len %d'%
      (len(tr_data),len(tr_cls_tr_at_data_tensor),len(tr_cls_val_at_data_tensor),len(tr_cls_te_at_data_tensor),len(ref_data),len(val_data),len(te_data)))


checkpoint_dir='./advreg-model'
if(not os.path.exists(checkpoint_dir)):
    os.mkdir(checkpoint_dir)

        
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
#         for key in self.state_dict():
#             if key.split('.')[-1] == 'weight':    
#                 nn.init.normal(self.state_dict()[key], std=0.01)
#                 print (key)
                
#             elif key.split('.')[-1] == 'bias':
#                 self.state_dict()[key][...] = 0
        
    def forward(self,x):
        hidden_out = self.features(x)
        
        
        
        return self.classifier(hidden_out),hidden_out#, hidden_out


def advtune_defense(num_epochs=50, use_cuda=True,batch_size=64,alpha=0,lr=0.0005,schedule=[25,80],gamma=0.1,tr_epochs=100,at_lr=0.0001,at_schedule=[100],at_gamma=0.5,at_epochs=200,n_classes=30):
 
    ############################################################ private training ############################################################

    print('Training using adversarial tuning...')

    model=LocationClassifier()
    optimizer=optim.Adam(model.parameters(), lr=lr)
    criterion=nn.CrossEntropyLoss()

    attack_model=InferenceAttack_HZ(n_classes)
    
    attack_optimizer=optim.Adam(attack_model.parameters(),lr=at_lr)
    attack_criterion=nn.MSELoss()

    if(use_cuda):
        attack_model=attack_model.cuda()
        model=model.cuda()




    best_acc=0
    best_test_acc=0    
    for epoch in range(num_epochs):
        if epoch in schedule:
            # decay the lr at certain epoches in schedule
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
                print('Epoch %d Local lr %f'%(epoch,param_group['lr']))
 
        c_batches = len(tr_cls_data_tensor)//batch_size
        if epoch == 0:
            print('----> NORMAL TRAINING MODE: c_batches %d '%(c_batches))


            train_loss, train_acc = train(tr_cls_data_tensor,tr_cls_label_tensor,
                                              model,criterion,optimizer,epoch,use_cuda,debug_='MEDIUM')    
            test_loss, test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda)    
            for i in range(5):
                at_loss, at_acc = train_attack(tr_cls_data_tensor,tr_cls_label_tensor,
                                               ref_data_tensor,ref_label_tensor,model,attack_model,criterion,
                                               attack_criterion,optimizer,attack_optimizer,epoch,use_cuda,debug_='MEDIUM')    

            print('Initial test acc {} train att acc {}'.format(test_acc, at_acc))

        else: 
            for e_num in schedule:
                if e_num==epoch:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= gamma
                        print('Epoch %d lr %f'%(epoch,param_group['lr']))

            att_accs =[] 
            rounds=(c_batches//2)

            for i in range(rounds):
 
                at_loss, at_acc = train_attack(tr_cls_data_tensor,tr_cls_label_tensor,
                                               ref_data_tensor,ref_label_tensor,
                                               model,attack_model,criterion,attack_criterion,optimizer,
                                               attack_optimizer,epoch,use_cuda,52,(i*52)%c_batches,batch_size=batch_size)

                att_accs.append(at_acc)

                tr_loss, tr_acc = train_privatly(tr_cls_data_tensor,tr_cls_label_tensor,model,
                                                 attack_model,criterion,optimizer,epoch,use_cuda,
                                                 2,(2*i)%c_batches,alpha=alpha,batch_size=batch_size)

            train_loss,train_acc = test(tr_cls_data_tensor,tr_cls_label_tensor,model,criterion,use_cuda)
            val_loss, val_acc = test(val_data_tensor,val_label_tensor,model,criterion,use_cuda)
            is_best = (val_acc > best_acc)

            if is_best:
                _, best_test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda)
                print("saving the best")

            best_acc=max(val_acc, best_acc)

            at_val_loss, at_val_acc = test_attack(tr_cls_te_at_data_tensor,tr_cls_te_at_label_tensor,
                                                     te_data_tensor,te_label_tensor,
                                                     model,attack_model,criterion,attack_criterion,
                                                     optimizer,attack_optimizer,epoch,use_cuda,debug_='MEDIUM')
            
            att_epoch_acc = np.mean(att_accs)
            
            save_checkpoint_global(
               {
                   'epoch': epoch,
                   'state_dict': model.state_dict(),
                   'acc': val_acc,
                   'best_acc': best_acc,
                   'optimizer': optimizer.state_dict(),
               },
               is_best,
               checkpoint=checkpoint_dir,
               filename='checkpoint_lr_%s_alpha_%d_trLen_%d_refLen_%d.pth.tar'%(str(args.lr).replace('.','_'), alpha,tr_len,ref_len),
               best_filename='mdoel_best_lr_%s_alpha_%d_trLen_%d_refLen_%d.pth.tar'%(str(args.lr).replace('.','_'), alpha,tr_len,ref_len),
            )
          
            print('epoch %d | tr_acc %.2f | val acc %.2f | best val acc %.2f | best te acc %.2f | attack avg acc %.2f | attack val acc %.2f'%(epoch,train_acc,val_acc,best_acc,best_test_acc,att_epoch_acc,at_val_acc), flush=True)

    ############################################################ private training ############################################################

advtune_defense(alpha=args.alpha, num_epochs=50, use_cuda=True, lr=args.lr)


best_model=LocationClassifier()
criterion=nn.CrossEntropyLoss()
use_cuda = True

resume_best=os.path.join(checkpoint_dir,'mdoel_best_lr_%s_alpha_%d_trLen_%d_refLen_%d.pth.tar'%(str(args.lr).replace('.','_'), args.alpha,tr_len,ref_len))

print(resume_best)
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
best_model = best_model.to(device)

_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda)
_,best_train = test(tr_cls_data_tensor, tr_cls_label_tensor, best_model, criterion, use_cuda)
print('\t===>  %s  AdvReg model: train acc %.4f val acc %.4f test acc %.4f'%(resume_best, best_train,best_val, best_test), flush=True)


