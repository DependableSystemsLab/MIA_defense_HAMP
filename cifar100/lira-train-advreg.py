import sys
import argparse
import os
parser = argparse.ArgumentParser()   
parser.add_argument('--train_size', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--alpha', type=int, default=6)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_tag', type=str, default='0', help='current shadow model index')
parser.add_argument('--folder_tag', type=str, default='advreg')
parser.add_argument('--res_folder', type=str, default='lira-advreg-fullMember')
parser.add_argument('--total_models', type=int, default=1)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import torch.optim as optim
import argparse
import os
import shutil
import time
import random
import torch.nn.functional as F
from densenet_advreg import densenet
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr_frac=0.45 
val_frac=0.05 
te_frac=0.5 
random.seed(1)
# prepare test data parts
transform_train = transforms.Compose([  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataloader = datasets.CIFAR100
num_classes = 100
trainset = dataloader(root='./data', train=True, download=False, transform=transform_train)
testset = dataloader(root='./data', train=False, download=False, transform=transform_test)

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

print("loading data")

print('total data len: ',len(X), flush=True)
print(X.shape, Y.shape)


if not os.path.isfile('./cifar_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open('./cifar_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./cifar_shuffle.pkl','rb'))



 


tr_len= int(args.train_size * 0.9) # private data for training the defended model
ref_len = tr_len  
te_len= int(args.train_size * 0.1)
val_len=int(args.train_size * 0.1)



print('generating data for adversarial tuning...', flush=True)


tr_data=X[all_indices[:tr_len]]
tr_label=Y[all_indices[:tr_len]]


ref_data=X[all_indices[tr_len:(tr_len+ref_len)]]
ref_label=Y[all_indices[tr_len:(tr_len+ref_len)]]

val_data=X[all_indices[(tr_len+ref_len):(tr_len+ref_len+val_len)]]
val_label=Y[all_indices[(tr_len+ref_len):(tr_len+ref_len+val_len)]]

te_data=X[all_indices[(tr_len+ref_len+val_len):(tr_len+ref_len+val_len+te_len)]]
te_label=Y[all_indices[(tr_len+ref_len+val_len):(tr_len+ref_len+val_len+te_len)]]





private_data_len = tr_len
all_non_private_data = X[all_indices[:private_data_len*2]]
all_non_private_label = Y[all_indices[:private_data_len*2]]
 

 
print('shadow data and label len ', all_non_private_data.shape, all_non_private_label.shape)
save_tag = int(args.save_tag)  

print('\tCurrent training id ', save_tag)
np.random.seed( 0 )



len_all_non_private_data = len(all_non_private_data)
keep = np.random.uniform(0,1,size=(args.total_models, len_all_non_private_data ))
order = keep.argsort(0)
keep = order < int( private_data_len/float(len_all_non_private_data) * args.total_models)
keep = np.array(keep[save_tag], dtype=bool)


private_data = all_non_private_data[keep]
private_label = all_non_private_label[keep]
print('first 20 labels ', private_label[:20])
print('total len of shadow training ', len(private_data), len_all_non_private_data)


lira_folder = args.res_folder
if(not os.path.exists(lira_folder)):
    try:
        os.mkdir(lira_folder)
    except:
        print('already existed')

np.save( os.path.join(lira_folder, '%s_keep.npy'%save_tag),  keep )


tr_data = private_data
tr_label = private_label


ref_data = all_non_private_data[~keep]
ref_label = all_non_private_label[~keep]



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
private_data_tensor=tr_cls_data_tensor
private_label_tensor=tr_cls_label_tensor



print('tr len %d at_tr len %d at_val len %d at_te len %d ref len %d val len %d test len %d'%
      (len(tr_data),len(tr_cls_tr_at_data_tensor),len(tr_cls_val_at_data_tensor),len(tr_cls_te_at_data_tensor),len(ref_data),len(val_data),len(te_data)), flush=True)

checkpoint_dir='./shadow-{}-trainSize-{}-fullMember'.format(args.folder_tag, str(  args.train_size ) )
if(not os.path.exists(checkpoint_dir)):
    try:
        os.mkdir(checkpoint_dir)
    except:
        print('folder existed!')

import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

global elapsed_time
elapsed_time = 0
start_time = time.time()

import torch.backends.cudnn as cudnn
def advtune_defense(num_epochs=50, use_cuda=True,batch_size=64,alpha=0,lr=0.0005,schedule=[25,80],gamma=0.1,tr_epochs=100,at_lr=0.0001,at_schedule=[100],at_gamma=0.5,at_epochs=200,n_classes=100):
 
    global elapsed_time
    ############################################################ private training ############################################################
    print('Training using adversarial tuning...')
    model=densenet(num_classes=n_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

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
        start_time = time.time()


        if epoch in schedule:
            # decay the lr at certain epoches in schedule
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
                print('Epoch %d Local lr %f'%(epoch,param_group['lr']))


        c_batches = len(tr_cls_data_tensor)//batch_size
        if epoch == 0:
            print('----> NORMAL TRAINING MODE: c_batches %d '%(c_batches), flush=True)


            train_loss, train_acc = train(tr_cls_data_tensor,tr_cls_label_tensor,
                                              model,criterion,optimizer,epoch,use_cuda,debug_='MEDIUM')    
            test_loss, test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda, batch_size=batch_size)    
            for i in range(5):
                at_loss, at_acc = train_attack(tr_cls_data_tensor,tr_cls_label_tensor,
                                               ref_data_tensor,ref_label_tensor,model,attack_model,criterion,
                                               attack_criterion,optimizer,attack_optimizer,epoch,use_cuda, batch_size=batch_size,debug_='MEDIUM')    

            print('Initial test acc {} train att acc {}'.format(test_acc, at_acc), flush=True)

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
                filename='protected_model-%d.pth.tar'%(save_tag),
                best_filename='protected_model_best-%d.pth.tar'%(save_tag),
            )
          
            print('epoch %d | tr_acc %.2f | val acc %.2f | best val acc %.2f | best te acc %.2f | attack avg acc %.2f | attack val acc %.2f'%(epoch,train_acc,val_acc,best_acc,best_test_acc,att_epoch_acc,at_val_acc), flush=True)


        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))
    ############################################################ private training ############################################################


advtune_defense(alpha=args.alpha, batch_size=args.batch_size, num_epochs=200, use_cuda=True, schedule = [60, 90, 150])



best_model=densenet(num_classes=100,depth=100,growthRate=12,compressionRate=2,dropRate=0)
criterion=nn.CrossEntropyLoss()
use_cuda = True
resume_best=os.path.join(checkpoint_dir,'protected_model_best-%d.pth.tar'%(save_tag))
 
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])

best_model = best_model.to(device)
_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda, device=device)
print(args.save_tag, '\t===> AdvReg model %s | train acc %.4f | val acc %.4f | test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)


