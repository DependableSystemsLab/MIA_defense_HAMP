import sys
import argparse
import os 
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=10000)
parser.add_argument('--train_org', type=int, default=0, help='flag for training teacher model')
parser.add_argument('--train_selena', type=int, default=0, help='flag for performing knowledge distillation')
parser.add_argument('--distill_lr', type=float, default=0.5, help='lr for distilation')
parser.add_argument('--lr', type=float, default=0.1, help='lr for the teacher model')
parser.add_argument('--K', type=int, default=25, help='num of teacher models')
parser.add_argument('--L', type=int, default=10, help='num of models that are not trained on each samples')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_tag', type=str, default='0', help='current shadow model index')
parser.add_argument('--total_models', type=int, default=128)
parser.add_argument('--folder_tag', type=str, default='selena')
parser.add_argument('--res_folder', type=str, default='lira-selena-fullMember')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
from util.densenet import densenet
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
user_lr=args.lr
org_model_checkpoint_dir='./shadow-{}-trainSize-{}-fullMember'.format(args.folder_tag, str(  args.train_size ) )
if(not os.path.exists(org_model_checkpoint_dir)):
    try:
        os.mkdir(org_model_checkpoint_dir)
    except:
        print('folder existed!')
is_train_org = args.train_org
is_train_ref = args.train_selena
BATCH_SIZE =64 
private_data_len= int(args.train_size * 0.9) 
ref_data_len = 0
te_len= int(args.train_size * 0.1)
val_len=int(args.train_size * 0.1)
attack_tr_len=private_data_len 
attack_te_len=0  
num_epochs=100
distil_epochs=200
tr_frac=0.45  
val_frac=0.05  
te_frac=0.5 
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

import random
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

dataloader = datasets.CIFAR10
num_classes = 10

trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
testset = dataloader(root='./data', train=False, download=True, transform=transform_test)

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
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [30, 60, 90]: 
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.



print('total data len: ',len(X), flush=True)
print(X.shape, Y.shape)


if not os.path.isfile('./cifar_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open('./cifar_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./cifar_shuffle.pkl','rb'))



private_data=X[all_indices[:private_data_len]]
private_label=Y[all_indices[:private_data_len]]



ref_data=X[all_indices[private_data_len : (private_data_len + ref_data_len)]]
ref_label=Y[all_indices[private_data_len : (private_data_len + ref_data_len)]]



val_data=X[all_indices[(private_data_len + ref_data_len):(private_data_len + ref_data_len + val_len)]]
val_label=Y[all_indices[(private_data_len + ref_data_len):(private_data_len + ref_data_len + val_len)]]

te_data=X[all_indices[(private_data_len + ref_data_len + val_len):(private_data_len + ref_data_len + val_len + te_len)]]
te_label=Y[all_indices[(private_data_len + ref_data_len + val_len):(private_data_len + ref_data_len + val_len + te_len)]]


 

attack_te_data=X[all_indices[(private_data_len + ref_data_len + val_len+ te_len):(private_data_len + ref_data_len + val_len+ te_len + attack_te_len)]]
attack_te_label=Y[all_indices[(private_data_len + ref_data_len + val_len+ te_len):(private_data_len + ref_data_len + val_len+ te_len + attack_te_len)]]

attack_tr_data=X[all_indices[(private_data_len + ref_data_len + val_len+ te_len + attack_te_len):(private_data_len + ref_data_len + val_len+ te_len + attack_te_len + attack_tr_len)]]
attack_tr_label=Y[all_indices[(private_data_len + ref_data_len + val_len+ te_len + attack_te_len):(private_data_len + ref_data_len + val_len+ te_len + attack_te_len + attack_tr_len)]]


remaining_data=X[all_indices[(private_data_len + ref_data_len + val_len+ te_len + attack_te_len + attack_tr_len):]]
remaining_label=Y[all_indices[(private_data_len + ref_data_len + val_len+ te_len + attack_te_len + attack_tr_len):]]



val_data = remaining_data[:val_len]
val_label = remaining_label[:val_len]
te_data = remaining_data[val_len:val_len+te_len]
te_label = remaining_label[val_len:val_len+te_len]

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


# the actual private data for shadow training might vary
private_data_len = len(private_data)






# get private data and label tensors required to train the unprotected model
private_data_tensor=torch.from_numpy(private_data).type(torch.FloatTensor)
private_label_tensor=torch.from_numpy(private_label).type(torch.LongTensor)


# get reference data and label tensors required to distil the knowledge into the protected model
ref_indices=np.arange((ref_data_len))
ref_data_tensor=torch.from_numpy(ref_data).type(torch.FloatTensor)
ref_label_tensor=torch.from_numpy(ref_label).type(torch.LongTensor)


## Member tensors required to train, validate, and test the MIA model:
# get member data and label tensors required to train MIA model
mia_train_members_data_tensor=private_data_tensor[:int(tr_frac*private_data_len)]
mia_train_members_label_tensor=private_label_tensor[:int(tr_frac*private_data_len)]

# get member data and label tensors required to validate MIA model
mia_val_members_data_tensor=private_data_tensor[int(tr_frac*private_data_len):int((tr_frac+val_frac)*private_data_len)]
mia_val_members_label_tensor=private_label_tensor[int(tr_frac*private_data_len):int((tr_frac+val_frac)*private_data_len)]

# get member data and label tensors required to test MIA model
mia_test_members_data_tensor=private_data_tensor[int((tr_frac+val_frac)*private_data_len):]
mia_test_members_label_tensor=private_label_tensor[int((tr_frac+val_frac)*private_data_len):]



## Non-member tensors required to train, validate, and test the MIA model:
attack_tr_data_tensors = torch.from_numpy(attack_tr_data).type(torch.FloatTensor)
attack_tr_label_tensors = torch.from_numpy(attack_tr_label).type(torch.LongTensor)

# get non-member data and label tensors required to train 
mia_train_nonmembers_data_tensor = attack_tr_data_tensors[:int(tr_frac*private_data_len)]
mia_train_nonmembers_label_tensor = attack_tr_label_tensors[:int(tr_frac*private_data_len)]

# get member data and label tensors required to validate MIA model
mia_val_nonmembers_data_tensor = attack_tr_data_tensors[int(tr_frac*private_data_len):int((tr_frac+val_frac)*private_data_len)]
mia_val_nonmembers_label_tensor = attack_tr_label_tensors[int(tr_frac*private_data_len):int((tr_frac+val_frac)*private_data_len)]

# get member data and label tensors required to test MIA model
mia_test_nonmembers_data_tensor = attack_tr_data_tensors[int((tr_frac+val_frac)*private_data_len):]
mia_test_nonmembers_label_tensor = attack_tr_label_tensors[int((tr_frac+val_frac)*private_data_len):]



# get non-member data and label tensors required to test the MIA model
attack_te_data_tensor=torch.from_numpy(attack_te_data).type(torch.FloatTensor)
attack_te_label_tensor=torch.from_numpy(attack_te_label).type(torch.LongTensor)


## Tensors required to validate and test the unprotected and protected models
# get validation data and label tensors
val_data_tensor=torch.from_numpy(val_data).type(torch.FloatTensor)
val_label_tensor=torch.from_numpy(val_label).type(torch.LongTensor)

# get test data and label tensors
te_data_tensor=torch.from_numpy(te_data).type(torch.FloatTensor)
te_label_tensor=torch.from_numpy(te_label).type(torch.LongTensor)
 



print('tr len %d | mia_members tr %d val %d te %d | mia_nonmembers tr %d val %d te %d | ref len %d | val len %d | test len %d | attack tr len %d | remaining data len %d'%
      (len(private_data_tensor),len(mia_train_members_data_tensor),len(mia_val_members_data_tensor),len(mia_test_members_data_tensor),
       len(mia_train_nonmembers_data_tensor), len(mia_val_nonmembers_data_tensor),len(mia_test_nonmembers_data_tensor),
       len(ref_data_tensor),len(val_data_tensor),len(te_data_tensor),len(attack_tr_data_tensors), len(remaining_data)), flush=True)





'''
For each training sample, we randomly select L models from the K teacher models and store them in ``non_model_indices_for_each_sample''
    These L models are the models that are not trained on the specific training samples
For each teacher model, we also generate a list ``sub_model_indices'' to its reference set (on which the teacher model should be trained)
'''
np.random.seed(0)
K = args.K
L = args.L
 
sub_model_indices = [[] for _ in range(K)]
non_model_indices_for_each_sample = np.zeros((private_data_len, L))

# partition this into K lists
training_indices = np.where(keep==True)[0] # indice to the training data in all shadow data   
#print(   training_indices.shape)
#print( len(training_indices), all_non_private_label[training_indices[0]], all_non_private_label[training_indices[10]] )
for cnt, each_ind in enumerate(training_indices): 
    non_model_indices = np.random.choice(K, L, replace=False) # out of K teacher models, L of them will not be trained on the current sample
    non_model_indices_for_each_sample[cnt] = non_model_indices # L indices for each sample 

    for i in range(K):
        if(i not in non_model_indices):
            # the current index will be stored for the i_th teacher model
            # these are the ``reference set'' that will be used for training the sub model
            sub_model_indices[i].append(each_ind)

sub_model_indices = np.asarray(sub_model_indices)
non_model_indices_for_each_sample = np.asarray(non_model_indices_for_each_sample)



def train_pub(train_data, labels, true_labels, model, t_softmax, optimizer, num_batchs=999999, batch_size=16, alpha=1):
    # switch to train mode
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    true_criterion=nn.CrossEntropyLoss()

    len_t = len(train_data)//batch_size
    if len(train_data) % batch_size:
        len_t += 1
    
    for ind in range(len_t):
        if ind > num_batchs:
            break

        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        true_targets=true_labels[ind*batch_size:(ind+1)*batch_size]
        inputs, targets, true_targets = inputs.to(device), targets.to(device), true_targets.to(device)
        inputs, targets, true_targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(true_targets)
        outputs  = model(inputs)
        loss = alpha*F.kl_div(F.log_softmax(outputs/t_softmax, dim=1), F.softmax(targets/t_softmax, dim=1)) + (1-alpha)*true_criterion(outputs,true_targets)
        losses.update(loss.item(), inputs.size(0)) 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg)


def train(train_data,labels,model,criterion,optimizer,epoch,use_cuda,num_batchs=999999,batch_size=32, uniform_reg=False):
    # switch to train mode
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    len_t = len(train_data)//batch_size
    if len(train_data)%batch_size:
        len_t += 1
    
    
    for ind in range(len_t): 

        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs  = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)


import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s



elapsed_time = 0
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
if(is_train_org): 
    
    for sub in range(K):
        
        model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0)
        model=model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=user_lr, momentum=0.9, weight_decay=1e-4)  
        best_val_acc=0
        best_test_acc=0        
        # derive the ``reference set'' for training the teacher model
        sub_train_x = all_non_private_data[sub_model_indices[sub]]
        sub_train_y = all_non_private_label[sub_model_indices[sub]]
        x_tensor = torch.from_numpy(sub_train_x).type(torch.FloatTensor)
        y_tensor = torch.from_numpy(sub_train_y).type(torch.LongTensor) 

        for epoch in range(num_epochs):
            start_time = time.time()
            adjust_learning_rate(optimizer, epoch)
            train_loss, train_acc = train(x_tensor, y_tensor, model, criterion, optimizer, epoch, use_cuda, batch_size= BATCH_SIZE, uniform_reg=False)

            val_loss, val_acc = test(val_data_tensor, val_label_tensor, model, criterion, use_cuda, device=device)

            is_best = val_acc > best_val_acc
            best_val_acc=max(val_acc, best_val_acc)

            if is_best:
                _, best_test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda)
                test_acc = best_test_acc
            else:
                _, test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda)

            save_checkpoint_global(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_val_acc,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                checkpoint=org_model_checkpoint_dir,
                filename='unprotected_model_sub_%s-%d.pth.tar'%(str(sub),save_tag),
                best_filename='unprotected_model_best_sub_%s-%d.pth.tar'%(str(sub),save_tag)
            )

            print('%d sub model |  epoch %d | tr acc %.2f loss %.2f | val acc %.2f loss %.2f | test acc %.2f | best val acc %.2f | best te acc %.2f'
                  %(sub, epoch, train_acc, train_loss, val_acc, val_loss, test_acc, best_val_acc, best_test_acc), flush=True)

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))


# distil the knowledge of the unprotected model in the ref data 
criterion=nn.CrossEntropyLoss()
sub_models = [[] for _ in range(K)]
for i in range(K):
    best_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).to(device)
    resume_best=os.path.join( org_model_checkpoint_dir,'unprotected_model_best_sub_%s-%d.pth.tar'%(str(i),save_tag))
    assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model %s'%resume_best
    checkpoint = os.path.dirname(resume_best)
    checkpoint = torch.load(resume_best)
    best_model.load_state_dict(checkpoint['state_dict'])
    _,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda)
    _,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda)
    _,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda)
    print(' %s sub model: train acc %.4f val acc %.4f test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)
    sub_models[i] = best_model




if(not args.train_selena):
    sys.exit() 




all_outputs=[]
non_model_indices_for_each_sample = non_model_indices_for_each_sample.astype(int)
outputs_avg = np.zeros((L, num_classes))
 

for cnt, each_ind in enumerate(training_indices):  
    inputs = private_data_tensor[cnt:cnt+1]
    inputs = inputs.to(device) 
    first = True
    # adaptive inference on the sub model, each model only predicts on the samples that were NOT used for training
    for i, model_index in enumerate(non_model_indices_for_each_sample[cnt]): 
        sub_model = sub_models[model_index]
        outputs = sub_model(inputs)
        if(first):
            outs = outputs 
            first=False
        else:
            outs = torch.cat( (outs, outputs), 0)
    # aggregate all the scores to generate soft labels for distillation
    all_outputs.append( torch.mean(outs, dim=0).cpu().detach().numpy() )
all_outputs = np.asarray(all_outputs) 


distil_label_tensor=(torch.from_numpy(all_outputs).type(torch.FloatTensor))



# train final protected model via knowledge distillation
distil_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).to(device)
distil_test_criterion=nn.CrossEntropyLoss()
distil_schedule=[60, 90, 150]
distil_lr=args.distill_lr
distil_best_acc=0
best_distil_test_acc=0
gamma=.1
t_softmax=1
if(is_train_ref):
    
    for epoch in range(distil_epochs):
        start_time = time.time()
        if epoch in distil_schedule:
            distil_lr *= gamma
            print('----> Epoch %d distillation lr %f'%(epoch,distil_lr))

        distil_optimizer=optim.SGD(distil_model.parameters(), lr=distil_lr, momentum=0.99, weight_decay=1e-5)

        distil_tr_loss = train_pub(private_data_tensor, distil_label_tensor, private_label_tensor, distil_model, t_softmax,
                                   distil_optimizer, batch_size=BATCH_SIZE, alpha=1)

        tr_loss,tr_acc = test(private_data_tensor, private_label_tensor, distil_model, distil_test_criterion, use_cuda, device=device)
        
        val_loss,val_acc = test(val_data_tensor, val_label_tensor, distil_model, distil_test_criterion, use_cuda, device=device)

        distil_is_best = val_acc > distil_best_acc

        distil_best_acc=max(val_acc, distil_best_acc)

        if distil_is_best:
            _,best_distil_test_acc = test(te_data_tensor, te_label_tensor, distil_model, distil_test_criterion, use_cuda, device=device)

        
        save_checkpoint_global(
            {
                'epoch': epoch,
                'state_dict': distil_model.state_dict(),
                'best_acc': distil_best_acc,
                'optimizer': distil_optimizer.state_dict(),
            },
            distil_is_best,
            checkpoint=org_model_checkpoint_dir,
            filename='protected_model-%d.pth.tar'%(save_tag),
            best_filename='protected_model_best-%d.pth.tar'%(save_tag),
        )
        print('epoch %d | distil loss %.4f | tr loss %.4f tr acc %.4f | val loss %.4f val acc %.4f | best val acc %.4f | best test acc %.4f'%(epoch,distil_tr_loss,tr_loss,tr_acc,val_loss,val_acc,distil_best_acc,best_distil_test_acc),flush=True)

        elapsed_time += time.time() - start_time
        print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))


criterion=nn.CrossEntropyLoss()
best_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).to(device)
resume_best=os.path.join(org_model_checkpoint_dir, 'protected_model_best-%d.pth.tar'%(save_tag))
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda, device=device)
print('\t===> SELENA %s | train acc %.4f | val acc %.4f | test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)

