import sys
sys.path.insert(0,'./util/')
import os
import argparse
parser = argparse.ArgumentParser()   
parser.add_argument('--isTrain', type=int, default = 0)
parser.add_argument('--train_size', type=int, default=10000)
parser.add_argument('--distill_lr', type=float, default=0.5, help='lr for hamp training')
parser.add_argument('--entropy_percentile', type=float, default=0.95, help="gamma parameter in hamp")
parser.add_argument('--entropy_penalty', type=int, default=0, help='flag to indicate whether to use regularization or not')
parser.add_argument('--alpha', type=float, default=1., help='alpha parameter in hamp')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_tag', type=str, default='0', help='current shadow model index')
parser.add_argument('--folder_tag', type=str, default='hamp')
parser.add_argument('--res_folder', type=str, default='lira-hamp-fullMember')
parser.add_argument('--total_models', type=int, default=128)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
from torch.nn import functional as F
from util.densenet import densenet
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
seed = 1
np.random.seed(seed)
float_formatter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
new_model_checkpoint_dir='./shadow-{}-trainSize-{}-fullMember'.format(args.folder_tag, str(  args.train_size ) )
if(not os.path.exists(new_model_checkpoint_dir)):
    try:
        os.mkdir(new_model_checkpoint_dir)
    except:
        print('folder exists already')


is_train_ref = args.isTrain 
BATCH_SIZE =64 
private_data_len= int(args.train_size * 0.9) 
ref_data_len = 0

te_len= int(args.train_size * 0.1)
val_len=int(args.train_size * 0.1)
attack_tr_len=private_data_len
attack_te_len=0 
distil_epochs=200
tr_frac=0.45  
val_frac=0.05  
te_frac=0.5 
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
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

 
def train_pub(train_data, labels, true_labels, model, t_softmax, optimizer, num_batchs=999999, batch_size=16, alpha=0.05):
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
        if(not args.entropy_penalty): 
            loss = F.kl_div(F.log_softmax(outputs, dim=1), targets )  
        else:
            entropy = Categorical(probs = F.softmax(outputs, dim=1)).entropy()
            loss1 = F.kl_div(F.log_softmax(outputs, dim=1), targets )   
            loss2 = -1 * alpha * torch.mean(entropy)
            loss = loss1 + loss2
        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg)





def entropy(preds, axis=0):
    logp = np.log(preds)
    entropy = np.sum( -preds * logp , axis=axis ) 
    return entropy

def get_top1(num_classes, entropy_threshold, reduced_prob=0.01):
    # reduced_prob : for reducing top-1 class's probability
    true_target = 1
    preds = np.zeros(num_classes)
    preds[true_target]  = 1.
    while(True):
        preds[true_target] -= reduced_prob
        preds[:true_target] += reduced_prob/(num_classes-1) 
        preds[true_target+1:] += reduced_prob/(num_classes-1) 
        if(entropy(preds) >= entropy_threshold):
            break
    return preds[true_target], preds[true_target+1]


def get_soft_labels(train_label, num_classes, top1, uniform_non_top1):
    new_soft_label = np.zeros( (train_label.shape[0], num_classes) ) 
    for i in range( train_label.shape[0] ):
        new_soft_label[i][train_label[i]] = top1
        new_soft_label[i][:train_label[i]] = uniform_non_top1
        new_soft_label[i][train_label[i]+1:] = uniform_non_top1
    print( new_soft_label[0], train_label[0], np.argmax(new_soft_label[0]) )
    print( 'top1 %.6f | non-top1 %.6f'%(np.max(new_soft_label[0]), np.min(new_soft_label[0])))
    return new_soft_label


use_cuda = torch.cuda.is_available()
num_class = num_classes
preds = np.ones(num_class)
preds /= float(num_class)   
highest_entropy = entropy(preds) 
# assign uniform class prob for all the non-top-1 classes
top1, uniform_non_top1 = get_top1(num_class, highest_entropy*args.entropy_percentile)
print("Highest entropy {:.4f} | entropy_percentile {:.4f} | entropy threshold {:.4f}".format(highest_entropy , args.entropy_percentile, highest_entropy*args.entropy_percentile))
private_label_modified = get_soft_labels(private_label, num_class, top1, uniform_non_top1)
private_label_modified_tensor=torch.from_numpy(private_label_modified).type(torch.FloatTensor)





 
distil_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).to(device)
distil_test_criterion=nn.CrossEntropyLoss()
distil_schedule=[60, 90, 150]
distil_lr=args.distill_lr
distil_best_acc=0
best_distil_test_acc=0
gamma=.1
t_softmax=1
best_train_acc = 0.


import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

elapsed_time = 0
if(is_train_ref):
    for epoch in range(distil_epochs): 
        start_time = time.time()
        if epoch in distil_schedule:
            distil_lr *= gamma
            print('  ----> Epoch %d   lr %f'%(epoch,distil_lr), flush=True)

        distil_optimizer=optim.SGD(distil_model.parameters(), lr=distil_lr, momentum=0.99, weight_decay=1e-5)
        distil_tr_loss = train_pub(private_data_tensor, private_label_modified_tensor, private_label_tensor, distil_model, t_softmax,
                                   distil_optimizer, batch_size=BATCH_SIZE, alpha=args.alpha)

        tr_loss,tr_acc = test(private_data_tensor, private_label_tensor, distil_model, distil_test_criterion, use_cuda, device=device)
        
        val_loss,val_acc = test(val_data_tensor, val_label_tensor, distil_model, distil_test_criterion, use_cuda, device=device)
        # the validation acy needs to increase by at least 1% 
        distil_is_best = val_acc > (distil_best_acc + 1.0)
        if distil_is_best:
            _,best_distil_test_acc = test(te_data_tensor, te_label_tensor, distil_model, distil_test_criterion, use_cuda, device=device)
            distil_best_acc=max(val_acc, distil_best_acc)
            best_train_acc = tr_acc
        save_checkpoint_global(
            {
                'epoch': epoch,
                'state_dict': distil_model.state_dict(),
                'best_acc': distil_best_acc,
                'optimizer': distil_optimizer.state_dict(),
            },
            distil_is_best,
            checkpoint=new_model_checkpoint_dir,
            filename='protected_model-%d.pth.tar'%(save_tag),
            best_filename='protected_model_best-%d.pth.tar'%(save_tag),
        )

        print('lr %.5f | alpha %.3f epoch %d | loss %.4f | tr acc %.4f | val acc %.4f || best train acc %.4f | best val acc %.4f | best test acc %.4f'%(distil_lr, args.alpha, epoch,distil_tr_loss,tr_acc,val_acc, best_train_acc, distil_best_acc,best_distil_test_acc), flush=True)

        elapsed_time += time.time() - start_time
        print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))



criterion=nn.CrossEntropyLoss()
best_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).to(device)
resume_best=os.path.join(new_model_checkpoint_dir, 'protected_model_best-%d.pth.tar'%(save_tag))


assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda, device=device)
print('\t===>  %s  HAMP model: train acc %.4f val acc %.4f test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)
