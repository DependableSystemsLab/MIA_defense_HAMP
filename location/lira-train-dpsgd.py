import argparse
import numpy as np
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=10000) 
parser.add_argument('--dp_batchsize', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)') 
parser.add_argument('--dp_norm_clip', type=float, default=1.0, metavar='M',
                    help='L2 norm clip (default: 1.0)')
parser.add_argument('--dp_noise_multiplier', type=float, default=1.0, metavar='M',
                    help='Noise multiplier (default: 1.0)')
parser.add_argument('--dp_microbatches',type=int, default=1, metavar='N',
                    help='micro batch size')
parser.add_argument('--dp_delta', type=float, default=1e-5, metavar='M',
                    help='target delta')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save_tag', type=str, default='0', help='current shadow model index')
parser.add_argument('--total_models', type=int, default=1)
parser.add_argument('--folder_tag', type=str, default='hamp')
parser.add_argument('--res_folder', type=str, default='lira-hamp-fullMember')
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import sys 
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
org_model_checkpoint_dir='./shadow-{}-trainSize-{}-fullMember'.format(args.folder_tag, str(  args.train_size ) )
if(not os.path.exists(org_model_checkpoint_dir)):
    try:
        os.mkdir(org_model_checkpoint_dir)
    except:
        print('folder existed!')
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
private_data_len= int(args.train_size * 0.9) 
ref_data_len = 0
te_len= 1000
val_len=int(args.train_size * 0.2)
attack_tr_len=private_data_len
attack_te_len=0 # attack test set is a subset from the attack train set and the private set
tr_frac=0.45 # we use 45% of private data as the members to train MIA model
val_frac=0.05 # we use 5% of private data as the members to validate MIA model
te_frac=0.5 # we use 50% of private data as the members to test MIA model




print("loading data", flush=True)
data_set_features= np.load('./location-features.npy') 
data_set_label= np.load('./location-labels.npy') 

X =data_set_features.astype(np.float64)
Y = data_set_label.astype(np.int32)
num_classes = 30
print('total data len: ',len(X), flush=True)
print(X.shape, Y.shape)


if not os.path.isfile('./location_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open('./location_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./location_shuffle.pkl','rb'))








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
 


print('tr len %d | mia_members tr %d val %d te %d | mia_nonmembers tr %d val %d te %d | ref len %d | val len %d | test len %d | attack te len %d | remaining data len %d'%
      (len(private_data_tensor),len(mia_train_members_data_tensor),len(mia_val_members_data_tensor),len(mia_test_members_data_tensor),
       len(mia_train_nonmembers_data_tensor), len(mia_val_nonmembers_data_tensor),len(mia_test_nonmembers_data_tensor),
       len(ref_data_tensor),len(val_data_tensor),len(te_data_tensor),len(attack_te_data_tensor), len(remaining_data)), flush=True)



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
        
        
        
        return self.classifier(hidden_out),hidden_out, hidden_out


def train(data_loader,model,criterion,optimizer,epoch,use_cuda,num_batchs=999999,batch_size=32):
    # switch to train mode
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
     
    for (inputs, targets) in data_loader: 


        inputs, targets = inputs.to(device), targets.to(device)


        # compute output
        outputs, _, _  = model(inputs)
        
        loss = criterion(outputs, targets)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 

    return (losses.avg, top1.avg)


criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
model=LocationClassifier()
model=model.cuda()
best_val_acc=0
best_test_acc=0

from torch.utils.data import TensorDataset, DataLoader
def construct_new_dataloader(img_npy, y_train, batch_size=64):

    modified_train_data = []
    for i in range(len(y_train)):
       modified_train_data.append([img_npy[i], y_train[i]])


    new_train_loader = DataLoader(dataset=modified_train_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=4
                                   )
    return new_train_loader



from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
private_data_loader = construct_new_dataloader(private_data_tensor.numpy(), private_label_tensor.numpy(), batch_size=args.dp_batchsize)
model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
privacy_engine = PrivacyEngine()
model, optimizer, private_data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=private_data_loader,
    noise_multiplier=args.dp_noise_multiplier,
    max_grad_norm=args.dp_norm_clip,
) 



for epoch in range(args.epochs):
    if True:
        lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


    train_loss, train_acc = train(private_data_loader, model, criterion, optimizer, epoch, use_cuda, batch_size= args.dp_microbatches)

    tr_loss,tr_acc = test(private_data_tensor, private_label_tensor, model, criterion, use_cuda, device=device)        
    val_loss, val_acc = test(val_data_tensor, val_label_tensor, model, criterion, use_cuda, device=device)


    is_best = val_acc > best_val_acc

    best_val_acc=max(val_acc, best_val_acc)

    if is_best:
        _, best_test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda)
        best_val_acc=max(val_acc, best_val_acc)
        _, best_test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda) 
        best_train_acc = tr_acc
        

    save_checkpoint_global(
        {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
        },
        is_best,
        checkpoint=org_model_checkpoint_dir,
            filename='protected_model-%d.pth.tar'%(save_tag),
            best_filename='protected_model_best-%d.pth.tar'%(save_tag),

    )
    print("LR = ", optimizer.param_groups[0]['lr'])
    print('  epoch %d | tr acc %.2f loss %.2f | val acc %.2f  | best train acc %.2f | best val acc %.2f | best te acc %.2f'
          %(epoch, tr_acc, train_loss, val_acc, best_train_acc, best_val_acc, best_test_acc), flush=True)




resume_best=os.path.join(org_model_checkpoint_dir, 'protected_model_best-%d.pth.tar'%(save_tag))
criterion=nn.CrossEntropyLoss()
best_model=LocationClassifier().cuda()
best_model = ModuleValidator.fix(best_model)
ModuleValidator.validate(best_model, strict=False)
optimizer = optim.SGD(best_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
best_model, optimizer, data_loader = privacy_engine.make_private(
    module=best_model,
    optimizer=optimizer,
    data_loader=private_data_loader,
    noise_multiplier=args.dp_noise_multiplier,
    max_grad_norm=args.dp_norm_clip,
) 
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
best_model = best_model.to(device)
_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda, device=device)
print('\t===> DPSGD model %s | train acc %.4f | val acc %.4f | test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)


 

