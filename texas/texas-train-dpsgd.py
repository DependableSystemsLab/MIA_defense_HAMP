import sys 
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import argparse
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
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
args = parser.parse_args()
 
org_model_checkpoint_dir='./dpsgd-trainSize-{}'.format( str(  args.train_size ) )
if(not os.path.exists(org_model_checkpoint_dir)):
    os.mkdir(org_model_checkpoint_dir)
private_data_len= int(args.train_size * 0.9) 
ref_data_len = private_data_len
te_len= int(args.train_size * 0.1)
val_len=int(args.train_size * 0.1)
attack_tr_len=private_data_len
attack_te_len=0 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr_frac=0.45  
val_frac=0.05  
te_frac=0.5  
print("loading data", flush=True)
data_set_features= np.load('./texas100-features.npy') 
data_set_label= np.load('./texas100-labels.npy') 
X =data_set_features.astype(np.float64)
Y = data_set_label.astype(np.int32)-1
print('total data len: ',len(X), flush=True)
print(X.shape, Y.shape)
if not os.path.isfile('./texas_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open('./texas_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./texas_shuffle.pkl','rb'))







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
 


print('tr len %d | mia_members tr %d val %d te %d | mia_nonmembers tr %d val %d te %d | ref len %d | val len %d | test len %d | attack te len %d |'%
      (len(private_data_tensor),len(mia_train_members_data_tensor),len(mia_val_members_data_tensor),len(mia_test_members_data_tensor),
       len(mia_train_nonmembers_data_tensor), len(mia_val_nonmembers_data_tensor),len(mia_test_nonmembers_data_tensor),
       len(ref_data_tensor),len(val_data_tensor),len(te_data_tensor),len(attack_te_data_tensor)), flush=True)






class TexasClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(TexasClassifier, self).__init__()

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
        return self.classifier(hidden_out),hidden_out, hidden_out




def train(data_loader,model,criterion,optimizer,epoch,use_cuda,num_batchs=999999,batch_size=32):
    # switch to train mode
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
     
    for (inputs, targets) in data_loader: 
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _, _  = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (losses.avg, top1.avg)




 
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
private_data_loader = construct_new_dataloader(private_data_tensor.numpy(), private_label_tensor.numpy(), batch_size=args.dp_batchsize)


criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
model=TexasClassifier() 
best_val_acc=0
best_test_acc=0
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

model = model.to(device)
best_train_acc = 0.
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
        filename='lr%s_batchsize%d_microbatch%d_noiseMul%s_epoch%d_clipThres%s_protected_model.pth.tar'%(str(args.lr).replace('.','-'),
                    args.dp_batchsize, args.dp_microbatches, str(args.dp_noise_multiplier).replace('.','-'), args.epochs, str(args.dp_norm_clip).replace('.','-')),
        best_filename='best_lr%s_batchsize%d_microbatch%d_noiseMul%s_epoch%d_clipThres%s_protected_model.pth.tar'%(str(args.lr).replace('.','-'),
                    args.dp_batchsize, args.dp_microbatches, str(args.dp_noise_multiplier).replace('.','-'), args.epochs, str(args.dp_norm_clip).replace('.','-')),
    )
    #print("LR = ", optimizer.param_groups[0]['lr'])
    print('  epoch %d | tr acc %.2f loss %.2f | val acc %.2f  | best train acc %.2f | best val acc %.2f | best te acc %.2f'
          %(epoch, tr_acc, train_loss, val_acc, best_train_acc, best_val_acc, best_test_acc), flush=True)




resume_best = os.path.join(org_model_checkpoint_dir, 'best_lr%s_batchsize%d_microbatch%d_noiseMul%s_epoch%d_clipThres%s_protected_model.pth.tar'%(str(args.lr).replace('.','-'),
                    args.dp_batchsize, args.dp_microbatches, str(args.dp_noise_multiplier).replace('.','-'), args.epochs, str(args.dp_norm_clip).replace('.','-')))



 
criterion=nn.CrossEntropyLoss()
best_model=TexasClassifier().cuda()
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

