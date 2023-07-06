import argparse
import os
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=10000)
parser.add_argument('--train_org', type=int, default=0, help='flag for training undefened model')
parser.add_argument('--train_dmp', type=int, default=0, help='flag for training DMP model')
parser.add_argument('--train_lr', type=float, default=0.0005)
parser.add_argument('--distill_lr', type=float, default=0.5)
parser.add_argument('--synt_data_path', type=str, default=None, help='path to the synthetic data')
parser.add_argument('--num_synt_sample', type=int, default=100000)
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--save_tag', type=str, default='0', help='current shadow model index')
parser.add_argument('--total_models', type=int, default=1)
parser.add_argument('--folder_tag', type=str, default='undefended')
parser.add_argument('--res_folder', type=str, default='lira-undefended-fullMember')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import sys
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
from torch.distributions import Categorical
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
org_model_checkpoint_dir= './shadow-{}-trainSize-{}-fullMember'.format(args.folder_tag, str(  args.train_size ) )
if(not os.path.exists(org_model_checkpoint_dir)):
    try:
        os.mkdir(org_model_checkpoint_dir)
    except:
        print('existed')
is_train_org = args.train_org
is_train_ref = args.train_dmp
BATCH_SIZE = 32
private_data_len= int(args.train_size * 0.9) 
ref_data_len = private_data_len
te_len= int(args.train_size * 0.1)
val_len=int(args.train_size * 0.1)
attack_tr_len=private_data_len
attack_te_len=0  
num_epochs=20
distil_epochs=100
tr_frac=0.45 
val_frac=0.05 # first 50% of private data as known members
te_frac=0.5 # remaining 50% of private data as test members
print("loading data")
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




all_non_private_data = X[all_indices[:private_data_len*2]]
all_non_private_label = Y[all_indices[:private_data_len*2]]


print('shadow data and label len ', all_non_private_data.shape, all_non_private_label.shape)
save_tag = int(args.save_tag)  

print('\tCurrent training id ', save_tag)
np.random.seed( 0 )

lira_folder = args.res_folder
if(not os.path.exists(lira_folder)):
    try:
        os.mkdir(lira_folder)
    except:
        print('already existed')

len_all_non_private_data = len(all_non_private_data)

if(args.folder_tag=='dmp'):
    # DMP is trained with the same set of shadow samples used to train the corresponding teacher model
    keep = np.load( os.path.join(lira_folder.replace('dmp', 'undefended'), '%s_keep.npy'%save_tag) )
    np.save( os.path.join(lira_folder, '%s_keep.npy'%save_tag),  keep )
else:
    keep = np.random.uniform(0,1,size=(args.total_models, len_all_non_private_data ))
    order = keep.argsort(0)
    # each sample will be trained on only a certain fraction of the shadow models (e.g., 50%) 
    keep = order < int( (private_data_len/float(len_all_non_private_data)) * args.total_models) 
    keep = np.array(keep[save_tag], dtype=bool)
    np.save( os.path.join(lira_folder, '%s_keep.npy'%save_tag),  keep )



private_data = all_non_private_data[keep]
private_label = all_non_private_label[keep]
print('first 20 labels ', private_label[:20])
print('total len of shadow training ', len(private_data), len_all_non_private_data)






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
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs, _, _ = model(inputs)
        loss = alpha*F.kl_div(F.log_softmax(outputs/t_softmax, dim=1), F.softmax(targets/t_softmax, dim=1))# + (1-alpha)*true_criterion(outputs,true_targets)
        losses.update(loss.item(), inputs.size(0))
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
        if ind > num_batchs:
            break
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs,_,_ = model(inputs)
        
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)


user_lr=args.train_lr 
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
model=TexasClassifier()
model=model.to(device)
optimizer=optim.Adam(model.parameters(), lr=user_lr)
best_val_acc=0
best_test_acc=0
if(is_train_org):
    for epoch in range(num_epochs): 

        train_loss, train_acc = train(private_data_tensor, private_label_tensor, model, criterion, optimizer, epoch, use_cuda, batch_size= BATCH_SIZE, uniform_reg=False)
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
            filename='unprotected_model-%d.pth.tar'%(save_tag),
            best_filename='unprotected_model_best-%d.pth.tar'%(save_tag),
        )

        print('  epoch %d | tr acc %.2f loss %.2f | val acc %.2f loss %.2f | test acc %.2f | best val acc %.2f | best te acc %.2f'
              %(epoch, train_acc, train_loss, val_acc, val_loss, test_acc, best_val_acc, best_test_acc), flush=True)


 
criterion=nn.CrossEntropyLoss()
best_model=TexasClassifier().to(device)
resume_best= os.path.join(org_model_checkpoint_dir.replace('dmp', 'undefended'), 'unprotected_model_best-%d.pth.tar'%(save_tag))

assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda)
print('\t===> %s Undefended model: train acc %.4f val acc %.4f test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)




if(not args.train_dmp):
    sys.exit() 

import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s
elapsed_time = 0 

start_time = time.time()

from ctgan import CTGANSynthesizer
X  = private_data
discrete_columns = [ ]

num_synt_sample = args.num_synt_sample
ctgan = CTGANSynthesizer(epochs=50 , batch_size=256, pac=128, verbose=True)
ctgan.fit(X, discrete_columns)
# Synthetic copy
samples = ctgan.sample(num_synt_sample)
threshold = 0.5 
binarized_samples = np.where(samples>= threshold, 1, 0)
   
ref_data_tensor = torch.from_numpy(binarized_samples).type(torch.FloatTensor)
print("\t synthetic reference data size : ", ref_data_tensor.size())
elapsed_time += time.time() - start_time
print('| Elapsed time for training GAN: %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))




batch_size=512
all_outputs=[]
outputs_index = []

len_t = len(ref_data_tensor)//batch_size

for ind in range(len_t):
    inputs = ref_data_tensor[ind*batch_size:(ind+1)*batch_size]
    if use_cuda:
        inputs = inputs.to(device)
    inputs = torch.autograd.Variable(inputs)
    outputs,_,_ = best_model(inputs)
    _, index = torch.max(outputs.data, 1)   
    outputs_index.append( index.data.cpu().numpy() )

    all_outputs.append(outputs.data.cpu().numpy())

if len(ref_data_tensor)%batch_size:
    inputs=ref_data_tensor[-(len(ref_data_tensor)%batch_size):]
    if use_cuda:
        inputs = inputs.to(device)
    inputs = torch.autograd.Variable(inputs)
    outputs,_,_ = best_model(inputs)

    _, index = torch.max(outputs.data, 1)   
    outputs_index.append( index.data.cpu().numpy() )

    all_outputs.append(outputs.data.cpu().numpy())

final_outputs=np.concatenate(all_outputs)
final_indexes = np.concatenate(outputs_index) 
# this is the labels assigned by the org model
reference_label_tensor = (torch.from_numpy(final_indexes).type(torch.LongTensor)) 
distil_label_tensor=(torch.from_numpy(final_outputs).type(torch.FloatTensor))

if(args.synt_data_path!=None):
    ref_label_tensor = reference_label_tensor 

# train final protected model via knowledge distillation
distil_model=TexasClassifier().to(device)
distil_test_criterion=nn.CrossEntropyLoss()
distil_schedule=[60, 90, 150]
distil_lr=args.distill_lr 
distil_best_acc=0
best_distil_test_acc=0
gamma=.1
t_softmax=1
filename='protected_model-%d.pth.tar'%(save_tag)
best_filename='protected_model_best-%d.pth.tar'%(save_tag)


if(is_train_ref):
    for epoch in range(distil_epochs):
        if epoch in distil_schedule:
            distil_lr *= gamma
            print('----> Epoch %d distillation lr %f'%(epoch,distil_lr))

        distil_optimizer=optim.SGD(distil_model.parameters(), lr=distil_lr, momentum=0.99, weight_decay=1e-5)
        distil_tr_loss = train_pub(ref_data_tensor, distil_label_tensor, ref_label_tensor, distil_model, t_softmax,
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
            filename=filename,
            best_filename=best_filename,
        )

        print('epoch %d | distil loss %.4f | tr loss %.4f tr acc %.4f | val loss %.4f val acc %.4f | best val acc %.4f | best test acc %.4f'%(epoch,distil_tr_loss,tr_loss,tr_acc,val_loss,val_acc,distil_best_acc,best_distil_test_acc))

print("\tBest model is saved at %s"%best_filename)


elapsed_time += time.time() - start_time
print('| Elapsed time for training DMP: %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))


criterion=nn.CrossEntropyLoss()
best_model=TexasClassifier().to(device)
resume_best=os.path.join(org_model_checkpoint_dir, best_filename )
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda)
print('\t ===> %s DMP model: train acc %.4f val acc %.4f test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)
