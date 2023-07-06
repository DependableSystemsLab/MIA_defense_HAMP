import argparse
import os
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=10000)
parser.add_argument('--org_model_path', type=str, default='None', help='path to the teacher model')
parser.add_argument('--gan_path', type=str, default='None', help='path to the GAN')
parser.add_argument('--train_dmp', type=int, default=0)
parser.add_argument('--num_synthetic', type=int, default=10000, help='num of synthetic samples to generate')
parser.add_argument('--distill_lr', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import sys
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os
from densenet import densenet
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
org_model_checkpoint_dir='./dmp-trainSize-{}-lr-{}'.format( str(  args.train_size ), str(args.lr).replace('.', '') )


if(not os.path.exists(org_model_checkpoint_dir)):
    os.mkdir(org_model_checkpoint_dir)
is_train_ref = args.train_dmp
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

dataloader = datasets.CIFAR100
num_classes = 100

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
        loss = alpha*F.kl_div(F.log_softmax(outputs/t_softmax, dim=1), F.softmax(targets/t_softmax, dim=1)) 
        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg)




use_cuda = torch.cuda.is_available()
criterion=nn.CrossEntropyLoss()
best_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).to(device)
resume_best= args.org_model_path
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])

_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda, device=device)
print(' %s Undefended model: train acc %.4f val acc %.4f test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)



nc=3 
# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
#number of discriminator filters
ndf = 64
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output


import time
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

start_time = time.time()
elapsed_time = 0


netG = Generator(ngpu).to(device)
#netG.apply(weights_init)
#load weights to test the model
netG.load_state_dict(torch.load(args.gan_path))


size_transform=transforms.Compose([
   transforms.Resize(32), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
inv_normalize = transforms.Compose([ 
    transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5,), (1/0.5, 1/0.5, 1/0.5))
])

len_t = args.num_synthetic//BATCH_SIZE
if args.num_synthetic%BATCH_SIZE:
    len_t += 1

with torch.no_grad():   
    first = True
    for ind in range(len_t):
        if(ind == len_t-1):
            noise = torch.randn(args.num_synthetic%BATCH_SIZE, nz, 1, 1, device=device)
        else:
            noise = torch.randn(BATCH_SIZE, nz, 1, 1, device=device)
        fake = netG(noise)
        if(first):
            # first de-normalize, then resize
            #print("org gan output", fake.cpu().numpy().max(), fake.cpu().numpy().min(), flush=True)
            fake = inv_normalize(fake)
            #print("inv norm gan output", fake.cpu().numpy().max(), fake.cpu().numpy().min(), flush=True)
            fake = size_transform(fake)
            #print("32*32 norm output", fake.cpu().numpy().max(), fake.cpu().numpy().min(), flush=True)
            synthetic_data = fake
            first=False
            #print(fake.size())
        else:
            fake = inv_normalize(fake)
            fake = size_transform(fake)
            synthetic_data = torch.cat( (synthetic_data, fake), 0 )

print( synthetic_data.cpu().numpy().max(), synthetic_data.cpu().numpy().min(), )
print("total num of synthetic data ", synthetic_data.size(), flush=True)
ref_data_tensor = synthetic_data

batch_size=64
all_outputs=[]
outputs_index = []

len_t = len(ref_data_tensor)//batch_size

for ind in range(len_t):
    inputs = ref_data_tensor[ind*batch_size:(ind+1)*batch_size]
    if use_cuda:
        inputs = inputs.to(device)
    inputs = torch.autograd.Variable(inputs)
    outputs  = best_model(inputs)
    _, index = torch.max(outputs.data, 1)   
    outputs_index.append( index.data.cpu().numpy() )
    all_outputs.append(outputs.data.cpu().numpy())

if len(ref_data_tensor)%batch_size:
    inputs=ref_data_tensor[-(len(ref_data_tensor)%batch_size):]
    if use_cuda:
        inputs = inputs.to(device)
    inputs = torch.autograd.Variable(inputs)
    outputs = best_model(inputs)
    _, index = torch.max(outputs.data, 1)   
    outputs_index.append( index.data.cpu().numpy() )
    all_outputs.append(outputs.data.cpu().numpy())

final_outputs=np.concatenate(all_outputs)
final_indexes = np.concatenate(outputs_index)
# this is the labels assigned by the org model
reference_label_tensor = (torch.from_numpy(final_indexes).type(torch.LongTensor)) 
distil_label_tensor=(torch.from_numpy(final_outputs).type(torch.FloatTensor))



# train final protected model via knowledge distillation
distil_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).to(device)
distil_test_criterion=nn.CrossEntropyLoss()
distil_schedule=[60, 90, 150]
distil_lr=args.distill_lr
distil_best_acc=0
best_distil_test_acc=0
gamma=.1
t_softmax=1


epoch_time = time.time() - start_time
elapsed_time += epoch_time
print('| Elapsed time for data preparation: %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))


if(is_train_ref):
    for epoch in range(distil_epochs):
        start_time = time.time()
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
            filename='dmp-lr-%s-synthetic-%d.pth.tar'%(str(args.distill_lr).replace('.','-'), args.num_synthetic ),
            best_filename='dmp_best-lr-%s-synthetic-%d.pth.tar'%(str(args.distill_lr).replace('.','-'), args.num_synthetic),
        )

        print('epoch %d | distil loss %.4f | tr loss %.4f tr acc %.4f | val loss %.4f val acc %.4f | best val acc %.4f | best test acc %.4f'%(epoch,distil_tr_loss,tr_loss,tr_acc,val_loss,val_acc,distil_best_acc,best_distil_test_acc),flush=True)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d hr, %02d min, %02d sec'  %(get_hms(elapsed_time)))


criterion=nn.CrossEntropyLoss()
best_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).to(device)
resume_best=os.path.join(org_model_checkpoint_dir,'dmp_best-lr-%s-synthetic-%d.pth.tar'%(str(args.distill_lr).replace('.','-'), args.num_synthetic) )
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])

_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda, device=device)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda, device=device)
print('\t===> DMP model %s | train acc %.4f | val acc %.4f | test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)
