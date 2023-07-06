import sys
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=10000)
parser.add_argument('--train_org', type=int, default=0, help='flag for training teacher model')
parser.add_argument('--train_selena', type=int, default=0, help='flag for performing knowledge distillation')
parser.add_argument('--train_lr', type=float, default=0.0005, help='lr for the teacher model')
parser.add_argument('--distill_lr', type=float, default=0.5, help='lr for distilation')
parser.add_argument('--K', type=int, default=25, help='num of teacher models')
parser.add_argument('--L', type=int, default=10, help='num of models that are not trained on each samples') 
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
org_model_checkpoint_dir='./selena-trainSize-{}'.format( str(  args.train_size )  )
if(not os.path.exists(org_model_checkpoint_dir)):
    os.mkdir(org_model_checkpoint_dir)
is_train_org = args.train_org
is_train_ref = args.train_selena
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
val_frac=0.05  
te_frac=0.5 
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

'''
For each training sample, we randomly select L models from the K teacher models and store them in ``non_model_indices_for_each_sample''
    These L models are the models that are not trained on the specific training samples
For each teacher model, we also generate a list ``sub_model_indices'' to its reference set (on which the teacher model should be trained)
'''
np.random.seed(0)
K = args.K
L = args.L
sub_model_indices_file_prefix = './texas_selena_submodel_indices_%s'%str(args.train_size)
non_model_indices_file = './non_model_indices_for_each_sample_%s.pkl'%str(args.train_size)
sub_model_indices = [[] for _ in range(K)]
non_model_indices_for_each_sample = np.zeros((private_data_len, L))
if not os.path.isfile( sub_model_indices_file_prefix + "_0.pkl" ):
    # partition this into K lists
    training_indices = all_indices[:private_data_len]  
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
    for i in range(K): 
        pickle.dump(sub_model_indices[i] , open(sub_model_indices_file_prefix + "_%s.pkl"%str(i), 'wb') )
    pickle.dump(non_model_indices_for_each_sample, open(non_model_indices_file, 'wb'))

else:
    for i in range(K):
        # indices to the ``reference set'' for the teacher model
        sub_model_indices[i] = pickle.load( open(sub_model_indices_file_prefix + "_%s.pkl"%str(i), 'rb') )  
    # indices to the ``teacher model'' for each sample, on which the indexed teacher model is ``not'' trained
    non_model_indices_for_each_sample = pickle.load( open(non_model_indices_file, 'rb') ) 









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


def train_pub(train_data, labels, true_labels, model, t_softmax, optimizer, num_batchs=999999, batch_size=16):
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
 
        outputs, _, _ = model(inputs)

        loss = F.kl_div(F.log_softmax(outputs/t_softmax, dim=1), F.softmax(targets/t_softmax, dim=1)) 
        losses.update(loss.item(), inputs.size(0))
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg)


def train(train_data,labels,model,criterion,optimizer,epoch,use_cuda,num_batchs=999999,batch_size=32):
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
n_classes=100
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
if(is_train_org):
    for sub in range(K):
        model=TexasClassifier()
        model=model.to(device)
        optimizer=optim.Adam(model.parameters(), lr=user_lr, weight_decay=1e-3)
        best_val_acc=0
        best_test_acc=0        

        # derive the ``reference set'' for training the teacher model
        sub_train_x = X[sub_model_indices[sub]]
        sub_train_y = Y[sub_model_indices[sub]]
        x_tensor = torch.from_numpy(sub_train_x).type(torch.FloatTensor)
        y_tensor = torch.from_numpy(sub_train_y).type(torch.LongTensor) 

        for epoch in range(num_epochs): 

            train_loss, train_acc = train(x_tensor, y_tensor, model, criterion, optimizer, epoch, use_cuda, batch_size= BATCH_SIZE)

            val_loss, val_acc = test(val_data_tensor, val_label_tensor, model, criterion, use_cuda)

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
                filename='unprotected_model_sub_%s.pth.tar'%str(sub),
                best_filename='unprotected_model_best_sub_%s.pth.tar'%str(sub),
            )

            print('%d sub model |  epoch %d | tr acc %.2f loss %.2f | val acc %.2f loss %.2f | test acc %.2f | best val acc %.2f | best te acc %.2f'
                  %(sub, epoch, train_acc, train_loss, val_acc, val_loss, test_acc, best_val_acc, best_test_acc), flush=True)

sub_models = [[] for _ in range(K)]

for i in range(K):
    best_model=TexasClassifier().to(device)
    resume_best=os.path.join( org_model_checkpoint_dir,'unprotected_model_best_sub_%s.pth.tar'%str(i))
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




training_indices = all_indices[:private_data_len]  
all_outputs=[]
non_model_indices_for_each_sample = non_model_indices_for_each_sample.astype(int)
for cnt, each_ind in enumerate(training_indices):  
    inputs = private_data_tensor[cnt:cnt+1]
    inputs = inputs.to(device)
    first = True
    # adaptive inference on the sub model, each model only predicts on the samples that were NOT used for training
    for i, model_index in enumerate(non_model_indices_for_each_sample[cnt]): 
        sub_model = sub_models[model_index]
        outputs,_,_ = sub_model(inputs)
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
distil_model=TexasClassifier().to(device)
distil_test_criterion=nn.CrossEntropyLoss()
distil_schedule=[60, 90, 150]
distil_lr=args.distill_lr 
distil_best_acc=0
best_distil_test_acc=0
gamma=.1
t_softmax=1
if(is_train_ref):
    for epoch in range(distil_epochs):
        if epoch in distil_schedule:
            distil_lr *= gamma
            print('----> Epoch %d distillation lr %f'%(epoch,distil_lr))

        distil_optimizer=optim.SGD(distil_model.parameters(), lr=distil_lr, momentum=0.99, weight_decay=1e-5)

        distil_tr_loss = train_pub(private_data_tensor, distil_label_tensor, ref_label_tensor, distil_model, t_softmax,
                                   distil_optimizer, batch_size=BATCH_SIZE)

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
            filename='protected_model-lr-%s.pth.tar'%(str(args.distill_lr).replace('.','-')),
            best_filename='protected_model_best-lr-%s.pth.tar'%(str(args.distill_lr).replace('.','-')),
        )

        print('epoch %d | distil loss %.4f | tr loss %.4f tr acc %.4f | val loss %.4f val acc %.4f | best val acc %.4f | best test acc %.4f'%(epoch,distil_tr_loss,tr_loss,tr_acc,val_loss,val_acc,distil_best_acc,best_distil_test_acc))



criterion=nn.CrossEntropyLoss()
best_model=TexasClassifier().to(device)
resume_best=os.path.join(org_model_checkpoint_dir, 'protected_model_best-lr-%s.pth.tar'%(str(args.distill_lr).replace('.','-')))
assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])
_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda)
print('\t===> %s SELENA model: train acc %.4f val acc %.4f test acc %.4f'%(resume_best, best_train,best_val, best_test), flush=True)

