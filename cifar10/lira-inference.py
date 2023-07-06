import sys
sys.path.insert(0,'./util/')
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=10000) 
parser.add_argument('--save_tag', type=str, default='0', help='current shadow model index')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--res_folder', type=str, required=True)
parser.add_argument('--isModifyOutput', type=int, default=0, help='indicator for using output modification')
parser.add_argument('--org_path', type=str, required=True, help='path to the evaluated model')
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.distributions import Categorical
from torch.nn import functional as F
from util.densenet import densenet

float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
private_data_len= int(args.train_size * 0.9) 
ref_data_len = 0
te_len= int(args.train_size * 0.1)
val_len=int(args.train_size * 0.1)
attack_tr_len=private_data_len
attack_te_len=0 
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

remaining_data=X[all_indices[(private_data_len +  attack_te_len + attack_tr_len):]]
remaining_label=Y[all_indices[(private_data_len +  attack_te_len + attack_tr_len):]]


 

all_non_private_data = X[all_indices[:private_data_len*2]]
all_non_private_label = Y[all_indices[:private_data_len*2]]

 
 

lira_folder= args.res_folder
keep =  np.load( os.path.join(lira_folder, '%s_keep.npy'%args.save_tag) )
  


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




criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
best_model=densenet(num_classes=num_classes,depth=100,growthRate=12,compressionRate=2,dropRate=0).cuda()
resume_best= args.org_path

assert os.path.isfile(resume_best), 'Error: no checkpoint directory %s found for best model'%resume_best
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best, map_location='cuda')
best_model.load_state_dict(checkpoint['state_dict']) 

'''
except:
    assert os.path.isfile(resume_best), 'Error: no checkpoint directory %s found for best model'%resume_best
    checkpoint = os.path.dirname(resume_best)
    checkpoint = torch.load(resume_best, map_location='cuda')

    # this was used in training AdvReg model earlier
    best_model = torch.nn.DataParallel(best_model)
    best_model.load_state_dict(checkpoint['state_dict']) 
'''

 

def random_flip(img_set):
    # generate random samples
    ret = np.empty(img_set.shape)
    for m, image in enumerate(img_set):
        random_vector = np.random.randint(255, size=image.shape)/255.
        for i in range(3):
            random_vector[i, :, :] -= mean[i]
            random_vector[i, :, :] /= std[i]
        ret[m, :] = random_vector 
    return ret
 
def get_output(model, x, batch_size=256):
    len_t = len(x)//batch_size
    if len(x)%batch_size:
        len_t += 1
    first = True
    for ind in range(len_t):
        outputs  = model.forward(x[ind*batch_size:(ind+1)*batch_size].cuda())
        outputs = outputs.data.cpu().numpy()
        if(first):
            outs = outputs
            first=False
        else:
            outs = np.r_[outs, outputs]
    
    return torch.from_numpy(outs).type(torch.FloatTensor).cuda() 

def alter_output(model, inputs, output, non_Mem_Generator=None, batch_size=256):
    # output modification by hamp

    non_member_train_data = non_Mem_Generator(inputs)
    non_member_length = non_member_train_data.shape[0]
    non_member_tensor = torch.from_numpy(non_member_train_data).type(torch.FloatTensor) 
    if use_cuda:
        non_member_tensor = non_member_tensor.cuda() 


    non_member_pred = get_output(model, non_member_tensor)
    
    non_member_pred = non_member_pred.cpu().detach().numpy()

    
    non_member_length = len(inputs)

    output_copy = output.copy()
    non_member_pred_copy =  non_member_pred.copy()
    output_sorted = np.sort(output_copy, axis=1)
    non_member_pred_sorted = np.sort(non_member_pred_copy, axis=1)
    new_output = np.zeros( output.shape ) 
    for i in range( output.shape[0] ):
        for j in range(output.shape[1]):
            new_output[i][ np.where(output[i]==output_sorted[i][j]) ] = non_member_pred_sorted[ i%non_member_length ][j]
    return new_output


non_Mem_Generator = random_flip 



def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator

def _model_predictions(model, x, y, batch_size=256):
    model.eval()
    return_outputs, return_labels = [], []

    return_labels = y.numpy() 

    #outputs = model.forward(x.cuda())

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

    if(args.isModifyOutput):
        outputs = alter_output( model, x.numpy(), outputs , non_Mem_Generator=non_Mem_Generator) 

    return_outputs = outputs
    return (return_outputs, return_labels)




all_shadow_data = all_non_private_data 
all_shadow_label = all_non_private_label 
all_shadow_data_tensor=torch.from_numpy(all_shadow_data).type(torch.FloatTensor)
all_shadow_label_tensor=torch.from_numpy(all_shadow_label).type(torch.LongTensor)



return_outputs, return_labels = _model_predictions(best_model, all_shadow_data_tensor, all_shadow_label_tensor, batch_size=256) 

return_outputs = return_outputs[:, :, np.newaxis]
return_outputs = return_outputs.transpose((0, 2, 1))
np.save( os.path.join(lira_folder, '%s_logit.npy'%args.save_tag), return_outputs ) 
#np.save( os.path.join(lira_folder, 'shadow_data.npy'),  all_shadow_data )
np.save( os.path.join(lira_folder, 'shadow_label.npy'),  all_shadow_label )

print(args.save_tag)
