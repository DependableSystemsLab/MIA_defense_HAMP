import sys
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
import sys
import os 
from torch.distributions import Categorical
import argparse
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=10000) 
parser.add_argument('--save_tag', type=str, default='0', help='current shadow model index')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--res_folder', type=str, required=True)
parser.add_argument('--isModifyOutput', type=int, default=0, help='indicator for using output modification')
parser.add_argument('--org_path', type=str, required=True, help='path to the evaluated model')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
org_model_checkpoint_dir='./shadow-org-trainSize-{}'.format( str(  args.train_size ) )
BATCH_SIZE = 32
private_data_len= int(args.train_size * 0.9) 
ref_data_len = private_data_len
te_len= int(args.train_size * 0.1)
val_len=int(args.train_size * 0.1)
attack_tr_len=private_data_len
attack_te_len=0 
num_epochs=100
distil_epochs=200
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
        return self.classifier(hidden_out)#,hidden_out, hidden_out


criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
criterion=nn.CrossEntropyLoss()
best_model=TexasClassifier().cuda()
resume_best= args.org_path
 


try:
    assert os.path.isfile(resume_best), 'Error: no checkpoint directory %s found for best model'%resume_best
    checkpoint = os.path.dirname(resume_best)
    checkpoint = torch.load(resume_best, map_location='cuda')
    best_model.load_state_dict(checkpoint['state_dict']) 

except:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    privacy_engine = PrivacyEngine()

    from collections import OrderedDict

    state_dict =  torch.load(resume_best)['state_dict']

    # state_dict in Opacus model is appended with the prefix, _model, we need to remove this:
    # e.g., _module.conv1.0.weight  ==> conv1.0.weight
    # we do so because the Opacus model also modifes the model during backprop (for DPSGD), which is incompatiable with performing attacks
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        #print(k , k[8:])
        name = k[8:] # remove `module.`
        new_state_dict[name] = v

    # load params 
    best_model = ModuleValidator.fix(best_model)
    ModuleValidator.validate(best_model, strict=False)
    best_model.load_state_dict(new_state_dict)





_,best_test = test(te_data_tensor, te_label_tensor, best_model, criterion, use_cuda)
_,best_val = test(val_data_tensor, val_label_tensor, best_model, criterion, use_cuda)
_,best_train = test(private_data_tensor, private_label_tensor, best_model, criterion, use_cuda)
print('\t===> %s  train acc %.4f val acc %.4f test acc %.4f'%(resume_best, best_train, best_val, best_test), flush=True)
 

def random_flip(img_set):
    ret = np.empty(img_set.shape)
    for m, image in enumerate(img_set):
        random_vector = np.random.choice(2, image.shape, replace=True) 
        out = random_vector
        ret[m, :] = out 
    return ret
 


def alter_output(model, inputs, output, non_Mem_Generator=None, batch_size=256):
    # output modification by hamp
    non_member_train_data = non_Mem_Generator(inputs)
    non_member_length = non_member_train_data.shape[0]
    non_member_tensor = torch.from_numpy(non_member_train_data).type(torch.FloatTensor) 
    if use_cuda:
        non_member_tensor = non_member_tensor.cuda() 

    with torch.no_grad():
    
        len_t = len(non_member_tensor)//batch_size
        if len(non_member_tensor)%batch_size:
            len_t += 1
        first = True
        for ind in range(len_t):
            outputs = model(non_member_tensor[ind*batch_size:(ind+1)*batch_size]) 
            if(first):
                outs = outputs
                first=False
            else:
                outs = torch.cat( (outs, outputs), 0)
        non_member_pred = outs
    
    non_member_pred = non_member_pred.cpu().detach().numpy()
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


lira_folder = args.res_folder

if(not os.path.exists(lira_folder)):
    os.mkdir(lira_folder)





all_non_private_data = X[all_indices[:private_data_len*2]]
all_non_private_label = Y[all_indices[:private_data_len*2]]
 

keep = np.r_[ np.ones(private_data_len), np.zeros(private_data_len) ]
keep = keep.astype(bool)

all_shadow_data = all_non_private_data 
all_shadow_label = all_non_private_label 
all_shadow_data_tensor=torch.from_numpy(all_shadow_data).type(torch.FloatTensor)
all_shadow_label_tensor=torch.from_numpy(all_shadow_label).type(torch.LongTensor)


return_outputs, return_labels = _model_predictions(best_model, all_shadow_data_tensor, all_shadow_label_tensor, batch_size=256) 

return_outputs = return_outputs[:, :, np.newaxis]
return_outputs = return_outputs.transpose((0, 2, 1))
np.save( os.path.join(lira_folder, '%s_logit.npy'%args.save_tag), return_outputs ) 
np.save( os.path.join(lira_folder, '%s_keep.npy'%args.save_tag), keep ) 
#np.save( os.path.join(lira_folder, 'shadow_data.npy'),  all_shadow_data )
np.save( os.path.join(lira_folder, 'shadow_label.npy'),  all_shadow_label )









