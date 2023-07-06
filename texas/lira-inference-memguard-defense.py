import sys
sys.path.insert(0,'./util/')
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=10000) 
parser.add_argument('--memguard_path', type=str, required=True, help='path to the shadow attack classifier')
parser.add_argument('--save_tag', type=str, default='0')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--res_folder', type=str, required=True)
parser.add_argument('--org_path', type=str, required=True, help='path to the evaluated model')
args = parser.parse_args()
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import *
from torch.distributions import Categorical
import argparse
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

 


use_cuda = torch.cuda.is_available()
criterion=nn.CrossEntropyLoss()
best_model=TexasClassifier().cuda()
resume_best= args.org_path

assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model %s'%resume_best
checkpoint = os.path.dirname(resume_best)
checkpoint = torch.load(resume_best)
best_model.load_state_dict(checkpoint['state_dict'])



def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator

def _model_predictions(model, x, y, batch_size=256):
    return_outputs, return_labels = [], []

    return_labels = y.numpy() 

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

 
    return_outputs.append( softmax_by_row(outputs ) )


    return_outputs = np.concatenate(return_outputs)

    return (return_outputs, return_labels, outputs)



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
 



shadow_performance = _model_predictions(best_model, all_shadow_data_tensor, all_shadow_label_tensor, batch_size=512)  


import numpy as np
np.random.seed(1000)
import imp
import keras
from keras.models import Model
import tensorflow.compat.v1 as tf
import os
import configparser
import argparse
from scipy.special import softmax 
tf.disable_eager_execution()

user_label_dim=100
num_classes=1

sess = tf.InteractiveSession( )
sess.run(tf.global_variables_initializer())




f_evaluate = shadow_performance[0]
f_evaluate_logits = shadow_performance[2]
l_evaluate = keep
print('dataset shape information: ', f_evaluate.shape, f_evaluate_logits.shape, l_evaluate.shape)



f_evaluate_origin=np.copy(f_evaluate)  #keep a copy of original one
f_evaluate_logits_origin=np.copy(f_evaluate_logits)
#############as we sort the prediction sscores, back_index is used to get back original scores#############
sort_index=np.argsort(f_evaluate,axis=1)
back_index=np.copy(sort_index)
for i in np.arange(back_index.shape[0]):
    back_index[i,sort_index[i,:]]=np.arange(back_index.shape[1])
f_evaluate=np.sort(f_evaluate,axis=1)
f_evaluate_logits=np.sort(f_evaluate_logits,axis=1)



print("f evaluate shape: {}".format(f_evaluate.shape))
print("f evaluate logits shape: {}".format(f_evaluate_logits.shape))



##########loading defense model -------------------------------------------------------------
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, concatenate
def model_defense_optimize(input_shape,labels_dim):
    inputs_b=Input(shape=input_shape)
    x_b=Activation('softmax')(inputs_b)
    x_b=Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(128,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    x_b=Dense(64,kernel_initializer=keras.initializers.glorot_uniform(seed=100),activation='relu')(x_b)
    outputs_pre=Dense(labels_dim,kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x_b)
    outputs=Activation('sigmoid')(outputs_pre)
    model = Model(inputs=inputs_b, outputs=outputs)
    return model

from keras.models import load_model
defense_model = load_model( args.memguard_path )
weights=defense_model.get_weights()
del defense_model

input_shape=f_evaluate.shape[1:]
print("Loading defense model...")

model=model_defense_optimize(input_shape=input_shape,labels_dim=num_classes)
model.compile(loss=keras.losses.binary_crossentropy,optimizer=tf.keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
model.set_weights(weights)
model.trainable=False



########evaluate the performance of defense's attack model on undefended data########
scores_evaluate = model.evaluate(f_evaluate_logits, l_evaluate, verbose=0)
print('\nevaluate loss on model:', scores_evaluate[0])
print('==>\tevaluate the NN attack accuracy on model (undefended data):', scores_evaluate[1])   # means MemGuard's attack model's attack accuracy on the undefended data

output=model.layers[-2].output[:,0]
c1=1.0  #used to find adversarial examples 
c2=10.0    #penalty such that the index of max score is keeped
c3=0.1
#alpha_value=0.0 

origin_value_placeholder=tf.placeholder(tf.float32,shape=(1,user_label_dim)) #placeholder with original confidence score values (not logit)
label_mask=tf.placeholder(tf.float32,shape=(1,user_label_dim))  # one-hot encode that encodes the predicted label 
c1_placeholder=tf.placeholder(tf.float32)
c2_placeholder=tf.placeholder(tf.float32)
c3_placeholder=tf.placeholder(tf.float32)

correct_label = tf.reduce_sum(label_mask * model.input, axis=1)
wrong_label = tf.reduce_max((1-label_mask) * model.input - 1e8*label_mask, axis=1)


loss1=tf.abs(output)
### output of defense classifier is the logit, when it is close to 0, the prediction by the inference is close to 0.5, i.e., random guess.
### loss1 ensures random guessing for inference classifier ###
loss2=tf.nn.relu(wrong_label-correct_label)
### loss2 ensures no changes to target classifier predictions ###
loss3=tf.reduce_sum(tf.abs(tf.nn.softmax(model.input)-origin_value_placeholder)) #L-1 norm
### loss3 ensures minimal noise addition

loss=c1_placeholder*loss1+c2_placeholder*loss2+c3_placeholder*loss3
gradient_targetlabel=K.gradients(loss,model.input)
label_mask_array=np.zeros([1,user_label_dim],dtype=np.float)
##########################################################
result_array=np.zeros(f_evaluate.shape,dtype=np.float)
result_array_logits=np.zeros(f_evaluate.shape,dtype=np.float)
success_fraction=0.0
max_iteration=300   #max iteration if can't find adversarial example that satisfies requirements
np.random.seed(1000)
for test_sample_id in np.arange(0,f_evaluate.shape[0]):
    if test_sample_id%100==0:
        print("test sample id: {}".format(test_sample_id), flush=True)
    max_label=np.argmax(f_evaluate[test_sample_id,:])
    origin_value=np.copy(f_evaluate[test_sample_id,:]).reshape(1,user_label_dim)
    origin_value_logits=np.copy(f_evaluate_logits[test_sample_id,:]).reshape(1,user_label_dim)
    label_mask_array[0,:]=0.0
    label_mask_array[0,max_label]=1.0
    sample_f=np.copy(origin_value_logits)
    result_predict_scores_initial=model.predict(sample_f)
    ########## if the output score is already very close to 0.5, we can just use it for numerical reason
    if np.abs(result_predict_scores_initial-0.5)<=1e-5:
        success_fraction+=1.0
        result_array[test_sample_id,:]=origin_value[0,back_index[test_sample_id,:]]
        result_array_logits[test_sample_id,:]=origin_value_logits[0,back_index[test_sample_id,:]]
        continue
    last_iteration_result=np.copy(origin_value)[0,back_index[test_sample_id,:]]
    last_iteration_result_logits=np.copy(origin_value_logits)[0,back_index[test_sample_id,:]]
    success=True
    c3=0.1
    iterate_time=1
    while success==True: 
        sample_f=np.copy(origin_value_logits)
        j=1
        result_max_label=-1
        result_predict_scores=result_predict_scores_initial
        while j<max_iteration and (max_label!=result_max_label or (result_predict_scores-0.5)*(result_predict_scores_initial-0.5)>0):
            gradient_values=sess.run(gradient_targetlabel,feed_dict={model.input:sample_f,origin_value_placeholder:origin_value,label_mask:label_mask_array,c3_placeholder:c3,c1_placeholder:c1,c2_placeholder:c2})[0][0]
            gradient_values=gradient_values/np.linalg.norm(gradient_values)
            sample_f=sample_f-0.1*gradient_values
            result_predict_scores=model.predict(sample_f)
            result_max_label=np.argmax(sample_f)
            j+=1        
        if max_label!=result_max_label:
            if iterate_time==1:
                print("failed sample for label not same for id: {},c3:{} not add noise".format(test_sample_id,c3))
                success_fraction-=1.0
            break                
        if ((model.predict(sample_f)-0.5)*(result_predict_scores_initial-0.5))>0:
            if iterate_time==1:
                print("max iteration reached with id: {}, max score: {}, prediction_score: {}, c3: {}, not add noise".format(test_sample_id,np.amax(softmax(sample_f)),result_predict_scores,c3))
            break
        last_iteration_result[:]=softmax(sample_f)[0,back_index[test_sample_id,:]]
        last_iteration_result_logits[:]=sample_f[0,back_index[test_sample_id,:]]
        iterate_time+=1 
        c3=c3*10
        if c3>100000:
            break
    success_fraction+=1.0
    result_array[test_sample_id,:]=last_iteration_result[:]
    result_array_logits[test_sample_id,:]=last_iteration_result_logits[:]
print("Success fraction: {}".format(success_fraction/float(f_evaluate.shape[0])))


scores_evaluate = model.evaluate(result_array_logits, l_evaluate, verbose=0)
print('evaluate loss on model:', scores_evaluate[0])
print('\n====> evaluate accuracy on model:', scores_evaluate[1])




return_outputs = result_array_logits
return_outputs = return_outputs[:, :, np.newaxis]
return_outputs = return_outputs.transpose((0, 2, 1))
np.save( os.path.join(lira_folder, '%s_logit.npy'%args.save_tag), return_outputs ) 
np.save( os.path.join(lira_folder, '%s_keep.npy'%args.save_tag), keep ) 
#np.save( os.path.join(lira_folder, 'shadow_data.npy'),  all_shadow_data )
np.save( os.path.join(lira_folder, 'shadow_label.npy'),  all_shadow_label )







