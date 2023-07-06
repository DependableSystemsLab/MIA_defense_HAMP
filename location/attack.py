import argparse
import os
parser = argparse.ArgumentParser() 
parser.add_argument('--train_size', type=int, default=10000)  
parser.add_argument('--path', type=str, required=True, help='path to the evaluated model')  
parser.add_argument('--getModelAcy', type=int, default=0, help='compute the model accuracy')   
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--isModifyOutput', type=int, default=0, help='flag for using output modification (for HAMP)')
parser.add_argument('--attack', type=str, default='entropy', help='attack method')
parser.add_argument('--fpr_threshold', default=0.001, type=float)


parser.add_argument('--isMemGuard', type=int, default=0, help='flag for evaluating MemGuard')
parser.add_argument('--prepMemGuard', type=int, default=0, help='flag for performing post-processing by MemGuard')
parser.add_argument('--memguard_path', type=str, default=None, help='path to the shadow attack model used by MemGuard')

parser.add_argument('--output_folder', type=str, default='attack_outputs', help='output folder for storing the attack outputs')
parser.add_argument('--save_tag', type=str, default=None, help='tag for saving the attack outputs (attack output and ground-truth membership labels), \
                                                                    used for computing the attack TPR later')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.distributions import Categorical
import datetime
import sys
sys.path.insert(0,'./util/')
from purchase_normal_train import *
from purchase_private_train import *
from purchase_attack_train import *
from purchase_util import * 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from score_based_MIA_util import black_box_benchmarks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resume_best= args.path
private_data_len= int(args.train_size * 0.9) 
ref_data_len = 0
te_len= 1000
val_len=int(args.train_size * 0.2)
attack_tr_len=private_data_len
attack_te_len=0 
data_path = './'
tr_frac=0.3 
val_frac=0.2 # adjusted as the train size is small
te_frac=0.5 
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

output_save_path=args.output_folder
if(not os.path.exists(output_save_path)):
    os.mkdir(output_save_path)


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
        return self.classifier(hidden_out)#,hidden_out, hidden_out






best_model=LocationClassifier()
best_model=best_model.cuda() 
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()



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


def random_flip(img_set):
    # generate random samples
    ret = np.empty(img_set.shape)
    for m, image in enumerate(img_set):
        random_vector = np.random.choice(2, image.shape, replace=True) 
        ret[m, :] = random_vector 
    return ret
random_sample = torch.from_numpy( random_flip(private_data_tensor.cpu().numpy()[:1])).type(torch.FloatTensor).cuda()

def alter_output(model, inputs, output, non_Mem_Generator=None, batch_size=256):

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

 
 


if(args.getModelAcy):
    import copy
    _, check_test_acc = test(te_data_tensor,te_label_tensor,copy.deepcopy(best_model),criterion,use_cuda)
    _, check_val_acc = test(val_data_tensor,val_label_tensor,copy.deepcopy(best_model),criterion,use_cuda)
    _, check_train_acc = test(private_data_tensor,private_label_tensor,copy.deepcopy(best_model),criterion,use_cuda)
    print('{} | train acc {:.4f} | val acc {:.4f} | test acc {:.4f}'.format(args.path, check_train_acc,check_val_acc,check_test_acc), flush=True)

non_Mem_Generator = random_flip 






from sklearn.metrics import roc_auc_score, roc_curve, auc 
def get_tpr(y_true, y_score, fpr_threshold=0.001, attack='None'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    accuracy = np.max(1-(fpr+(1-tpr))/2)
    auc_score = auc(fpr, tpr) 
    highest_tpr = tpr[np.where(fpr<fpr_threshold)[0][-1]]


    ## TPR and FPR are in ascending order
    ## TNR and FNR are in descending order
    fnr = 1 - tpr 
    tnr = 1 - fpr
    highest_tnr = tnr[np.where(fnr<fpr_threshold)[0][0]] 

    print( '\t\t===> %s: TPR %.2f%% @%.3f%%FPR | TNR %.2f%% @%.3f%%FNR | AUC %.4f'%( attack, highest_tpr*100, fpr_threshold*100, highest_tnr*100, fpr_threshold*100, auc_score  ) )





# for direct query attack
tr_members = np.r_[ mia_train_members_data_tensor.numpy(), mia_val_members_data_tensor.numpy()]
tr_members_y = np.r_[ mia_train_members_label_tensor.numpy(), mia_val_members_label_tensor.numpy()]
tr_non_members = np.r_[ mia_train_nonmembers_data_tensor.numpy(), mia_val_nonmembers_data_tensor.numpy()]
tr_non_members_y = np.r_[ mia_train_nonmembers_label_tensor.numpy(), mia_val_nonmembers_label_tensor.numpy()]
tr_m_true = np.r_[ np.ones(tr_members.shape[0]), np.zeros(tr_non_members.shape[0]) ]


if(args.attack == 'entropy'):
    def softmax_by_row(logits, T = 1.0):
        mx = np.max(logits, axis=-1, keepdims=True)
        exp = np.exp((logits - mx)/T)
        denominator = np.sum(exp, axis=-1, keepdims=True)
        return exp/denominator

    def _model_predictions(model, x, y, non_Mem_Generator, batch_size=256, isModifyOutput=0):
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

        if(isModifyOutput): 
            outputs = alter_output( model, x.cpu().numpy(), outputs  , non_Mem_Generator=non_Mem_Generator)
     
        return_outputs.append( softmax_by_row(outputs ) )


        return_outputs = np.concatenate(return_outputs)

        if(not args.prepMemGuard):
            return (return_outputs, return_labels)
        else:
            # return also the logit values for memguard
            return (return_outputs, return_labels, outputs)



    shadow_model = best_model


    shadow_train_performance = _model_predictions(shadow_model, torch.from_numpy(tr_members).type(torch.FloatTensor), torch.from_numpy(tr_members_y).type(torch.LongTensor),
                                                 non_Mem_Generator, isModifyOutput=args.isModifyOutput)
    shadow_test_performance = _model_predictions(shadow_model, torch.from_numpy(tr_non_members).type(torch.FloatTensor), torch.from_numpy(tr_non_members_y).type(torch.LongTensor),
                                                non_Mem_Generator, isModifyOutput=args.isModifyOutput)      

    target_train_performance = _model_predictions(best_model, mia_test_members_data_tensor, mia_test_members_label_tensor, non_Mem_Generator, isModifyOutput=args.isModifyOutput)
    target_test_performance = _model_predictions(best_model, mia_test_nonmembers_data_tensor, mia_test_nonmembers_label_tensor, non_Mem_Generator, isModifyOutput=args.isModifyOutput)



    if(args.prepMemGuard):


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
 

        user_label_dim=30
        num_classes=1

        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.3
        config_gpu.gpu_options.visible_device_list = "0"

        sess = tf.InteractiveSession(config=config_gpu)
        sess.run(tf.global_variables_initializer())





        train_outputs1 = shadow_train_performance[0] # score for first half of members for training attack (train_member_pred)
                                                                            # this is the known member set for adversary
        len1 = len(train_outputs1)

        # "data for testing the attack"
        train_outputs2 = target_train_performance[0]  # score for the second half of members for evaluating attack (test_member_pred)
                                                        # this is the actual members for evaluating the attack  
        len2 = len(train_outputs2)
        train_outputs = np.concatenate((train_outputs1, train_outputs2))


        test_outputs1 = shadow_test_performance[0] # data that were used for training the attack (these are not members) 
                                                        # this is the known non-member set for the adversary   

        test_outputs2 = target_test_performance[0] # test set 
                                                    # this is the actual non-members for evaluating the attack

        test_outputs = np.concatenate((test_outputs1, test_outputs2))


        train_logits1 = shadow_train_performance[2]
        train_logits2 = target_train_performance[2]
        train_logits = np.concatenate((train_logits1, train_logits2))

        test_logits1 = shadow_test_performance[2]
        test_logits2 = target_test_performance[2]
        test_logits = np.concatenate((test_logits1, test_logits2))
         





        min_len = min(len(train_outputs), len(test_outputs))
        print('selected number of members and non-members are: ', min(len(train_outputs), len(test_outputs)), min(len(train_logits), len(test_logits)))


        f_evaluate = np.concatenate((train_outputs[:min_len], test_outputs[:min_len]))
        f_evaluate_logits = np.concatenate((train_logits[:min_len], test_logits[:min_len]))
        l_evaluate = np.zeros(len(f_evaluate))
        l_evaluate[:min_len] = 1
        print('dataset shape information: ', f_evaluate.shape, f_evaluate_logits.shape, l_evaluate.shape, min_len)





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
                print("test sample id: {}".format(test_sample_id))
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


        file_path='./'
        if not os.path.exists(os.path.join(file_path, 'memguard_defense_results')):
            os.makedirs(os.path.join(file_path, 'memguard_defense_results'))

        np.savez(os.path.join(file_path, 'memguard_defense_results', 'purchase_shadow_defense.npz'), defense_output=result_array, defense_logits = result_array_logits, 
                 tc_outputs=f_evaluate_origin)


        # these the conf scores after memguard
        np.save(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_member_%s.npy'%str(args.train_size)), result_array[:len1])
        np.save(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_member_%s.npy'%str(args.train_size)), result_array[len1:len1+len2])
        np.save(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_nonmember_%s.npy'%str(args.train_size)), result_array[len1+len2:len1+len2+len1])
        np.save(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_non_member_%s.npy'%str(args.train_size)), result_array[len1+len2+len1:])


        np.save(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_member_logit_%s.npy'%str(args.train_size)), result_array_logits[:len1])
        np.save(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_member_logit_%s.npy'%str(args.train_size)), result_array_logits[len1:len1+len2])
        np.save(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_nonmember_logit_%s.npy'%str(args.train_size)), result_array_logits[len1+len2:len1+len2+len1])
        np.save(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_non_member_logit_%s.npy'%str(args.train_size)), result_array_logits[len1+len2+len1:])


        sys.exit() 

    if(args.isMemGuard):
        file_path='./'
        print("\t====> loading MemGuard's modified output")

        shadow_train_performance = ( np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_member_%s.npy'%str(args.train_size))), shadow_train_performance[1] )
        shadow_test_performance = ( np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_nonmember_%s.npy'%str(args.train_size))), shadow_test_performance[1] )
    
        target_train_performance = ( np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_member_%s.npy'%str(args.train_size))), target_train_performance[1] )
        target_test_performance = ( np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_non_member_%s.npy'%str(args.train_size))), target_test_performance[1] )


    #print("\t===> correctness-based, entropy-based, m-entropy-based-, confidence-based attacks ", args.path)
    MIA = black_box_benchmarks(shadow_train_performance,shadow_test_performance,
                         target_train_performance,target_test_performance,num_classes=30)
    MIA._mem_inf_benchmarks()

    save_tag = args.save_tag


    y_score = np.r_[ MIA.s_tr_m_entr, MIA.t_tr_m_entr, MIA.s_te_m_entr, MIA.t_te_m_entr ]
    y_true = np.r_[ np.ones( len(MIA.t_tr_labels) + len(MIA.s_tr_labels)  ), np.zeros( len(MIA.t_te_labels) + len(MIA.s_te_labels) ) ]


    y_score *= -1 # roc default > takes positive label; but we want < takes positive label
    #np.save( os.path.join(output_save_path, 'm-entropy-based-%s.npy'%save_tag), np.r_[y_true, y_score] ) 
    get_tpr(y_true, y_score, args.fpr_threshold, 'm-entropy-based-%s.npy'%save_tag)

    y_score = np.r_[ MIA.s_tr_conf, MIA.t_tr_conf, MIA.s_te_conf, MIA.t_te_conf ] 
    #np.save( os.path.join(output_save_path,  'confidence-based-%s.npy'%save_tag), np.r_[y_true, y_score] ) 
    get_tpr(y_true, y_score, args.fpr_threshold, 'confidence-based-%s.npy'%save_tag)

    y_score = np.r_[ MIA.s_tr_entr, MIA.t_tr_entr,  MIA.s_te_entr, MIA.t_te_entr ]
    y_score *= -1 # roc default > takes positive label; but we want < takes positive label 
    #np.save( os.path.join(output_save_path,  'entropy-based-%s.npy'%save_tag), np.r_[y_true, y_score] )
    get_tpr(y_true, y_score, args.fpr_threshold, 'entropy-based-%s.npy'%save_tag)


elif(args.attack == 'loss'):


    def get_output(model, x, batch_size=1024):
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
        
    import tensorflow as tf
    import math

    def loss_threshold_attack(model, shadow_model, x_, y_true, shadow_x, shadow_y, non_Mem_Generator, num_classes=100):
        criterion_classifier = nn.CrossEntropyLoss(reduction='mean')
        x_train_sha = torch.from_numpy(shadow_x).type(torch.FloatTensor) 
        y_train_sha = torch.from_numpy(shadow_y).type(torch.LongTensor) 
        if use_cuda:
            x_train_sha = x_train_sha.cuda() 
            y_train_sha = y_train_sha.cuda()
            x_ = x_.cuda() 

        pred_outputs = get_output(shadow_model, x_train_sha, 256) #shadow_model(x_train_sha)   
        if(args.isModifyOutput):
            pred_outputs = alter_output( shadow_model, x_train_sha.cpu().numpy(), pred_outputs.data.cpu().numpy() , non_Mem_Generator=non_Mem_Generator)
            pred_outputs = torch.from_numpy(pred_outputs).type(torch.FloatTensor).cuda() 

        one_hot = np.zeros( (y_true.shape[0], num_classes) )
        for i in range( y_true.shape[0] ):
            one_hot[i] = tf.keras.utils.to_categorical( y_true[i], num_classes=num_classes) 
        # average loss for the shadow model on its training samples
        avg_loss= criterion_classifier(pred_outputs , y_train_sha  ).view([-1,1])
        avg_loss = avg_loss.cpu().detach().numpy()[0]


        test_pred_outputs = get_output(model, x_, 256) # model(x_) 
        if(args.isModifyOutput): 
            test_pred_outputs = alter_output( model, x_.cpu().numpy(), test_pred_outputs.data.cpu().numpy() , non_Mem_Generator=non_Mem_Generator)
            test_pred_outputs = torch.from_numpy(test_pred_outputs).type(torch.FloatTensor).cuda() 

        preds = F.softmax(test_pred_outputs, 1).cpu().detach().numpy() 

        x_loss = np.asarray([-math.log(y_pred) if y_pred > 0 else y_pred+1e-50 for y_pred in preds[one_hot.astype(bool) ] ])
        m_pred = np.where(x_loss <= avg_loss, 1, 0) 
        return m_pred

    def get_pred_loss(model, x, y, non_Mem_Generator, isModifyOutput=0):
        criterion = nn.CrossEntropyLoss(reduction='none')
        pred_outputs = get_output(model, x)


        if(isModifyOutput): 
            pred_outputs = alter_output( model, x.cpu().numpy(), pred_outputs.data.cpu().numpy() , non_Mem_Generator=non_Mem_Generator)
            preds = F.softmax( torch.from_numpy(pred_outputs).type(torch.FloatTensor).cuda(), 1).cpu().detach().numpy() 
        else:
            preds = F.softmax( pred_outputs, 1).cpu().detach().numpy() 

        one_hot = np.zeros( (len(y), num_classes) )
        for i in range( len(y) ):
            one_hot[i] = tf.keras.utils.to_categorical( y[i], num_classes=num_classes) 
 
        loss = np.asarray([-math.log(y_pred) if y_pred > 0 else y_pred+1e-50 for y_pred in preds[one_hot.astype(bool) ] ])
        return loss

    def memguard_get_pred_loss(preds, y):
        criterion = nn.CrossEntropyLoss(reduction='none') 


        one_hot = np.zeros( (len(y), num_classes) )
        for i in range( len(y) ):
            one_hot[i] = tf.keras.utils.to_categorical( y[i], num_classes=num_classes) 
 
        loss = np.asarray([-math.log(y_pred) if y_pred > 0 else y_pred+1e-50 for y_pred in preds[one_hot.astype(bool) ] ])
        return loss


    query_model = best_model

    known_tr_loss = get_pred_loss(query_model, torch.from_numpy(tr_members).type(torch.FloatTensor), 
                                    torch.from_numpy(tr_members_y).type(torch.LongTensor), non_Mem_Generator, args.isModifyOutput)
    known_te_loss = get_pred_loss(query_model, torch.from_numpy(tr_non_members).type(torch.FloatTensor), 
                                    torch.from_numpy(tr_non_members_y).type(torch.LongTensor), non_Mem_Generator, args.isModifyOutput)


    tr_loss = get_pred_loss(best_model, mia_test_members_data_tensor, mia_test_members_label_tensor, non_Mem_Generator, args.isModifyOutput)
    te_loss = get_pred_loss(best_model, mia_test_nonmembers_data_tensor, mia_test_nonmembers_label_tensor, non_Mem_Generator, args.isModifyOutput)


    if(args.isMemGuard):
        file_path = './'
        print("loading MemGuard's modified output")

        known_tr_loss = memguard_get_pred_loss( np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_member_%s.npy'%str(args.train_size))), \
                                                        torch.from_numpy(tr_members_y).type(torch.LongTensor))

        known_te_loss = memguard_get_pred_loss( np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_nonmember_%s.npy'%str(args.train_size))), \
                                                        torch.from_numpy(tr_non_members_y).type(torch.LongTensor))


        tr_loss = memguard_get_pred_loss( np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_member_%s.npy'%str(args.train_size))), \
                                                        mia_test_members_label_tensor)

        te_loss = memguard_get_pred_loss( np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_non_member_%s.npy'%str(args.train_size))), \
                                                        mia_test_nonmembers_label_tensor)        



    y_true = np.r_[ np.ones(len(known_tr_loss)+len(tr_loss)) , np.zeros(len(known_te_loss) +len(te_loss)) ]
    y_score = np.r_[ known_tr_loss, tr_loss, known_te_loss, te_loss ]


    y_score *= -1 # roc default > takes positive label; but we want < takes positive label

    save_tag = args.save_tag
    #np.save( os.path.join(output_save_path, 'loss-based-%s.npy'%save_tag), np.r_[y_true, y_score] ) 
    get_tpr(y_true, y_score, args.fpr_threshold, 'loss-based-%s.npy'%save_tag)

    '''
    print("\t===> loss-based attack ", args.path)
    m_pred = loss_threshold_attack(best_model, query_model, torch.cat( (mia_test_members_data_tensor, mia_test_nonmembers_data_tensor), 0), 
                                    torch.cat( (mia_test_members_label_tensor, mia_test_nonmembers_label_tensor), 0), 
                                    tr_members, tr_members_y,
                                    non_Mem_Generator,  30)

    m_true = np.r_[ np.ones( len(mia_test_members_data_tensor) ), np.zeros( len(mia_test_nonmembers_data_tensor) )  ] 
    pred_label = m_pred
    eval_label = m_true
    print("\tAccuracy: %.4f | Precision %.4f | Recall %.4f | f1_score %.4f" % ( accuracy_score(eval_label, pred_label), precision_score(eval_label,pred_label),\
                                recall_score(eval_label,pred_label), f1_score(eval_label,pred_label)))

    '''

elif(args.attack == 'nn'):
    import logging
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import tensorflow as tf

    class InferenceAttack_BB(nn.Module):
        def __init__(self,num_classes):
            self.num_classes=num_classes
            super(InferenceAttack_BB, self).__init__()
            
            self.features=nn.Sequential(
                nn.Linear(30,1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.Linear(512,64),
                nn.ReLU(),
                )

            self.labels=nn.Sequential(
               nn.Linear(num_classes,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                )

            self.loss=nn.Sequential(
               nn.Linear(1,num_classes),
                nn.ReLU(),
                nn.Linear(num_classes,64),
                nn.ReLU(),
                )
            
            self.combine=nn.Sequential(
                nn.Linear(64*3,512),
                nn.ReLU(),
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,1),
                )

            for key in self.state_dict():
                # print (key)
                if key.split('.')[-1] == 'weight':    
                    nn.init.normal_(self.state_dict()[key], std=0.01)
                    
                elif key.split('.')[-1] == 'bias':
                    self.state_dict()[key][...] = 0
            self.output= nn.Sigmoid()
        
        def forward(self,x1,one_hot_labels,loss):

            out_x1 = self.features(x1)
            
            out_l = self.labels(one_hot_labels)
            
            out_loss= self.loss(loss)

            is_member =self.combine( torch.cat((out_x1,out_l,out_loss),1))
            
            return self.output(is_member)
  
    def attack_bb(train_data, labels, attack_data, attack_label, model, inference_model, classifier_criterion, classifier_criterion_noreduct, criterion_attck, classifier_optimizer,
                  optimizer, epoch, use_cuda, uniform_out=None, num_batchs=1000, is_train=False, batch_size=64, non_Mem_Generator=None, eval_set_identifier_4_memguard='train'):
        global best_acc

        losses = AverageMeter()
        top1 = AverageMeter()
        mtop1_a = AverageMeter()
        mtop5_a = AverageMeter()

        acys = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        F1_Scores = AverageMeter()

        inference_model.eval()
        
        skip_batch=0
        
        if is_train:
            inference_model.train()
        
        model.eval()
        
        batch_size = batch_size//2
        #len_t =  min((len(attack_data)//batch_size) ,(len(train_data)//batch_size))-1
        
        len_t = len(train_data)//batch_size
        if len(train_data)%batch_size:
            len_t += 1



        if(args.isMemGuard):

            file_path='./'
            # load the logit values instead of the confidence scores
            shadow_train = np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_member_logit_%s.npy'%str(args.train_size)))
            shadow_test = np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_known_nonmember_logit_%s.npy'%str(args.train_size)))
            target_train = np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_member_logit_%s.npy'%str(args.train_size)))
            target_test = np.load(os.path.join(file_path, 'memguard_defense_results', 'memguard_test_non_member_logit_%s.npy'%str(args.train_size)))


            ##### load memguard's prediciton logits on known members and non-members 
            if(eval_set_identifier_4_memguard == 'train'):
                tr_outputs =  shadow_train[:int(len(shadow_train)* ( tr_frac/(tr_frac+val_frac)) )]
                te_outputs = shadow_test[:int(len(shadow_test)* ( tr_frac/(tr_frac+val_frac)) )]

            elif(eval_set_identifier_4_memguard == 'val'):
                tr_outputs =  shadow_train[int(len(shadow_train)* ( tr_frac/(tr_frac+val_frac)) ) :]
                te_outputs =  shadow_test[int(len(shadow_test)* ( tr_frac/(tr_frac+val_frac)) ) :]
            elif(eval_set_identifier_4_memguard=='test'):
                tr_outputs = target_train
                te_outputs = target_test 

 

            all_tr_outputs = tr_outputs
            all_te_outputs = te_outputs

        for ind in range(skip_batch, len_t):
            if ind >= skip_batch+num_batchs:
                break
             

            tr_input = train_data[ind*batch_size:(ind+1)*batch_size]
            tr_target = labels[ind*batch_size:(ind+1)*batch_size]

            te_input = attack_data[ind*batch_size:(ind+1)*batch_size]
            te_target = attack_label[ind*batch_size:(ind+1)*batch_size]
            
            tr_input, tr_target = tr_input.cuda(), tr_target.cuda()
            te_input , te_target = te_input.cuda(), te_target.cuda()

            v_tr_input, v_tr_target = torch.autograd.Variable(tr_input), torch.autograd.Variable(tr_target)
            v_te_input, v_te_target = torch.autograd.Variable(te_input), torch.autograd.Variable(te_target)
 

            model_input = torch.cat((v_tr_input, v_te_input))
            if(not args.isMemGuard):
                # compute output 

                if(is_train):
                    pred_outputs = model(model_input)
 
                else:
                    with torch.no_grad():
                        pred_outputs = model(model_input) 


                if(args.isModifyOutput): 
                    pred_outputs = alter_output( model, model_input, pred_outputs.data.cpu().numpy() , non_Mem_Generator=non_Mem_Generator) 
                    pred_outputs = torch.from_numpy(pred_outputs).type(torch.FloatTensor).cuda()

            else:
                 
                pred_outputs = np.r_[ all_tr_outputs[ind*batch_size:(ind+1)*batch_size], all_te_outputs[ind*batch_size:(ind+1)*batch_size] ] 
                pred_outputs = torch.from_numpy(pred_outputs).type(torch.FloatTensor).cuda()



            infer_input= torch.cat((v_tr_target,v_te_target))
            
            # "label for training the MI inference_model"
            one_hot_tr = torch.from_numpy(np.zeros(pred_outputs.size())).cuda().type(torch.cuda.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1,1]).data,1)

            infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

            loss_= classifier_criterion_noreduct(pred_outputs, infer_input).view([-1,1])
            #torch.autograd.Variable(torch.from_numpy(c.view([-1,1]).data.cpu().numpy()).cuda())

            preds = torch.autograd.Variable(torch.from_numpy(pred_outputs.data.cpu().numpy()).cuda())


            member_output = inference_model(pred_outputs, infer_input_one_hot, loss_) 


            is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.zeros(v_tr_input.size(0)),np.ones(v_te_input.size(0)))),[-1,1])).cuda()
            
            v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

            loss = criterion_attck(member_output, v_is_member_labels)

            # measure accuracy and record loss
            prec1=np.mean((member_output.data.cpu().numpy() >0.5)==v_is_member_labels.data.cpu().numpy())
            losses.update(loss.item(), model_input.size(0))
            top1.update(prec1, model_input.size(0))


            predicted_member_labels = member_output.data.cpu().numpy() >0.5
            actual_member_labels = v_is_member_labels.data.cpu().numpy()


            accuracy = tf.keras.metrics.Accuracy()
            precision = tf.keras.metrics.Precision()
            recall = tf.keras.metrics.Recall()
            accuracy.update_state(actual_member_labels, predicted_member_labels)
            precision.update_state(actual_member_labels, predicted_member_labels)
            recall.update_state(actual_member_labels, predicted_member_labels)

            #print( precision.result(),  recall.result(),  2 * (precision.result() * recall.result()) / (precision.result() + recall.result()))
            if( precision.result() + recall.result() != 0 ):
                F1_Score = 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())
            else:
                F1_Score = 0

            acys.update(accuracy.result(),  model_input.size(0))
            precisions.update(precision.result(),  model_input.size(0))
            recalls.update(recall.result(),  model_input.size(0))
            F1_Scores.update(F1_Score,  model_input.size(0))


            # compute gradient and do SGD step
            optimizer.zero_grad()
            if is_train:
                loss.backward()
                optimizer.step()

            # plot progress
            if False and ind%10==0:
                print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                        batch=ind ,
                        size=len_t,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        ))

            if(ind==0):
                raw_pred_socre = member_output.data.cpu().numpy()
                true_label = v_is_member_labels.data.cpu().numpy()
            else:
                raw_pred_socre = np.r_[ raw_pred_socre,  member_output.data.cpu().numpy()]
                true_label = np.r_[ true_label,  v_is_member_labels.data.cpu().numpy()] 

        return (losses.avg, top1.avg, acys.avg, precisions.avg, recalls.avg, F1_Scores.avg, raw_pred_socre, true_label)


    user_lr=0.0005
    at_lr=0.0005
    at_schedule=[100]
    at_gamma=0.1
    n_classes=num_classes
    criterion_classifier = nn.CrossEntropyLoss(reduction='none')
    attack_criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    best_opt=optim.Adam(best_model.parameters(), lr=user_lr)
    best_at_val_acc=0
    best_at_test_acc=0
    attack_epochs=200 
    resume_best=args.path
    attack_model = InferenceAttack_BB(n_classes)
    attack_model = attack_model.cuda()
    attack_optimizer = optim.Adam(attack_model.parameters(),lr=at_lr)
    non_Mem_Generator = random_flip
    BATCH_SIZE = 64
    save_tag = args.save_tag

    for epoch in range(attack_epochs):
        if epoch in at_schedule:
            for param_group in attack_optimizer.param_groups:
                param_group['lr'] *= at_gamma
                print('Epoch %d Local lr %f'%(epoch,param_group['lr']))
 


        at_loss, at_acc, at_acy, at_precision, at_recall, at_f1, _, _ = attack_bb(mia_train_members_data_tensor, mia_train_members_label_tensor,
                                    mia_train_nonmembers_data_tensor, mia_train_nonmembers_label_tensor,
                                    best_model, attack_model, criterion, criterion_classifier, attack_criterion, best_opt,
                                    attack_optimizer, epoch, use_cuda, non_Mem_Generator=non_Mem_Generator , is_train=True, batch_size=BATCH_SIZE, eval_set_identifier_4_memguard='train' )

        at_val_loss, at_val_acc, at_val_acy, at_val_precision, at_val_recall, at_val_f1, _, _  = attack_bb(mia_val_members_data_tensor, mia_val_members_label_tensor,
                                            mia_val_nonmembers_data_tensor, mia_val_nonmembers_label_tensor,
                                            best_model, attack_model, criterion, criterion_classifier, attack_criterion, best_opt,
                                            attack_optimizer, epoch, use_cuda, non_Mem_Generator=non_Mem_Generator , is_train=False, batch_size=BATCH_SIZE, eval_set_identifier_4_memguard='val' )

        is_best = at_val_acc > best_at_val_acc

        if is_best:
            at_test_loss, best_at_test_acc, at_best_acy, at_best_precision, at_best_recall, at_best_f1, y_score, y_true  = attack_bb(mia_test_members_data_tensor, mia_test_members_label_tensor,
                                                       mia_test_nonmembers_data_tensor, mia_test_nonmembers_label_tensor, 
                                                       best_model, attack_model, criterion, criterion_classifier, attack_criterion, best_opt,
                                                       attack_optimizer, epoch, use_cuda, non_Mem_Generator=non_Mem_Generator ,  is_train=False, batch_size=BATCH_SIZE, eval_set_identifier_4_memguard='test' )


        best_at_val_acc = max(best_at_val_acc, at_val_acc)
        if(epoch == attack_epochs-1):
            print()
            print("\t===>   NN-based attack ", args.path)

        if( (epoch+1)%5==0 ):
            #print(' Epoch %d | current stats acy: %.4f precision: %.4f recall: %.4f F1_Score: %.4f | best test stats: %.4f precision: %.4f recall: %.4f F1_Score: %.4f '\
            #            %(epoch, at_val_acc, at_val_precision, at_val_recall, at_val_f1,\
            #                 best_at_test_acc, at_best_precision, at_best_recall, at_best_f1) , flush=True)
            print(' Epoch %d '%epoch)
            get_tpr(y_true, y_score, args.fpr_threshold, 'nn-based-%s.npy'%save_tag)
 
    #np.save( os.path.join(output_save_path, 'nn-based-%s.npy'%save_tag), np.r_[y_true, y_score] )
    get_tpr(y_true, y_score, args.fpr_threshold, 'nn-based-%s.npy'%save_tag)











