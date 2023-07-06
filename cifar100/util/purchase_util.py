from purchase_models import *

use_cuda = torch.cuda.is_available()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)

def save_checkpoint_user(user_num, state, is_best, checkpoint=None, filename='checkpoint.pth.tar',extra_checkpoints=False):
    assert (checkpoint != None), 'Error: No checkpoint path provided!'

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    if extra_checkpoints:
        e_path=checkpoint+'/user_%d_checkpoints'%user_num
        if not os.path.isdir(e_path):
            mkdir_p(e_path)
        e_filepath=e_path+'/checkpoint_epoch_%d.pth.tar'%state['epoch']
        print('User %d saving extra checkpoint @epoch %d'%(user_num,state['epoch']))
        torch.save(state, e_filepath)

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'user_%d_model_best.pth.tar'%(user_num)))

def save_checkpoint_global(state, is_best, checkpoint=None, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    assert (checkpoint != None), 'Error: No checkpoint path provided!'

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    filepath = os.path.join(checkpoint, filename)
    #torch.save(state, filepath)
    if is_best:
        filepath = os.path.join(checkpoint, best_filename)
        torch.save(state, filepath)
        #shutil.copyfile(filepath, os.path.join(checkpoint, best_filename))


def write_csv(csv_file_path,row,header=False):
    if header:
        with open(csv_file_path, 'w') as f:
            f_w = csv.writer(f)
            f_w.writerow(row)
    else:
        with open(csv_file_path, 'a') as f:
            f_w = csv.writer(f)
            f_w.writerow(row)

def train_pub(train_data,labels,idx_predictions,model,t_softmax,optimizer,epoch,use_cuda,num_batchs=999999,debug_='MEDIUM',batch_size=16):
    # switch to train mode
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()

    len_t =  (len(train_data)//batch_size)-1
    
    for ind in range(len_t):
        if ind > num_batchs:
            break

        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs,_ = model(inputs)

        loss = F.kl_div(F.log_softmax(outputs/t_softmax, dim=1), F.softmax(targets/t_softmax, dim=1))
        # measure loss
        losses.update(loss.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if False and debug_=='HIGH' and ind%100==0:
            print  ('Classifier: ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f}'.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    ))

    return (losses.avg)