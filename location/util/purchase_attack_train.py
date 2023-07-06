from purchase_models import *

def train_attack(train_data,labels,attack_data,attack_label,model,attack_model,criterion,attack_criterion,optimizer,
                 attack_optimizer,epoch,use_cuda,num_batchs=100000,skip_batch=0,debug_='MEDIUM',batch_size=16):
    model.eval()
    attack_model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    r=np.arange(len(attack_data))
    np.random.shuffle(r)
    
    end = time.time()
    batch_size = batch_size//2
    len_t =  min((len(attack_data)//batch_size) ,(len(train_data)//batch_size))-1

    for ind in range(skip_batch, len_t):

        if ind >= skip_batch+num_batchs:
            break
        # measure data loading time
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        
        inputs_attack = attack_data[r[ind*batch_size:(ind+1)*batch_size]]
        targets_attack = attack_label[r[ind*batch_size:(ind+1)*batch_size]]

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs_attack , targets_attack = inputs_attack.cuda(), targets_attack.cuda()

        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        inputs_attack , targets_attack = torch.autograd.Variable(inputs_attack), torch.autograd.Variable(targets_attack)

        # compute output
        outputs, h_layer = model(inputs)
        outputs_non, h_layer_non = model(inputs_attack)

        comb_inputs_h = torch.cat((h_layer,h_layer_non))
        comb_inputs = torch.cat((outputs,outputs_non))

        attack_input = comb_inputs
        
        one_hot_tr = torch.from_numpy((np.zeros((attack_input.size(0),outputs.size(1))))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr=one_hot_tr.scatter_(1,torch.cat((targets,targets_attack)).type(torch.cuda.LongTensor).view([-1,1]).data,1)
        
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        attack_output = attack_model(attack_input,comb_inputs_h,infer_input_one_hot).view([-1])

        att_labels = np.zeros((inputs.size(0)+inputs_attack.size(0)))
        att_labels [:inputs.size(0)] =1.0
        att_labels [inputs.size(0):] =0.0
        is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor)

        if use_cuda:
            is_member_labels = is_member_labels.cuda()

        v_is_member_labels = torch.autograd.Variable(is_member_labels)
        
        loss_attack = attack_criterion(attack_output, v_is_member_labels)
        
        prec1=np.mean(np.equal((attack_output.data.cpu().numpy() >0.5),(v_is_member_labels.data.cpu().numpy()> 0.5)))
        losses.update(loss_attack.item(), attack_input.size(0))
        top1.update(prec1, attack_input.size(0))
        
        #print ( attack_output.data.cpu().numpy(),v_is_member_labels.data.cpu().numpy() ,attack_input.data.cpu().numpy())

        # compute gradient and do SGD step
        attack_optimizer.zero_grad()
        loss_attack.backward()
        attack_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if False and debug_=='HIGH' and ind%100==0:
            print('Attack model: ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f}'
                  .format(
                      batch=ind + 1,
                      size=len_t,
                      data=data_time.avg,
                      bt=batch_time.avg,
                      loss=losses.avg,
                      top1=top1.avg,
                  ))

    return (losses.avg, top1.avg)


def test_attack(train_data,labels,attack_data,attack_label,model,attack_model,criterion,attack_criterion,
                optimizer,attack_optimizer,epoch,use_cuda,batch_size=16,debug_='MEDIUM'):

    model.eval()
    attack_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    end = time.time()
    len_t =  min((len(attack_data)//batch_size) ,(len(train_data)//batch_size))-1
    for ind in range(len_t):
        # measure data loading time
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        
        inputs_attack = attack_data[ind*batch_size:(ind+1)*batch_size]
        targets_attack = attack_label[ind*batch_size:(ind+1)*batch_size]        

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs_attack , targets_attack = inputs_attack.cuda(), targets_attack.cuda()
        
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            inputs_attack , targets_attack = torch.autograd.Variable(inputs_attack), torch.autograd.Variable(targets_attack)


        # compute output
        outputs,h_layer = model(inputs)
        outputs_non,h_layer_non = model(inputs_attack)
        

        comb_inputs_h = torch.cat((h_layer,h_layer_non))
        comb_inputs = torch.cat((outputs,outputs_non))

        attack_input = comb_inputs        
        
        one_hot_tr = torch.from_numpy((np.zeros((attack_input.size(0),outputs.size(1))))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr=one_hot_tr.scatter_(1,torch.cat((targets,targets_attack)).type(torch.cuda.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)


        attack_output = attack_model(attack_input,comb_inputs_h,infer_input_one_hot).view([-1])

        att_labels = np.zeros((inputs.size(0)+inputs_attack.size(0)))
        att_labels [:inputs.size(0)] =1.0
        att_labels [inputs.size(0):] =0.0

        is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor)

        if use_cuda:
            is_member_labels = is_member_labels.cuda()
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels)

        loss = attack_criterion(attack_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1=np.mean(np.equal((attack_output.data.cpu().numpy() >0.5),(v_is_member_labels.data.cpu().numpy()> 0.5)))
        losses.update(loss.item(), attack_input.size(0))
        top1.update(prec1, attack_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if False and debug_=='HIGH' and ind%100==0:
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '
                  .format(
                      batch=ind + 1,
                      size=len_t,
                      data=data_time.avg,
                      bt=batch_time.avg,
                      loss=losses.avg,
                      top1=top1.avg,
                  ))

    return (losses.avg, top1.avg)
