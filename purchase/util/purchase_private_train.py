from purchase_models import *

def train_privatly(train_data,labels,model,inference_model,criterion,optimizer,epoch,use_cuda,
                   num_batchs=10000,skip_batch=0,alpha=0.5,verbose=False,batch_size=16,loss_fun='mean'):
    model.train()
    inference_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    attack_criterion = nn.MSELoss()
    len_t =  (len(train_data)//batch_size)-1

    for ind in range(skip_batch,len_t):
        if ind >= skip_batch+num_batchs:
            break

        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        
        targets = labels[ind*batch_size:(ind+1)*batch_size]

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs,h_layer = model(inputs)

        one_hot_tr = torch.from_numpy((np.zeros((outputs.size(0),outputs.size(1))))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.cuda.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        
        inference_output = inference_model(outputs,h_layer,infer_input_one_hot)
        att_labels = np.ones((inputs.size(0)))
        is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor)

        if use_cuda:
            is_member_labels = is_member_labels.cuda()

        v_is_member_labels = torch.autograd.Variable(is_member_labels)

        loss = criterion(outputs, targets) +(alpha*(inference_output.mean()-0.5))
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if False and verbose and ind%100==0:
            print  (alpha, '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))

    return (losses.avg, top1.avg)