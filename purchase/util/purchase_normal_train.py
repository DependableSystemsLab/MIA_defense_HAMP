from purchase_models import *

def train(train_data,labels,model,criterion,optimizer,epoch,use_cuda,device=torch.device('cuda'),num_batchs=999999,debug_='MEDIUM',batch_size=32, uniform_reg=False):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    end = time.time()
    len_t =  (len(train_data)//batch_size)-1
    
    for ind in range(len_t):
        if ind > num_batchs:
            break
        # measure data loading time
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        try:
            outputs,_,_ = model(inputs)
        except:
            try:
                outputs,_ = model(inputs)
            except:
                outputs = model(inputs)
        uniform_=torch.ones(len(outputs))/len(outputs)
        
        if uniform_reg==True:
            loss = criterion(outputs, targets) + F.kl_div(uniform_,outputs)
        else:
            loss = criterion(outputs, targets)

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
        if False and debug_=='HIGH' and ind%100==0:
            print  ('Classifier: ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))

    return (losses.avg, top1.avg)


# In[9]:


def test(test_data,labels,model,criterion,use_cuda,device=torch.device('cuda'), debug_='MEDIUM',batch_size=16):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time() 
    
    len_t = len(test_data)//batch_size
    if len(test_data)%batch_size:
        len_t += 1

    data_time.update(time.time() - end)
 
    total = 0
    for ind in range(len_t):
        inputs =  test_data[ind*batch_size:(ind+1)*batch_size].to(device)
        targets = labels[ind*batch_size:(ind+1)*batch_size].to(device)

        total += len(inputs)
        # compute output
        try:
            outputs,_,_ = model(inputs)
        except:
            try:
                outputs,_ = model(inputs)
            except:
                outputs = model(inputs)
 
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))


 

    return (losses.avg, top1.avg)








