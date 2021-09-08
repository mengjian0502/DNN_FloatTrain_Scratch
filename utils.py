"""
Utilities of training
"""

import torch
import time
import shutil
import tabulate
import numpy as np
import pandas as pd
from models.modules import HWconv2d

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(trainloader, net, net_ref, optimizer, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses_ref = AverageMeter()
    top1_ref = AverageMeter()
    top5_ref = AverageMeter()

    # switch to train mode
    net_ref.train()
    
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda()
        one_hot_targets = torch.nn.functional.one_hot(targets, 10).float()

        # # reference model    
        out_ref = net_ref(inputs)
        loss_ref = criterion(out_ref, one_hot_targets)


        # hw model
        outputs, loss = net.feed_forward(inputs, one_hot_targets)

        net.zero_grad()
        net.feed_backward()

        # # Torch SGD 
        optimizer.zero_grad()
        loss_ref.backward()
        optimizer.step()

        prec1_ref, prec5_ref = accuracy(out_ref.data, targets, topk=(1, 5))
        losses_ref.update(loss_ref.item(), inputs.size(0))
        top1_ref.update(prec1_ref.item(), inputs.size(0))
        top5_ref.update(prec5_ref.item(), inputs.size(0))

        # Pytorch gradient
        print(f"\nGradient comparison")
        for v in net_ref.parameters():
            print(f"Torch Grad: {list(v.size())} | grad min {v.grad.min()} | grad max {v.grad.max()}")

        # convolution layer
        for n in range(len(net.features)):
            f = net.features[n]
            if isinstance(f, HWconv2d):
                grad = f.w_grad
                print(f"HW Conv Grad: {list(grad.size())} | grad min {grad.min()} | grad max {grad.max()}")
        # FC layer
        print(f"HW FC Grad: {net.fc.w_grad.size()} | min = {net.fc.w_grad.min()} | max = {net.fc.w_grad.max()}")
        print(f"HW FC Grad: {net.fc.b_grad.size()} | min = {net.fc.b_grad.min()} | max = {net.fc.b_grad.max()}\n")

        net.weight_update()
        
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        print("HW Model Acc: Batch [{}]/[{}], top1 = {}, top5 = {}, loss = {}".format(batch_idx, len(trainloader), prec1.item(), prec5.item(), loss.item()))
        print("SW Ref Model Acc: Batch [{}]/[{}], top1 = {}, top5 = {}, loss = {}\n".format(batch_idx, len(trainloader), prec1_ref.item(), prec5_ref.item(), loss_ref.item()))

    res = {
        'acc':top1_ref.avg,
        'loss':losses_ref.avg,
        } 
    return res



def test(testloader, net, net_ref, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    test_loss = 0

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            one_hot_targets = torch.nn.functional.one_hot(targets, 10).float()

            outputs, loss = net.feed_forward(inputs, one_hot_targets)

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            test_loss += loss.item()

            batch_time.update(time.time() - end)
            end = time.time()
            # break
    res={
        'acc':top1.avg,
        'loss':losses.avg,
        } 
    return res


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)

def adjust_learning_rate_schedule(optimizer, epoch, gammas, schedule, lr, mu):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, save_path+filename)
    if is_best:
        shutil.copyfile(save_path+filename, save_path+'model_best.pth.tar')


def log2df(log_file_name):
    '''
    return a pandas dataframe from a log file
    '''
    with open(log_file_name, 'r') as f:
        lines = f.readlines() 
    # search backward to find table header
    num_lines = len(lines)
    for i in range(num_lines):
        if lines[num_lines-1-i].startswith('---'):
            break
    header_line = lines[num_lines-2-i]
    num_epochs = i
    columns = header_line.split()
    df = pd.DataFrame(columns=columns)
    for i in range(num_epochs):
        df.loc[i] = [float(x) for x in lines[num_lines-num_epochs+i].split()]
    return df 
