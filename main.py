"""
MNIST DNN training from scratch
"""

import os
import logging
import time
import argparse
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import models
from models.modules import *
from dataset import get_loader

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model', type=str, choices=['cnn_mnist', 'vgg7'], help='model type')
parser.add_argument('--model_ref', type=str, choices=['cnn_mnist_torch', 'vgg7_torch'], help='reference model type')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 80], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)    

    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)

    logger.info(args)

    train_loader, valid_loader, num_classes = get_loader(args)

    # Prepare the model
    logger.info('==> Building model..\n')
    model_cfg = getattr(models, args.model)
    model_ref_cfg = getattr(models, args.model_ref)

    model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
    model_ref = model_ref_cfg.base(*model_cfg.args, **model_cfg.kwargs)
    
    logger.info(model)
    logger.info(model_ref)

    if args.use_cuda:
        model = model.cuda()
        model_ref = model_ref.cuda()

    # Calibrate the weights
    with torch.no_grad():
        conv = 0
        sw_conv_weight = []
        sw_linear_weight = []
        sw_linear_bias = []
        for m in model_ref.modules():
            if isinstance(m, nn.Conv2d):
                sw_conv_weight.append(m.weight.data)
                print(f"Torch Conv: {m.weight.data.size()} | min = {m.weight.data.min()} | max = {m.weight.data.max()}")
                conv += 1
            elif isinstance(m, nn.Linear):
                sw_linear_weight.append(m.weight.data)
                sw_linear_bias.append(m.bias.data)
                print(f"Torch FC: {m.weight.data.size()} | min = {m.weight.data.min()} | max = {m.weight.data.max()}")    
        print("================\n")

        hw_conv = 0
        hw_lin = 0
        for n in range(len(model.features)):
            f = model.features[n]
            if isinstance(f, HWconv2d):
                f.weight.data = sw_conv_weight[hw_conv]
                hw_conv += 1
                print(f"HW Conv: {f.weight.data.size()} | min = {f.weight.data.min()} | max = {f.weight.data.max()}")

        model.fc.weight.data = sw_linear_weight[hw_lin]
        model.fc.bias.data = sw_linear_bias[hw_lin]
        print(f"HW FC: {model.fc.weight.data.size()} | min = {model.fc.weight.data.min()} | max = {model.fc.weight.data.max()}")


    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(model_ref.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0, nesterov=False)

    # Training
    epoch_time = AverageMeter()
    best_acc = 0.
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'val_loss', 'val_acc', 'best_acc']

    for epoch in range(args.epochs):

        # Training phase
        train_results = train(train_loader, model, model_ref, optimizer, criterion)

        # Test phase
        valid_results = test(valid_loader, model, model_ref, criterion)
        is_best = valid_results['acc'] > best_acc

        if is_best:
            best_acc = valid_results['acc']
        
        values = [epoch + 1, args.lr, train_results['loss'], train_results['acc'], valid_results['loss'], valid_results['acc'], best_acc]
        print_table(values, columns, epoch, logger)

if __name__ == '__main__':
    main()