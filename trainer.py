#!/usr/bin/env python3

#MIT License
#
#Copyright (c) 2018 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#python3 trainer.py --gpu=0 --id="baseline_resnet34_cifar10" --arch="resnet34" --root="./" --data="../datasets/cifar10" --dataset="cifar10"
#python3 trainer.py --gpu=1 --id="rmor_resnet34_cifar100_b0.25" --arch="rmor34" --root="./" --data="../datasets/cifar100" --dataset="cifar100"

import numpy as np
import os
import sys
#Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
# ArgParser
import argparse
from time import gmtime, strftime


def return_cifar10_training(dataset_path, download=False, mini_batch_size=64):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(size=[32,32], padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                            download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True, num_workers=8)
    return trainloader

def return_cifar100_training(dataset_path, download=False, mini_batch_size = 64):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(size=[32,32], padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR100(root=dataset_path, train=True,
                                            download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True, num_workers=8)
    return trainloader

def save_checkpoint(net, epoch, learning_rate, net_name, root_path):
    time_string = strftime("%d%m%Y_%H%M%S", gmtime())
    state = {'net': net.state_dict(), 'epoch': epoch, 'lr': learning_rate, 'time': time_string}
    if not os.path.isdir(root_path): os.mkdir(root_path)
    print('[INFO] Saving checkpoint: ' + 'ckpt_'+str(time_string)+'_ep'+str(epoch)+'.t7')
    if(root_path.endswith('/')): root_path = root_path[:-1]
    torch.save(state, root_path + '/ckpt_'+str(net_name)+'_'+str(time_string)+'_ep'+str(epoch)+'.t7')


def main():
    ##Parser
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')
    parser.add_argument('--id', default='', type=str, help='experiment ID')
    parser.add_argument('--arch', default='resnet34', type=str, help='architecture type: resnet18/152')
    parser.add_argument('--epochs', default=200, type=int, help='total training epochs')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size')
    parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
    parser.add_argument('--root', default='./', type=str, help='root path')
    parser.add_argument('--data', default='./', type=str, help='data path')
    parser.add_argument('--prate', default=100, type=int, help='print-rate on terminal')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset type: cifar10, cifar100')
    args = parser.parse_args()
    DEVICE_ID = args.gpu
    LEARNING_RATE = args.lr
    TOT_EPOCHS = args.epochs
    start_epoch = 0
    MINI_BATCH_SIZE = args.batch
    ROOT_PATH = args.root
    DATASET_PATH = args.data
    NET_TYPE = args.arch
    PRINT_RATE = args.prate
    ID = args.id
    print("[INFO] ID: " + str(ID))
    print("[INFO] Root path: " + str(ROOT_PATH))
    print("[INFO] Dataset path: " + str(DATASET_PATH))
    print("[INFO] Total epochs: " + str(TOT_EPOCHS))
    print("[INFO] Mini-batch size: " + str(MINI_BATCH_SIZE))
    print("[INFO] Learning rate: " + str(LEARNING_RATE))
    print("[INFO] Printing rate: " + str(PRINT_RATE))
    #torch.cuda.set_device(DEVICE_ID)
    #ATTENTION: os.environment must be called before any torch.cuda call
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(DEVICE_ID)
    print('[INFO] Is CUDA available: ' + str(torch.cuda.is_available()))
    print('[INFO] TOT available devices: ' + str(torch.cuda.device_count()))
    print('[INFO] Setting device: ' + str(DEVICE_ID))
    #device = torch.device('cuda:'+str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    #print("[INFO] Torch is using device: " + str(torch.cuda.current_device()))
    if(args.dataset == "cifar10"):
        TOT_CLASSES = 10
    elif(args.dataset == "cifar100"):
        TOT_CLASSES = 100
    else:
        raise ValueError('[ERROR] the dataset ' + args.dataset + ' is unknown.') 
    ##Generate net
    if(NET_TYPE == 'resnet18'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [2,2,2,2], num_classes=TOT_CLASSES)
    elif(NET_TYPE == 'resnet34'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=TOT_CLASSES)
    elif(NET_TYPE == 'resnet50'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(Bottleneck, [3,4,6,3], num_classes=TOT_CLASSES)
    elif(NET_TYPE == 'resnet101'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(Bottleneck, [3,4,23,3], num_classes=TOT_CLASSES)
    elif(NET_TYPE == 'resnet152'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(Bottleneck, [3,8,36,3], num_classes=TOT_CLASSES)
    elif(NET_TYPE == 'mor18'):
        from models.mor import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [2,2,2,2], num_classes=TOT_CLASSES)
    elif(NET_TYPE == 'mor34'):
        from models.mor import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=TOT_CLASSES)
    elif(NET_TYPE == 'moround18'):
        from models.mor import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [2,2,2,2], num_classes=TOT_CLASSES, round_g=True)
    elif(NET_TYPE == 'moround34'):
        from models.mor import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=TOT_CLASSES, round_g=True)
    elif(NET_TYPE == 'moround101'):
        from models.mor import ResNet, BasicBlock, Bottleneck
        net = ResNet(Bottleneck, [3,4,23,3], num_classes=TOT_CLASSES, round_g=True)
    elif(NET_TYPE == 'rmor34'):
        from models.rmor import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=TOT_CLASSES, round_g=False)
    else:
        raise ValueError('[ERROR] the architecture type ' + str(NET_TYPE) + ' is unknown.') 
    print("[INFO] Architecture: " + str(NET_TYPE))          
    net.to(device)
    if args.resume:
        print('[INFO] Resuming from checkpoint: ' + str(args.resume))
        checkpoint = torch.load(str(args.resume))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        print('[INFO] Starting from epoch: ' + str(start_epoch))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    #optimizer =  optim.Adam(net.parameters(), lr=LEARNING_RATE)

    time_string = strftime("%d%m%Y_%H%M%S", gmtime())
    if(ID!=''): writer_path = ROOT_PATH + '/' + ID + '/log/' + time_string
    else: writer_path = ROOT_PATH + '/log' + time_string
    writer = SummaryWriter(log_dir=writer_path)
    if(args.dataset == "cifar10"):
        trainloader = return_cifar10_training(dataset_path=DATASET_PATH, 
                                              download = False, mini_batch_size = MINI_BATCH_SIZE)
    elif(args.dataset == "cifar100"):
        trainloader = return_cifar100_training(dataset_path=DATASET_PATH, 
                                                download = False, mini_batch_size = MINI_BATCH_SIZE)
    else:
        raise ValueError('[ERROR] the dataset ' + args.dataset + ' is unknown.') 
    print('[INFO] Loaded dataset: ' + args.dataset)   
    global_step = 0
    gumbel_tau_array = np.linspace(start=0.9, stop=0.01, num=TOT_EPOCHS-5, endpoint=True)
    gumbel_tau_array = np.hstack((gumbel_tau_array, np.full(5+1, 0.01)))
    for epoch in range(start_epoch, TOT_EPOCHS):  #loop over the dataset multiple times
        loss_list = list()
        for i, data in enumerate(trainloader, 0):     
            # Zero the parameter gradients
            optimizer.zero_grad()      
            # Get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward
            
            if('rmor' in NET_TYPE):
                outputs = net(inputs, gumbel_tau=gumbel_tau_array[epoch])
            else:
                outputs = net(inputs)     
            # Estimate loss
            if(hasattr(net, 'return_regularizer')):
                loss = criterion(outputs, labels) + 0.01 * net.return_regularizer()
            elif(hasattr(net, 'return_loss')):
                loss = criterion(outputs, labels) + 1.0 * net.return_loss(device)
            else:
                loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # Backward
            loss.backward()
            # Optimize
            optimizer.step()
            # Estimate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = 100 * ((predicted == labels).sum().item()) / labels.size(0)         
            if(global_step % 5 == 0):          
                writer.add_scalar('loss', loss, global_step)
                writer.add_scalar('accuracy', accuracy, global_step)
                if('rmor' in NET_TYPE):
                    writer.add_scalar('gumbel_tau', gumbel_tau_array[epoch], global_step)
                if(hasattr(net, 'return_regularizer')):
                    writer.add_scalar('regularizer', net.return_regularizer(), global_step)
                if(hasattr(net, 'return_histograms')):
                    for i, histogram in enumerate(net.return_histograms()):
                        writer.add_histogram('gate_' + str(i), histogram, global_step)
                if(hasattr(net, 'return_activity')):
                    activity = net.return_activity()
                    writer.add_histogram('activity', activity, global_step)
            global_step += 1 #increasing the global step     
            if(i % PRINT_RATE == 0): print('[%d, %5d] lr: %.5f; loss: %.5f; accuracy: %.3f' 
                                           %(epoch, global_step, LEARNING_RATE, np.mean(loss_list), accuracy))
        #Epoch finished
        print('[%d, %5d] lr: %.5f; loss: %.5f; accuracy: %.5f'
               %(epoch, global_step, LEARNING_RATE, np.mean(loss_list), accuracy))
        if(epoch==60 or epoch==120 or epoch==160):
             if(ROOT_PATH.endswith('/')): root_path = ROOT_PATH[:-1]
             else: root_path = ROOT_PATH
             if(ID!=''): checkpoint_path = root_path + '/' + ID + '/checkpoint/'
             else: checkpoint_path = ROOT_PATH + '/checkpoint/'
             save_checkpoint(net, epoch, LEARNING_RATE, NET_TYPE, checkpoint_path)
             LEARNING_RATE = LEARNING_RATE * 0.2
             for g in optimizer.param_groups:
                 g['lr'] = LEARNING_RATE
             print("[INFO] Learning rate: " + str(LEARNING_RATE))
        if(epoch==TOT_EPOCHS-1):
             if(ROOT_PATH.endswith('/')): root_path = ROOT_PATH[:-1]
             else: root_path = ROOT_PATH
             if(ID!=''): checkpoint_path = root_path + '/' + ID + '/checkpoint/'
             else: checkpoint_path = ROOT_PATH + '/checkpoint/'
             save_checkpoint(net, epoch, LEARNING_RATE, NET_TYPE, checkpoint_path)             

if __name__ == "__main__":
    main() 
