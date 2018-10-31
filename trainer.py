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
from models.resnet import ResNet, BasicBlock, Bottleneck
# ArgParser
import argparse
from time import gmtime, strftime


def return_cifar10_training(dataset_path, download = False, mini_batch_size = 64):

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(size=[32,32], padding=4)]
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                            download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True, num_workers=8)
    return trainloader

def return_cifar10_testing(dataset_path, download = False, mini_batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                           download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=8)
    return testloader         

def save_checkpoint(net, epoch, net_name, root_path):
    time_string = strftime("%d%m%Y_%H%M%S", gmtime())
    state = {'net': net.state_dict(), 'epoch': epoch, 'time': time_string}
    if not os.path.isdir('checkpoint'): os.mkdir('checkpoint')
    print('[INFO] Saving checkpoint: ' + 'ckpt_'+str(time_string)+'_ep'+str(epoch)+'.t7')
    if(root_path.endswith('/')): root_path = root_path[:-1]
    torch.save(state, root_path + '/checkpoint/ckpt_'+str(net_name)+str(time_string)+'_ep'+str(epoch)+'.t7')


def main():
    ##Parser
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')
    parser.add_argument('--arch', default='resnet34', type=str, help='architecture type: resnet18/152')
    parser.add_argument('--epochs', default=200, type=int, help='total training epochs')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size')
    parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
    parser.add_argument('--root', default='./', type=str, help='root path')
    parser.add_argument('--data', default='./', type=str, help='data path')
    args = parser.parse_args()
    DEVICE_ID = args.gpu
    LEARNING_RATE = args.lr
    TOT_EPOCHS = args.epochs
    MINI_BATCH_SIZE = args.batch
    ROOT_PATH = args.root
    DATASET_PATH = args.data
    NET_TYPE = args.arch
    print("[INFO] Root path: " + str(ROOT_PATH))
    print("[INFO] Dataset path: " + str(DATASET_PATH))
    print("[INFO] Total epochs: " + str(TOT_EPOCHS))
    print("[INFO] Mini-batch size: " + str(MINI_BATCH_SIZE))
    print("[INFO] Learning rate: " + str(LEARNING_RATE))
    ##Generate net
    if(NET_TYPE == 'resnet18'):
        net = ResNet(BasicBlock, [2,2,2,2])
    elif(NET_TYPE == 'resnet34'):
        net = ResNet(BasicBlock, [3, 4, 6, 3])
    elif(NET_TYPE == 'resnet50'):
        net = ResNet(Bottleneck, [3,4,6,3])
    elif(NET_TYPE == 'resnet101'):
        net = ResNet(Bottleneck, [3,4,23,3])
    elif(NET_TYPE == 'resnet152'):
        net = ResNet(Bottleneck, [3,8,36,3])
    else:
        raise ValueError('[ERROR] the architecture type ' + str(NET_TYPE) + ' is unknown.') 
    print("[INFO] Architecture: " + str(NET_TYPE))      
    torch.cuda.set_device(DEVICE_ID)
    device = torch.device('cuda:'+str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    print("[INFO] Torch is using device: " + str(torch.cuda.current_device()))    
    net.to(device)
    if args.resume:
        print('[INFO] Resuming from checkpoint: ' + str(args.resume))
        checkpoint = torch.load(str(args.resume))
        net.load_state_dict(checkpoint['net'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    writer = SummaryWriter(log_dir=ROOT_PATH)
    trainloader = return_cifar10_training(dataset_path=DATASET_PATH, 
                                          download = False, mini_batch_size = MINI_BATCH_SIZE)
    global_step = 0
    for epoch in range(0, TOT_EPOCHS):  #loop over the dataset multiple times
        loss_list = list()
        for i, data in enumerate(trainloader, 0):     
            # Zero the parameter gradients
            optimizer.zero_grad()      
            # Get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward
            outputs = net(inputs)   
            #TODO Estimate regularizer          
            # Estimate loss
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
            global_step += 1 #increasing the global step     

        print('[%d, %5d] lr: %.5f; loss: %.5f; accuracy: %.5f' 
               %(epoch, global_step, LEARNING_RATE, np.mean(loss_list), accuracy))
        if(epoch==60 or epoch==120 or epoch==160):
             save_checkpoint(net, epoch, NET_TYPE, ROOT_PATH)
             LEARNING_RATE = LEARNING_RATE * 0.2
             for g in optimizer.param_groups:
                 g['lr'] = LEARNING_RATE
             

if __name__ == "__main__":
    main() 
