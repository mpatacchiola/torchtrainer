
#Massimiliano Patacchiola 2018
#This is used to train only additional connections of the Adartss algorithm
#placed on top of a pre-trained resnet34 network.

import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np
import os
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

download = False
mini_batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='../../datasets/cifar10', train=True,
                                        download=download, transform=transform)
dataset_size = len(trainset)
print(dataset_size)
portion_size = dataset_size // 2                       

indices = np.arange(0, dataset_size)
np.random.shuffle(indices)
train_idx, valid_idx = indices[0:portion_size], indices[portion_size:dataset_size]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
                                     
trainloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, sampler=train_sampler, num_workers=4)
valloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, sampler=valid_sampler, num_workers=4)

#sys.exit()

testset = torchvision.datasets.CIFAR10(root='../../datasets/cifar10', train=False,
                                       download=download, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet, BasicBlock, AdartsBlock #we do not import the models of torchvision, because they do not work on CIFAR-10
block_type="AdartsBlock"
if(block_type=="AdartsBlock"):
    net = ResNet(AdartsBlock, [3, 4, 6, 3]) #RESNET-34
    print("Block type ..... AdartsBlock")
elif(block_type=="BasicBlock"):
    net = ResNet(BasicBlock, [3, 4, 6, 3]) #RESNET-34
    print("Block type ..... BasicBlock")
net_name = 'Adarts_resnet34'
#for param in net.parameters():
#    param.requires_grad = False #Disable gradient estimation

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

########################################################################
# ArgParser
import argparse
import os
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
args = parser.parse_args()
#Hyperparam
start_epoch = 0
tot_epochs = 5
lambda_start = 0.0
lambda_stop = 0.5
lambda_steps = 60000 * 100 #dataset size * epochs
lambda_array = np.linspace(start=lambda_start, stop=lambda_stop, num=lambda_steps, endpoint=True)
#Check parameters
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(str(args.resume))
    net.load_state_dict(checkpoint['net'])
    #start_epoch = checkpoint['epoch']
    print('start_epoch: ' + str(start_epoch))


########################################################################
# 2. Enable or disable the pruning
def enable_pruning():
    #print('Pruning enabled!')
    for param in net.parameters():
        param.requires_grad = False
    for name, module in net.named_children():           
        ##[3, 4, 6, 3]
        if('layer1' in name): tot_blocks=3
        elif('layer2' in name): tot_blocks=4
        elif('layer3' in name): tot_blocks=6
        elif('layer4' in name): tot_blocks=3
        else: tot_blocks=0
        for i in range(tot_blocks):
            module[i].conv0_1.weight.requires_grad = True
            module[i].conv0_2.weight.requires_grad = True
            module[i].hidden0.weight.requires_grad = True
            module[i].hidden0.bias.requires_grad = True
            module[i].linear0.weight.requires_grad = True
            module[i].linear0.bias.requires_grad = True
            #torch.nn.init.zeros_(module[i].linear0.bias)           
       
def disable_pruning():
    large_int = 100.0
    #print('Pruning disabled!')
    for param in net.parameters():
        param.requires_grad = True
    for name, module in net.named_children():
        ##[3, 4, 6, 3]
        if('layer1' in name): tot_blocks=3
        elif('layer2' in name): tot_blocks=4
        elif('layer3' in name): tot_blocks=6
        elif('layer4' in name): tot_blocks=3
        else: tot_blocks=0
        for i in range(tot_blocks):
            module[i].conv0_1.weight.requires_grad = False
            module[i].conv0_2.weight.requires_grad = False        
            module[i].hidden0.weight.requires_grad = False
            module[i].hidden0.bias.requires_grad = False
            module[i].linear0.weight.requires_grad = False
            module[i].linear0.bias.requires_grad = False
            #torch.nn.init.constant_(module[i].linear0.bias, val=large_int) 


########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

# print images
#imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.3f %%' % (100 * correct / total))

########################################################################
# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %.3f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    
    
print("Done!")       
