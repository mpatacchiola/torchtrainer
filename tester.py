
#Massimiliano Patacchiola 2018
#This is used to train only additional connections of the Adartss algorithm
#placed on top of a pre-trained resnet34 network.

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

def return_cifar10_testing(dataset_path, download = False, mini_batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                           download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=8)
    return testloader  

def return_cifar100_testing(dataset_path, download=False, mini_batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR100(root=dataset_path, train=False,
                                            download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=8)
    return testloader
        
def main():
    ##Parser
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')
    parser.add_argument('--id', default='', type=str, help='experiment ID')
    parser.add_argument('--arch', default='resnet34', type=str, help='architecture type: resnet18/152')
    parser.add_argument('--batch', default=1000, type=int, help='mini-batch size')
    parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
    parser.add_argument('--root', default='./', type=str, help='root path')
    parser.add_argument('--data', default='./', type=str, help='data path')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset type: cifar10, cifar100')
    args = parser.parse_args()
    DEVICE_ID = args.gpu
    MINI_BATCH_SIZE = args.batch
    ROOT_PATH = args.root
    DATASET_PATH = args.data
    NET_TYPE = args.arch
    ID = args.id
    print("[INFO] ID: " + str(ID))
    print("[INFO] Root path: " + str(ROOT_PATH))
    print("[INFO] Dataset path: " + str(DATASET_PATH))
    print("[INFO] Mini-batch size: " + str(MINI_BATCH_SIZE))
    #torch.cuda.set_device(DEVICE_ID)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(DEVICE_ID)
    print('[INFO] Is CUDA available: ' + str(torch.cuda.is_available()))
    print('[INFO] TOT available devices: ' + str(torch.cuda.device_count()))
    print('[INFO] Setting device: ' + str(DEVICE_ID))
    #device = torch.device('cuda:'+str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        net = ResNet(BasicBlock, [2,2,2,2], num_classes=TOT_CLASSES, round_g=True)
    elif(NET_TYPE == 'mor34'):
        from models.mor import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=TOT_CLASSES, round_g=True)
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
        net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=TOT_CLASSES, round_g=True)
        gate_matrix = np.zeros(16, dtype=np.float32)
        gate_dict = {}
    else:
        raise ValueError('[ERROR] the architecture type ' + str(NET_TYPE) + ' is unknown.') 
    print("[INFO] Architecture: " + str(NET_TYPE))          
    net.to(device)
    if args.resume:
        print('[INFO] Resuming from checkpoint: ' + str(args.resume))
        checkpoint = torch.load(str(args.resume))
        net.load_state_dict(checkpoint['net'])
    else:
        raise ValueError('[ERROR] You must use --resume to load a checkpoint in order to test the model!')        

    #Load the test set
    if(args.dataset == "cifar10"):
        testloader = return_cifar10_testing(DATASET_PATH, download=False, mini_batch_size=1000)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif(args.dataset == "cifar100"):
        testloader = return_cifar100_testing(DATASET_PATH, download=True, mini_batch_size=1000)
        #classes = (
        # "beaver","dolphin","otter","seal","whale","aquarium fish","flatfish","ray","shark",
        # "trout", "orchids", "poppies", "roses", "sunflowers", "tulips", "bottles", "bowls", "cans", "cups","plates",
        # "apples","mushrooms", "oranges","pears","sweet peppers",
        # "clock","computer keyboard","lamp","telephone","television","bed","chair","couch","table","wardrobe",
        # "bee","beetle", "butterfly", "caterpillar", "cockroach",
        # "bear","leopard","lion", "tiger","wolf",
        # "bridge","castle","house","road","skyscraper","cloud","forest","mountain","plain","sea",
        # "camel","cattle","chimpanzee","elephant","kangaroo","fox","porcupine","possum","raccoon",
        # "skunk","crab","lobster","snail","spider","worm","baby","boy", "girl", "man", "woman",
        # "crocodile", "dinosaur", "lizard", "snake", "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel", 
        # "maple", "oak", "palm", "pine", "willow",
        # "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "rocket", "streetcar", "tank", "tractor")
         
        classes = (
         'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
         'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
         'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
         'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
         'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
         'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
         'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
         'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
         'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
         'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
         'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
         'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
         'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
         'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
         
    else:
        raise ValueError('[ERROR] the dataset ' + args.dataset + ' is unknown.') 
            
    #Check
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
       if('rmor' in NET_TYPE):
           outputs = net(images, gumbel_tau=0.01) #1e-10)
       else:
           outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    #Whole dataset accuracy
    correct = 0
    total = 0
    performance = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            if('rmor' in NET_TYPE):
                outputs = net(images, gumbel_tau=0.01) #1e-10) #setting tau but not using it
                gate_array = np.stack(net.gate_output_list, axis=1) #size[1000, 16]
                gate_matrix += np.sum(gate_array, axis=0)
                for row in range(gate_array.shape[0]):
                    key = tuple(gate_array[row].tolist())
                    gate_dict[key] = gate_dict.get(key, 0) + 1
            else:
                outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if(hasattr(net, 'return_performance')):
                performance = net.return_performance()
    print('Accuracy of the network on the 10000 test images: %.5f %%' % (100 * correct / total))
    if(hasattr(net, 'return_performance')):
        print('Performance: ' + str(performance))
        print('Block usage: ' + str(gate_matrix / np.sum(gate_matrix)))
        print('Tot configurations: ' + str(len(gate_dict)))
        key_list =list()
        for key in gate_dict:
            key_list.append(key)
        key_matrix = np.stack(key_list, axis=0) #size[138, 16]
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(key_matrix)
        plt.colorbar()
        plt.savefig(ROOT_PATH + '/block_activations.png', dpi=300)
        plt.imshow(gate_array[0:30, :])
        plt.savefig(ROOT_PATH + '/last_batch_block_activations.png', dpi=300)
        #torchvision.utils.save_image(images[0:10, :, :, :], ROOT_PATH + '/last_batch_images.png')
        for i in range(30):
           _, predicted = torch.max(outputs, 1)
           prediction = classes[predicted[i].item()]
           label = classes[labels[i].item()] 
           torchvision.utils.save_image(images[i, :, :, :], ROOT_PATH + str(i) + '_' + str(label) + '_' + str(prediction) + '.png')
    #Per-Class accuracy
    class_correct = list(0. for i in range(TOT_CLASSES))
    class_total = list(0. for i in range(TOT_CLASSES))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            if('rmor' in NET_TYPE):
                outputs = net(images, gumbel_tau=0.01) #1e-10) #setting tau but not using it        
            else:
                outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(TOT_CLASSES):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(TOT_CLASSES):
        print('Accuracy of %15s : %.3f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        
        
if __name__ == "__main__":
    main() 
