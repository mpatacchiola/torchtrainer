'''Mixture of Residuals (MoR)

pytorch implementation with ResNet as base
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, g, round_g=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        g = torch.reshape(g, (g.size(0), 1, 1, 1))
        if(round_g):
            out = out * torch.round(g) #mpatacchiola
        else:
            out = out * g #mpatacchiola
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, g, round_g=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        g = torch.reshape(g, (g.size(0), 1, 1, 1))
        if(round_g):
            out = out * torch.round(g) #mpatacchiola
        else:
            out = out * g #mpatacchiola
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, round_g=False):
        super(ResNet, self).__init__()
        self.global_id = 0 #mpatacchiola
        self.round_g = round_g
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        #Define the gate function
        #input: 32x32
        self.convg1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.bng1 = nn.BatchNorm2d(64)
        #conv1: 16x16
        self.convg2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.bng2 = nn.BatchNorm2d(128)
        #conv2: 8x8
        self.convg3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True)
        self.bng3 = nn.BatchNorm2d(256)
        #conv3: 4x4
        self.convg4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True)
        self.bng4 = nn.BatchNorm2d(512)
        #conv4: 2x2
        self.convg5 = nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.bng5 = nn.BatchNorm2d(128)
        #conv5: 1x1
        self.hiddeng = nn.Linear(in_features=128, out_features=128, bias=True)
        self.linearg = nn.Linear(in_features=128, out_features=sum(num_blocks), bias=True)
        #torch.nn.init.constant_(self.linearg.bias, val=1.0) #init the output to 'carry' behaviour
        #torch.nn.init.uniform_(self.linearg.bias, a=-1.5, b=1.5)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #Gate output
        outg = F.relu(self.bng1(self.convg1(x)))
        outg = F.relu(self.bng2(self.convg2(outg)))
        outg = F.relu(self.bng3(self.convg3(outg)))
        outg = F.relu(self.bng4(self.convg4(outg)))
        outg = F.relu(self.bng5(self.convg5(outg)))
        outg = F.relu(self.hiddeng(torch.squeeze(outg)))
        outg = self.linearg(outg)
        self.gate_output = torch.sigmoid(outg)

        #ResNet output
        out = F.relu(self.bn1(self.conv1(x)))
        block_counter = 0
        self.histograms_list = list()
        for block in self.layer1:
            g = self.gate_output[:,block_counter] #mpatacchiola
            self.histograms_list.append(g.squeeze())       
            out = block(out, g, self.round_g)
            block_counter += 1  
        for block in self.layer2:
            g = self.gate_output[:,block_counter] #mpatacchiola 
            self.histograms_list.append(g.squeeze())         
            out = block(out, g, self.round_g)
            block_counter += 1  
        for block in self.layer3:
            g = self.gate_output[:,block_counter] #mpatacchiola
            self.histograms_list.append(g.squeeze())      
            out = block(out, g, self.round_g)
            block_counter += 1  
        for block in self.layer4:
            g = self.gate_output[:,block_counter] #mpatacchiola
            self.histograms_list.append(g.squeeze())          
            out = block(out, g, self.round_g)
            block_counter += 1  
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
    def return_regularizer(self):
        #L2 penalty
        target = -0.5
        return torch.mean(torch.pow(torch.add(self.gate_output,target), 2))
        
    def return_histograms(self):
        return self.histograms_list

    def return_performance(self):
        used_blocks = torch.sum(torch.round(self.gate_output), dim=1)
        mean_used_blocks = torch.mean(used_blocks).cpu().numpy()
        std_used_blocks = torch.std(used_blocks).cpu().numpy()
        return [mean_used_blocks, std_used_blocks]
        
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

