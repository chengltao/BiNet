import torch
import torch.nn as nn
import torch.nn.functional as F





# short_cut structure
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(3,3,3),
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BiNet(nn.Module):
    def __init__(self):
        super(BiNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(5,1,1),
                               stride=(2,1,1), padding=(0,0,0), bias=False)
        self.bn1 = nn.BatchNorm3d(64)

        self.layer1 = self._make_layer(64)
        self.layer2 = self._make_layer(128)
        self.layer3 = self._make_layer(256)
        self.layer4 = self._make_layer(512)
        self.linear = nn.Linear(4608, 15)
    def _make_layer(self,  planes):
        strides = [2, 1]
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out1 = F.avg_pool3d(out, kernel_size=(10, 5, 5))
        out1 = out1.view(out1.size(0), -1)
        out = self.layer2(out)
        out2 = F.avg_pool3d(out, kernel_size=(8, 3, 3))
        out2 = out2.view(out2.size(0), -1)
        out = self.layer3(out)
        out3 = F.avg_pool3d(out, kernel_size=(5, 3, 3))
        out3 = out3.view(out3.size(0), -1)
        out = self.layer4(out)
        out = F.avg_pool3d(out, 2)
        out = out.view(out.size(0), -1)
        #feature fusion
        out5 = torch.cat([out1, out2], dim=1)
        out6 = torch.cat([out5, out3], dim=1)
        out7 = torch.cat([out6, out], dim=1)

        out = self.linear(out7)

        return F.log_softmax(out, dim=1)



