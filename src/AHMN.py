import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F


def conv_down_layers(inp, oup, ):

    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_layers(inp, oup, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class AHMN(nn.Module):
    # (expansion, out_planes, num_blocks, stride)

    cfg = [(1,  16, 1, 1),
           (6,  32, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  64, 3, 2),
           (6,  128, 4, 2)]

    def __init__(self, in_channel,num_action = 2):
        super(AHMN, self).__init__()
        self.seen = 0
        self.channel = 64
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        # self.transform = transform
        self.conv1 = nn.Conv2d(in_channel, self.channel, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel)

        self.layers1 = conv_down_layers(self.channel,self.channel,)

        self.layers2 = conv_down_layers(self.channel,self.channel,)

        self.layers3 = conv_down_layers(self.channel,self.channel,)

        self.layers4 = conv_down_layers(self.channel,self.channel,)
        # Policy net

        self.layers4_p = conv_layers(self.channel,self.channel,)

        self.layers3_p = conv_layers(self.channel,self.channel,)

        self.layers2_p = conv_layers(self.channel,self.channel,)
        # Value net

        self.layers4_v = conv_layers(self.channel,self.channel,)

        self.layers3_v = conv_layers(self.channel,self.channel,)

        self.layers2_v = conv_layers(self.channel,self.channel,)


        self.output_layer_p = nn.Conv2d(self.channel, 1, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(1)
        self.output_layer_v = nn.Conv2d(self.channel, 1, kernel_size=1)

        self.bn3 = nn.BatchNorm2d(1)


    def forward(self, x):

        self.features = []
        x = F.relu6(self.bn1(self.conv1(x)))
        # print(x.size())

        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        # print(x4.size())

        # Policy net
        x4_p = self.layers4_p(x4)
        x5_p = self.layers3_p(x4_p)
        x6_p = self.layers2_p(x5_p)

        policy = torch.sigmoid((self.output_layer_p(x6_p)))
        # print(x10_p.size())
        # policy = torch.softmax(x10_p,dim=1)

        # Value net

        x4_v = self.layers4_v(x4)
        x5_v = self.layers3_v(x4_v)
        x6_v = self.layers2_v(x5_v)

        value = F.relu6(self.bn3(self.output_layer_v(x6_v)))
        # if self.training is True:
        return policy, value
        # print('')
        # return  map3




