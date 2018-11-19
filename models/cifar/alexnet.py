'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
from mypath import Path
from torch.autograd import Variable

__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=10, pretrain=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #self.classifier = nn.Linear(256, num_classes)
        self.conv1da = nn.Conv1d(in_channels=1536, out_channels=512*2, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.conv1db = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.convLinear = nn.Conv1d(in_channels=512, out_channels=num_classes, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.gclassifier = nn.Linear(512, num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self._initialize_weights()
        if pretrain:
            self._load_pretrained_weights()

    def forward(self, x, dist):
        [N, T, M, C, H, W] = x.shape
        base_out = self.features(x.view(-1, C, H, W)).view(N*T, M, -1)

        node1 = self.conv1da(base_out.permute(0,2,1))
        node1 = node1.view(N*T, 2, 512, 12)
        node1 = torch.einsum('nkcv,nkvw->ncw', (node1, dist.view(N*T, 2, 12, 12).float()))

        node1 = F.relu(node1)
        pooled_feat = self.pool(node1).squeeze(2)
        pooled_feat = self.gclassifier(pooled_feat)
        group_out = self.avg_pool(pooled_feat.view(N, -1, T))
        
        return group_out.squeeze(2)


    def _load_pretrained_weights(self):
        """Initialiaze network."""

        net = 'alexnet' 
        p_dict = torch.load(Path.model_dir(net))
        print("p_dict", [item for item in p_dict["state_dict"]])
        s_dict = self.state_dict()
        for item in s_dict:
             print("sdict", item)
        for name in p_dict['state_dict']:
            #    continue
            if name in s_dict:
              s_dict[name] = p_dict['state_dict'][name]
        self.load_state_dict(s_dict)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)

def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
