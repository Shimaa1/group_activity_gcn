'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
from torch.autograd import Variable

class _gcn(nn.Module):

    def __init__(self, num_classes):

        super(_gcn, self).__init__()
        self.conv1da = nn.Conv1d(in_channels=4096, out_channels=512*2, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.conv1db = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.convLinear = nn.Conv1d(in_channels=512, out_channels=num_classes, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.gclassifier = nn.Linear(512, num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(256, num_classes)
        self.lstm = nn.LSTM(512, 512, 1)

    def forward(self, x, dist):
        [N, T, M, C, H, W] = x.shape
        with torch.no_grad():
            base_out = self.base_model(x.view(-1, C, H, W)).view(N*T, M, -1)
        
        node1 = self.conv1da(base_out.permute(0,2,1))
        node1 = node1.view(N*T, 2, 512, 12)
        dista = dist[:,:,0,:,:,:]
        distb = dist[:,:,1,:,:,:]
        node1a = torch.einsum('nkcv,nkvw->ncw', (node1, dista.view(N*T, 2, 12, 12).float()))
        node1b = torch.einsum('nkcv,nkvw->ncw', (node1, distb.view(N*T, 2, 12, 12).float()))
        node1 = node1a+node1b
        node1 = F.relu(node1)
       
        pooled_feat = self.pool(node1).squeeze(2)
        video_feat, _ = self.lstm(pooled_feat.view(N, T, -1))      
        #video_feat, _ = self.lstm(pooled_feat.view(T, N, -1))      
        group_cls = self.gclassifier(video_feat[:,-1,:])
        #group_cls = self.gclassifier(video_feat[-1])
        #pooled_feat = self.gclassifier(pooled_feat)
        #group_out = self.avg_pool(pooled_feat.view(N, -1, T))
  
        return group_cls
        #return group_out.squeeze(2)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)


    def create_architecture(self):

        self._init_modules()
        self._initialize_weights()
