'''VGG for CIFAR10. FC layers are removed.
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

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    # 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._initialize_weights()

    def forward(self, x):
        dout = self.fc1(x)
        # print("self.fc1", self.fc1(x)[:,:6])
        # print("dout", dout)
        #
        # assert 1==0
        dout = self.fc2(dout)
        return dout

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, pretrain=False, train=False):
        super(VGG, self).__init__()
        self.Train = train
        self.features = features
        self.pclassifier = nn.Linear(3072, 9)
        self.fc6 = nn.Linear(12288, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.conv1da = nn.Conv1d(in_channels=3072, out_channels=512*2, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.conv1db = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.convLinear = nn.Conv1d(in_channels=512, out_channels=num_classes, kernel_size=1, \
        padding=0, stride=1, dilation=1, bias=False)
        self.gclassifier = nn.Linear(512, num_classes)
        self.mlp1 = MLP(input_dim=6144, hidden_dim=2048, output_dim=512)
        self.mlp2 = MLP(input_dim=1024, hidden_dim=512, output_dim=256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(256, num_classes)
        self._initialize_weights()
        if pretrain:
            self._load_pretrained_weights()

    def forward(self, x, dist):
        [N, T, M, C, H, W] = x.shape
        base_out = self.features(x.view(-1, C, H, W)).view(N*T, M, -1)
        #with torch.no_grad():
        #    p_out = base_out.view(N*T*M, -1)
        #    p_out = self.pclassifier(p_out)
        #    probs = nn.Softmax(dim=1)(p_out)
        #    preds = torch.max(probs, 1)[1]
        #    # print("preds", preds.view(N,T,M))
        #    base_out = Variable(base_out)

        node1 = self.conv1da(base_out.permute(0,2,1))
        node1 = node1.view(N*T, 2, 512, 12)
        node1 = torch.einsum('nkcv,nkvw->ncw', (node1, dist.view(N*T, 2, 12, 12).float()))
        # node1 = torch.bmm(dist.view(-1,12,12).float(), node1.permute(0,2,1))
        node1 = F.relu(node1)

        # nodelinear = self.convLinear(node1.permute(0,2,1))
        pooled_feat = self.pool(node1).squeeze(2)
        pooled_feat = self.gclassifier(pooled_feat)
        if self.Train:
           pooled_feat = F.dropout(pooled_feat, p=0.5)
        group_out = self.avg_pool(pooled_feat.view(N, -1, T))
        
        return group_out.squeeze(2)

        # node = torch.zeros(N*T, M, 512).cuda()
        # for i in range(M):
        #     for j in range(i):
        #         node[:,i,:] += self.mlp1(torch.cat((base_out[:,i,:], base_out[:,j,:]), dim=1))
        #     for j in range(i+1,M):
        #         node[:,i,:] += self.mlp1(torch.cat((base_out[:,i,:], base_out[:,j,:]), dim=1))
        #
        # node2 = torch.zeros(N*T, M, 256).cuda()
        # for i in range(M):
        #     for j in range(i):
        #         node2[:,i,:] += self.mlp2(torch.cat((node[:,i,:], node[:,j,:]), dim=1))
        #     for j in range(i+1,M):
        #         node2[:,i,:] += self.mlp2(torch.cat((node[:,i,:], node[:,j,:]), dim=1))
        #
        # pooled_feat = self.pool(node2.permute(0,2,1)).squeeze(2)
        # pooled_feat = self.linear(pooled_feat)
        # group_out = self.avg_pool(pooled_feat.view(N, T, -1).permute(0,2,1)).squeeze(2)
        # return group_out

    def _load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "module.features.0.weight": "features.0.weight",
                        "module.features.0.bias": "features.0.bias",
                        # Conv2
                        "module.features.2.weight": "features.2.weight",
                        "module.features.2.bias": "features.2.bias",
                        # Conv3a
                        "module.features.5.weight": "features.5.weight",
                        "module.features.5.bias": "features.5.bias",
                        # Conv3b
                        "module.features.7.weight": "features.7.weight",
                        "module.features.7.bias": "features.7.bias",
                        # Conv4a
                        "module.features.10.weight": "features.10.weight",
                        "module.features.10.bias": "features.10.bias",
                        # Conv4b
                        "module.features.12.weight": "features.12.weight",
                        "module.features.12.bias": "features.12.bias",
                        # Conv5a
                        "module.features.14.weight": "features.14.weight",
                        "module.features.14.bias": "features.14.bias",
                         # Conv5b
                        "module.features.16.weight": "features.16.weight",
                        "module.features.16.bias": "features.16.bias",

                        "module.features.19.weight": "features.19.weight",
                        "module.features.19.bias": "features.19.bias",

                        "module.features.21.weight": "features.21.weight",
                        "module.features.21.bias": "features.21.bias",

                        "module.features.23.weight": "features.23.weight",
                        "module.features.23.bias": "features.23.bias",

                        "module.features.25.weight": "features.25.weight",
                        "module.features.25.bias": "features.25.bias",

                        "module.features.28.weight": "features.28.weight",
                        "module.features.28.bias": "features.28.bias",

                        "module.features.30.weight": "features.30.weight",
                        "module.features.30.bias": "features.30.bias",

                        "module.features.32.weight": "features.32.weight",
                        "module.features.32.bias": "features.32.bias",

                        "module.features.34.weight": "features.34.weight",
                        "module.features.34.bias": "features.34.bias",
                        # fc6
                        "module.fc6.weight": "fc6.weight",
                        "module.fc6.bias": "fc6.bias",
                        # fc7
                        "module.fc7.weight": "fc7.weight",
                        "module.fc7.bias": "fc7.bias",
                        #classifier
                        "module.classifier.weight":"pclassifier.weight",
                        "module.classifier.bias":"pclassifier.bias"
                        }
        net = "vgg19bn"
        p_dict = torch.load(Path.model_dir(net))
        print("p_dict", [item for item in p_dict["state_dict"]])
        s_dict = self.state_dict()
        for item in s_dict:
             print("sdict", item)
        for name in p_dict['state_dict']:
            #if name not in corresp_name:
            #    print("not", name)
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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model
