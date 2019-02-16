'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
import torchvision.models as models
#from mypath import Path
from torch.autograd import Variable

from models.cifar.gcn import _gcn

class vgg(_gcn):

    def __init__(self, num_classes=1000, net='vgg19', model_dir=''):

        _gcn.__init__(self, num_classes)
        self.net = net
        self.group_cls = num_classes
        self.model_dir = model_dir

    def _init_modules(self):

        if self.net == 'vgg19':
            model = models.vgg19()
            model.classifier = nn.Sequential(
                nn.Linear(3072, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0),
                nn.Linear(4096, 9),
            )
   
            state_dict = torch.load(self.model_dir)
             
            model.load_state_dict({k:v for k,v in state_dict['state_dict'].items() if k in model.state_dict()})


        elif self.net == 'vgg19_bn':
            model = models.vgg19_bn()
            print("Loading pretrained weights from %s" % Path.model_dir(self.net))
            self._load_pretrained_weights(self.net)

        #self.player_cls = nn.Sequential(*list(model.classifier._modules.values())[-2:])
        model.classifier = nn.Sequential(*list(model.classifier._modules.values())[:-1])
        self.base_model = model



