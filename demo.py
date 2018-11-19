'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from torch.autograd import Variable
from tqdm import tqdm
# from torchvision import transforms

from dataset_vol_graph import VolleyballDataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='volleyball', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--video_dir', default='' , type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--bbox_dir', default='' , type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
# assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():

    if args.dataset == "volleyball":
        num_classes = 8

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    model = models.__dict__[args.arch](num_classes=num_classes, pretrain=True)
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, \
    weight_decay=args.weight_decay)

    with open(os.path.join(args.video_dir, 'annotations.txt'),'r') as f:
        video_file = f.readlines()

    label_index = {0:'r_set', 1:'r_spike', 2:'r-pass', 3:'r_winpoint', 4:'l_winpoint', 5:'l-pass', \
    6:'l-spike', 7:'l_set'}
    video_index = {'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3, 'l_winpoint': 4, \
    'l-pass': 5, 'l-spike': 6, 'l_set': 7}
    group_index = np.zeros((8), dtype=np.int)

    for video in video_file:
        video_path = os.path.join(args.video_dir, video.split(" ")[0][:-4])
        bbox_path = os.path.join(args.bbox_dir, video.split(" ")[0][:-4], 'tracklets.txt')

        print("video_path", video_path)
        print("label is", video.split(" ")[1])

        if video_index[video.split(' ')[1]] != 2:
            continue

        buffer, dist = load_frames(video_path, bbox_path, transform_test)
        buffer = buffer[np.newaxis, :]

        inputs = torch.from_numpy(buffer[:,::2,:,:,:,:]).to(device)
        dist = torch.from_numpy(dist[::2,:,:,:]).to(device)

        with torch.no_grad():
            outputs, player_preds = model(inputs, dist)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs,1)[1]
            print("preds", label_index[preds.cpu().numpy()[0]])
            group_index[preds.cpu().numpy()[0]] += 1

    print("group_index", group_index)


            #draw_bbox
            # with open(bbox_path, 'r') as f:
            #     det_lines = f.readlines()
            # det_lines = [item.strip().split('\t') for item in det_lines]
            # imglist = os.listdir(video_path)
            # imglist = sorted(imglist)
            # for i in range(len(imglist)):
            #     im = cv2.imread(os.path.join(video_path, imglist[i]))
            #     for id in range(len(det_lines)):
            #         frame_show = im
            #         buffer_bbox = [int(x) for x in det_lines[id][i+1].split(' ')]
            #         cv2.rectangle(frame_show, (buffer_bbox[0], buffer_bbox[1]), \
            #         (buffer_bbox[0]+buffer_bbox[2], buffer_bbox[1]+buffer_bbox[3]),(0,0,255),2)
            #         cv2.putText(frame_show, str(player_preds[int(i/2)][id].cpu().numpy()),\
            #         (buffer_bbox[0], buffer_bbox[1]), cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
            #     cv2.imshow("id", frame_show)
            #     # cv2.waitKey(0)

    return


def load_frames(file_dir, bbox_dir, transform2):
    with open(bbox_dir, 'r') as f:
        det_lines = f.readlines()

    det_lines = [item.strip().split('\t') for item in det_lines]

    if len(det_lines) < 12:
        for i in range(12-len(det_lines)):
            det_lines.append(det_lines[-(i+1)])  #person number 12

    frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
    frame_count = len(frames)

    buffer = np.empty((frame_count, 12, 3, 109, 64), np.dtype('float32'))
    dist = np.zeros((frame_count, 2, 12, 12), np.dtype('float64'))

    for i, frame_name in enumerate(frames):
        frame = cv2.imread(frame_name)
        seq_x = np.zeros((12), dtype="float64")
        for j in range(len(det_lines)):
            buffer_bbox = [int(x) for x in det_lines[j][i+1].split(' ')]
            j_center_h = buffer_bbox[1]+buffer_bbox[3]/2
            seq_x[j] = j_center_h
        seq_index = np.argsort(seq_x)
        m = 0
        for item in seq_index:
            buffer_bbox = [int(x) for x in det_lines[item][i+1].split(' ')]
            person = frame[buffer_bbox[1]:buffer_bbox[1]+buffer_bbox[3], \
            buffer_bbox[0]:buffer_bbox[0]+buffer_bbox[2]]
            person = cv2.resize(person,(112, 192))
            person = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
            person = transform2(person)
            buffer[i][m][:] = person
            m += 1

        dist_index = np.argsort(dist[i], axis=1)

        for l in np.arange(12):
            dist[i][0][l][l] = 1/12
        for l in np.arange(12):
            dist[i][1][l][:l] = 1/12
            dist[i][1][l][l+1:] = 1/12

    return buffer, dist

if __name__ == '__main__':
    main()
