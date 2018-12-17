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
import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import models.cifar as models
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm
# from torchvision import transforms

#from dataset_vol_graph import VolleyballDataset
from dataset_vol_graph_mid5 import VolleyballDataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from models.cifar.vgg19 import vgg

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='volleyball', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
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
parser.add_argument('--model-dir', default='/home/junwen/opengit/player-classification-video/checkpoints/volleyball/vgg19_64_mid5_preImageNet_flip_drop/model_best.pth.tar',
                    type=str, metavar='PATH', help='path to load player classification(default: 71%)')
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
parser.add_argument('--gpu-id', default='cuda:0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")

state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
# assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

person_label_index = {0:'waiting', 1:'setting', 2:'digging', 3:'falling', 4:'spiking', \
        5:'blocking', 6:'jumping', 7:'moving', 8:'standing'}

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == "volleyball":
        num_classes = 8
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    testloader = DataLoader(VolleyballDataset(split='test', transforms=transform_test), \
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = vgg(num_classes=num_classes, net=args.arch, model_dir=args.model_dir)

    model.create_architecture()

    model.to(device)
    #model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'volleyball-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        #print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    
def visualize_player(name, bboxs, player_outputs):
    player_outputs = player_outputs.cpu().numpy()[0]
    frames = sorted(os.listdir(name[0]))
    with open(bboxs[0], 'r') as f:
        det_lines = f.readlines()
    det_lines = [item.strip().split('\t') for item in det_lines]

    frame_count = len(frames)-32
    for idx in range(frame_count):
        i = idx+16
        img = cv2.imread(os.path.join(name[0],frames[i]))
        for j in range(player_outputs.shape[1]):
            buffer_bbox = [int(x) for x in det_lines[j][i].split(' ')]
            img = cv2.rectangle(img, (buffer_bbox[0],buffer_bbox[1]), (buffer_bbox[0]+buffer_bbox[2], buffer_bbox[1]+buffer_bbox[3]), (0,225,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, person_label_index[player_outputs[idx][j]], (buffer_bbox[0], buffer_bbox[1]), font, 2, (0,225,225), 2, cv2.LINE_AA)
        cv2.imshow(frames[i],img)
        cv2.waitKey(0)
    return


def test(testloader, model, criterion, epoch, use_cuda):
    label_index = {0:'r_set', 1:'r_spike', 2:'r-pass', 3:'r_winpoint', 4:'l_winpoint', \
        5:'l-pass', 6:'l-spike', 7:'l_set'}

    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (name, bboxs, inputs, targets, dists) in enumerate(testloader):
        # measure data loading
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets, dists = inputs.to(device), targets.to(device), dists.to(device)

        with torch.no_grad():
            outputs, player_outputs = model(inputs, dists)
            p_max_value, p_max_index = torch.max(player_outputs, 3)
       
        loss = criterion(outputs, targets)

        max_value, max_index = torch.max(outputs, 1)
        if label_index[targets.cpu().numpy()[0]] != label_index[max_index.cpu().numpy()[0]]:
            print('name', name)
            print('max_index', label_index[max_index.cpu().numpy()[0]])
            print('targets', label_index[targets.cpu().numpy()[0]])
            visualize_player(name, bboxs, p_max_index)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

if __name__ == '__main__':
    main()
