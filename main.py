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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm

#from dataset_vol_graph_early16 import VolleyballDataset
from dataset_vol_graph_early import VolleyballDataset
#from dataset_vol_graph_mid5 import VolleyballDataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from models.cifar.vgg19 import vgg

import train_test_model as T
import models.cifar.gan as GAN
import video_dataset_processing as vdpro
import time
import util

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
parser.add_argument('--group-pretrain', default='/mnt/data8tb/junwen/checkpoints/group_gcn/volleyball/vgg19_64_4096fixed_gcn2layer_lr0.01_pre71_mid5_lstm2/model_best.pth.tar', type=str, metavar='PATH', help='path to load group rec(default: 79%)')
parser.add_argument('--model-dir', default='/home/junwen/opengit/player-classification-video/checkpoints/volleyball/vgg19_64_mid5_preImageNet_flip_drop/model_best.pth.tar', type=str, metavar='PATH', help='path to load player classification(default: 71%)')
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

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.Resize(64),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
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


    trainloader = DataLoader(VolleyballDataset(split='train', transforms=transform_train), \
    batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testloader = DataLoader(VolleyballDataset(split='test', transforms=transform_test), \
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    x_dim = 2048
    h_dim = 1024
    z_dim = 1024
    E_model = GAN.Encoder(x_dim, h_dim, z_dim)
    G1 = GAN.Decoder1(z_dim, h_dim, x_dim)
    G2 = GAN.Decoder2(z_dim, h_dim, x_dim)
    D_model = GAN.Discriminator(x_dim, h_dim, num_classes)
    E_solver = optim.Adam(E_model.parameters(), lr=state['lr'])
    G1_solver = optim.Adam(G1.parameters(), lr=state['lr'])
    G2_solver = optim.Adam(G2.parameters(), lr=state['lr'])
    D_solver = optim.Adam(D_model.parameters(), lr=state['lr'])

    E_model.to(device)
    G1.to(device)
    G2.to(device)
    D_model.to(device)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = vgg(num_classes=num_classes, net=args.arch, model_dir=args.model_dir)

    model.create_architecture()
    model.to(device)

    #model resume
    model.load_state_dict(torch.load(args.group_pretrain)['state_dict'])

    #for k,v in model.state_dict().items():
    #    print(k)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_file = open("train_out.txt", "w")
    test_file = open("test_out.txt", "w")

    util.print_model(E_model, G1, G2, D_model)

    # adjust learning rate
    scheduler_E = optim.lr_scheduler.StepLR(E_solver, 100, 0.1)
    scheduler_G1 = optim.lr_scheduler.StepLR(G1_solver, 100, 0.1)
    scheduler_G2 = optim.lr_scheduler.StepLR(G2_solver, 100, 0.1)
    scheduler_D = optim.lr_scheduler.StepLR(D_solver, 100, 0.1)


    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

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
        E_model.load_state_dict(checkpoint['E_state_dict'])
        G2.load_state_dict(checkpoint['G1_state_dict'])
        G1.load_state_dict(checkpoint['G2_state_dict'])
        D_model.load_state_dict(checkpoint['D_state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    epoch = 1
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, level_accuracy = T.test(epoch, device, testloader, model, E_model, G1, G2, D_model, test_file, num_classes)
        test_acc = torch.mean(level_accuracy)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        scheduler_E.step()
        scheduler_G1.step()
        scheduler_G2.step()
        scheduler_D.step()

        #adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss = T.train(epoch, device, trainloader, model, E_model, E_solver, G1, G1_solver, G2, G2_solver, D_model, D_solver, train_file, num_classes)
        test_loss, level_accuracy = T.test(epoch, device, testloader, model, E_model, G1, G2, D_model, test_file, num_classes)

        # save model
        test_acc = torch.mean(level_accuracy)
        #logger.append([state['lr'], test_acc])
        print(train_loss, test_loss, test_acc)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'E_state_dict': E_model.state_dict(),
                'G1_state_dict': G1.state_dict(),
                'G2_state_dict': G2.state_dict(),
                'D_state_dict': D_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                #'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)



def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
