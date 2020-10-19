'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

#add this import first to fix the issue: ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found
import matplotlib.pyplot as plt

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from math import cos, pi
import tune

from celeba import CelebA
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

from loss import FocalLoss

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='/home/MSAI/ch0001ka/face-attribute-prediction/', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--loss', type=str, default='ce',
                    help='loss, ce | focalloss')

# Optimization options
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=10,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--turning-point', type=int, default=100,
                    help='epoch number from linear to exponential decay mode')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--cardinality', type=int, default=32, help='ResNeXt model cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNeXt model base width (number of channels in each group).')
parser.add_argument('--groups', type=int, default=3, help='ShuffleNet model groups')
# Miscs
parser.add_argument('--manual-seed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--e_file', type=str, default='test_40_att_list.txt',
                    help='file to evaluate')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('-t', '--tune', dest='tune', action='store_true',
                    help='enable ray tune')

best_prec1 = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def main(c: dict):
    global args, best_prec1

    # if using tune, config will overwrite the args
    # TODO: write in a generic way
    # configable_list = ["lr", "arch", "lr_decay", "step", "loss"]
    # for c_name in configable_list:
    #     if c_name in c:
    #         args.

    if "lr" in c:
        args.lr = c["lr"]
    if "arch" in c:
        args.arch = c["arch"]
    if "lr_decay" in c:
        args.lr_decay = c["lr_decay"]
    if "step" in c:
        args.step = c["step"]
    if "loss" in c:
        args.loss = c["loss"]


    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # Random seed
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    # elif args.arch.startswith('resnext'):
    #     model = models.__dict__[args.arch](
    #                 baseWidth=args.base_width,
    #                 cardinality=args.cardinality,
    #             )
    elif args.arch.startswith('shufflenet'):
        model = models.__dict__[args.arch](
                    groups=args.groups
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.to(device) #.cuda()
        else:
            model = torch.nn.DataParallel(model).to(device) #.cuda()
    else:
        model.to(device)#.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss().to(device)#.cuda()
    elif args.loss == "focalloss":
        criterion = FocalLoss(device)
    else:
        print("ERROR: ------ Unkown loss !!! ------")

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = 'CelebA-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Comment to avoid the cudnn cloud pickle error
    # cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = CelebA(
        args.data,
        'train_40_att_list.txt',
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        CelebA(args.data, 'val_40_att_list.txt', transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        CelebA(args.data, args.e_file, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        val_loss, prec1, top1_avg_att = validate(test_loader, model, criterion)
        print(top1_avg_att)
        return

    # visualization
    if not args.tune:
        writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, prec1, top1_avg_att = validate(val_loader, model, criterion)

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        acc_att_dict = {}
        for i, acc in enumerate(top1_avg_att):
            acc_att_dict[f"att-{i}"] = acc
        # tensorboardX
        if not args.tune:
            writer.add_scalar('learning rate', lr, epoch + 1)
            writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
            writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)
            writer.add_scalars('att-accuracy', acc_att_dict, epoch + 1)
        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)

        if args.tune:
            tune.report(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, prec1=prec1, att_acc=acc_att_dict, lr=lr)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    if not args.tune:
        writer.close()

    print('Best accuracy:')
    print(best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    bar = Bar('Processing', max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(40)]
    top1 = [AverageMeter() for _ in range(40)]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        # measure accuracy and record loss
        loss = []
        prec1 = []
        for j in range(len(output)):
            loss.append(criterion(output[j], target[:, j]))
            prec1.append(accuracy(output[j], target[:, j], topk=(1,)))

            losses[j].update(loss[j].item(), input.size(0))
            top1[j].update(prec1[j][0].item(), input.size(0))
        losses_avg = [losses[k].avg for k in range(len(losses))]
        top1_avg = [top1[k].avg for k in range(len(top1))]
        loss_avg = sum(losses_avg) / len(losses_avg)
        prec1_avg = sum(top1_avg) / len(top1_avg)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_sum = sum(loss)
        loss_sum.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=loss_avg,
                    top1=prec1_avg,
                    )
        bar.next()
    bar.finish()
    return (loss_avg, prec1_avg)


def validate(val_loader, model, criterion):
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(40)]
    top1 = [AverageMeter() for _ in range(40)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.to(device) #.cuda(non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss
            loss = []
            prec1 = []
            for j in range(len(output)):
                loss.append(criterion(output[j], target[:, j]))
                prec1.append(accuracy(output[j], target[:, j], topk=(1,)))

                losses[j].update(loss[j].item(), input.size(0))
                top1[j].update(prec1[j][0].item(), input.size(0))
            losses_avg = [losses[k].avg for k in range(len(losses))]
            top1_avg = [top1[k].avg for k in range(len(top1))]
            loss_avg = sum(losses_avg) / len(losses_avg)
            prec1_avg = sum(top1_avg) / len(top1_avg)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=loss_avg,
                    top1=prec1_avg,
                    )
        bar.next()
    bar.finish()

    # print(top1_avg)
    return (loss_avg, prec1_avg, top1_avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    """Sets the learning rate to the initial LR decayed by 10 following schedule"""
    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** (epoch // args.step))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * epoch / args.epochs)) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - epoch / args.epochs)
    elif args.lr_decay == 'linear2exp':
        if epoch < args.turning_point + 1:
            # learning rate decay as 95% at the turning point (1 / 95% = 1.0526)
            lr = args.lr * (1 - epoch / int(args.turning_point * 1.0526))
        else:
            lr *= args.gamma
    elif args.lr_decay == 'schedule':
        if epoch in args.schedule:
            lr *= args.gamma
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    if args.tune:
        import ray
        from ray import tune

        # import hyperopt as hp
        # from ray.tune.suggest.hyperopt import HyperOptSearch

        ray.shutdown()
        ray.init(local_mode=True)

        config = {
            # "lr": tune.grid_search([0.1, 0.5]),
            "lr": tune.grid_search([0.005]),
            # "arch": tune.grid_search(["resnext50_32x4d", "resnet50"]),
            "arch": tune.grid_search(["resnext50_32x4d"]),
            # "step": tune.grid_search([5]),
            "lr_decay": tune.grid_search(["linear"]),
            "loss": tune.grid_search(["focalloss"]),
            # "remark": tune.grid_search(["resume_linear"])
        }
        # hyperopt = HyperOptSearch(metric="valid_loss", mode="min", space=config)

        tune.run(main,
                 config=config,
                 name="run_report",
                 local_dir="./ray_results",
                 # search_alg=hyperopt,
                 # num_samples=2,
                 # stop={"training_iteration": 5},
                 resources_per_trial={"gpu": 1}
                 )
    else:
        main({})
