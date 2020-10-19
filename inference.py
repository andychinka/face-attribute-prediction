import torch
import torchvision.transforms as transforms

import pandas as pd
import numpy as np

import argparse
import os

import models
from celeba import CelebATestFromDir

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='/home/MSAI/ch0001ka/advcv/testset', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--outputfile', default='predictions.txt', type=str, metavar='PATH')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference():
    global args
    args = parser.parse_args()

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.to(device) #.cuda()
    else:
        model = torch.nn.DataParallel(model).to(device) #.cuda()

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_loader = torch.utils.data.DataLoader(
        CelebATestFromDir(args.data, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=32, shuffle=False, pin_memory=True)

    model.eval()

    preds_df = pd.DataFrame()
    preds_att = torch.LongTensor().to(device) # finally with size: (# of data, 40)

    with torch.no_grad():

        for i, (input, filename) in enumerate(data_loader):
            bs = input.size(0)
            output = model(input)

            batch_preds = torch.zeros(bs, len(output)).long().to(device)
            neg_labels = -torch.ones(bs).long().to(device)

            for j in range(len(output)):
                _, index = torch.max(output[j], dim=1)

                pred = torch.where(index == 0, neg_labels, index) #convert 0 to -1
                batch_preds[:, j] = pred

            preds_att = torch.cat((preds_att, batch_preds))
            preds_df = pd.concat([preds_df, pd.Series(filename)],
                                 ignore_index=True)

    preds_att_df = pd.DataFrame(preds_att.cpu().numpy())
    preds_df = pd.concat([preds_df, preds_att_df], axis=1)
    preds_df.to_csv(args.outputfile, sep=" ", header=False, index=False)


if __name__ == "__main__":
    inference()