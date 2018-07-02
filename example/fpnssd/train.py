'''FPNSSD512 train on KITTI.'''
from __future__ import print_function

import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from PIL import Image

from models.fpnssd import FPNSSD512
from models.fpnssd import FPNSSDBoxCoder

from loss.mutibox_loss import MutiBoxLoss
from datasets.detdataset import TrainDataset
from datasets.transforms import resize, random_flip, random_paste, random_crop, random_distort


parser = argparse.ArgumentParser(description='PyTorch FPNSSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='/home/hector/project/proj-pytorch/pytorch-dt/example/fpnssd/model/resnet50.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='/home/hector/project/proj-pytorch/pytorch-dt/example/fpnssd/checkpoint/ckpt.pth', type=str, help='checkpoint path')
args = parser.parse_args()

# Data
print('==> Preparing dataset..')
img_size = 512
box_coder = FPNSSDBoxCoder()
def transform_train(img, boxes, labels):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size,img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

trainset = TrainDataset(root='/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/Image/',
                       list_file='/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/train_reset_train.txt',
                       transform=transform_train)

def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

valset = TrainDataset(root='/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/Image/',
                      list_file='/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/train_reset_val.txt',
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=False, num_workers=8)

# Model
print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = FPNSSD512(num_classes=61).to(device)
net.fpn.load_state_dict(torch.load(args.model), strict=False)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume or True:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

criterion = MutiBoxLoss(num_classes=61)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        loc_targets = loc_targets.to(device)
        cls_targets = cls_targets.to(device)

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.item(), train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))

def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(valloader):
            inputs = inputs.to(device)
            loc_targets = loc_targets.to(device)
            cls_targets = cls_targets.to(device)

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss.item()
            print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss.item(), test_loss/(batch_idx+1), batch_idx+1, len(valloader)))

    # Save checkpoint
    global best_loss
    test_loss /= len(valloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, args.checkpoint)
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
