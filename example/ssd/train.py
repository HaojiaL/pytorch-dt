from __future__ import print_function

import os
import random
import argparse

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from torch.autograd import Variable

from models.ssd import SSD300, SSDBoxCoder

from loss.mutibox_loss import MutiBoxLoss
from datasets.detdataset import TrainDataset
from datasets.transforms import resize, random_flip, random_paste, random_crop, random_distort

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='/home/hector/project/proj-pytorch/pytorch-dt/example/ssd/model/vgg16_reducedfc.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='/home/hector/project/proj-pytorch/pytorch-dt/example/ssd/checkpoint/ckpt.pth', type=str, help='checkpoint path')
args = parser.parse_args()

print('==> Building model..')
net = SSD300(num_classes=61)

# fix load net
d = torch.load(args.model)
d_proc = {'.'.join(k.split('.')[1:]): v for k, v in d.items() if 'classifier' not in k}
net.extractor.features.layers.load_state_dict(d_proc, strict=False)

# net.load_state_dict(torch.load(args.model))

best_loss = float('inf')
start_epoch = 0
if args.resume or True:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

print('==> Preparing dataset..')
box_coder = SSDBoxCoder(net)
img_size = 300


def transform_train(img, boxes, labels):
    img = random_distort(img)
    # if random.random() < 0.5:
    #     img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size,img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=False, num_workers=8)

net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

criterion = MutiBoxLoss(num_classes=61)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        # inputs = Variable(inputs)
        # loc_targets = Variable(loc_targets)
        # cls_targets = Variable(cls_targets)

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data.item(), train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))


def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(valloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data.item()
        print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data.item(), test_loss/(batch_idx+1), batch_idx+1, len(valloader)))

    # Save checkpoint
    global best_loss
    print(best_loss)
    test_loss /= len(valloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(args.checkpoint)):
            os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, args.checkpoint)
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+2000):
    train(epoch)
    test(epoch)
