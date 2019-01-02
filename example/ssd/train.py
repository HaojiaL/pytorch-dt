from __future__ import print_function

import os
import random
import argparse

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

# from torch.autograd import Variable

from models.ssd import SSD300, SSDBoxCoder

from loss.mutibox_loss import MutiBoxLoss
from datasets.detdataset import TrainDataset
from datasets.transforms import resize, random_flip, random_paste, random_crop, random_distort

import torch.nn.init as init

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='./basemodel/vgg16.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='./example/ssd+/checkpoint/ckpt.pth', type=str, help='checkpoint path')
args = parser.parse_args()

print('==> Building model..')
net = SSD300(num_classes=2)

net.apply(weights_init)

# fix load net
d = torch.load(args.model)
d_proc = {'.'.join(k.split('.')[1:]): v for k, v in d.items() if 'classifier' not in k}
net.extractor.features.layers.load_state_dict(d_proc, strict=False)

# net.load_state_dict(torch.load(args.model))

best_loss = float('inf')
start_epoch = 0
if args.resume:
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
        transforms.Normalize((0.4140, 0.4265, 0.4172), (0.2646, 0.2683, 0.2751))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


trainset = TrainDataset(root='/home/zwj/project/data/caltech/JPEGImages/',
                       list_file='/home/zwj/project/data/caltech/ImageSets/Main/trainval_torch.txt',
                       transform=transform_train)


def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4140, 0.4265, 0.4172), (0.2646, 0.2683, 0.2751))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


valset = TrainDataset(root='/home/zwj/project/data/caltech/JPEGImages/',
                      list_file='/home/zwj/project/data/caltech/ImageSets/Main/test_torch.txt',
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=20, shuffle=False, num_workers=8)

# net.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
net.to(device)

criterion = MutiBoxLoss(num_classes=2)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        loc_targets = loc_targets.to(device)
        cls_targets = cls_targets.to(device)
        # inputs = Variable(inputs.cuda())
        # loc_targets = Variable(loc_targets.cuda())
        # cls_targets = Variable(cls_targets.cuda())
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
        inputs = inputs.to(device)
        loc_targets = loc_targets.to(device)
        cls_targets = cls_targets.to(device)
        # inputs = Variable(inputs.cuda(), volatile=True)
        # loc_targets = Variable(loc_targets.cuda())
        # cls_targets = Variable(cls_targets.cuda())

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
            # 'net': net.module.state_dict(),
            'net': net.state_dict(),
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
