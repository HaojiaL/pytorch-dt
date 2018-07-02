import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import sys
# os.chdir('/home/hector/project/proj-pytorch/pytorch-dt/')
sys.path.insert(0, '/home/hector/project/proj-pytorch/pytorch-dt/')

from torch.autograd import Variable
from datasets.transforms import resize
from datasets.detdataset import TrainDataset
from utils.voc_eval import voc_eval
from models.fpnssd import FPNSSD512, FPNSSDBoxCoder

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (60000, rlimit[1]))

print('Loading model..')
net = FPNSSD512(num_classes=61)
net.cuda()
net = torch.nn.DataParallel(net)
checkpoint = torch.load('/home/hector/project/proj-pytorch/pytorch-dt/example/fpnssd/checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])

net.eval()

print('Preparing dataset..')
img_size = 512
def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    return img, boxes, labels

dataset = TrainDataset(root='/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/Image/',
                      list_file='/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/train_reset_val.txt',
                      transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
box_coder = FPNSSDBoxCoder()

pred_boxes = []
pred_labels = []
pred_scores = []
gt_boxes = []
gt_labels = []

# with open('torchcv/datasets/voc/voc07_test_difficult.txt') as f:
#     gt_difficults = []
#     for line in f.readlines():
#         line = line.strip().split()
#         d = [int(x) for x in line[1:]]
#         gt_difficults.append(d)

def eval(net, dataset):
    for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
        print('%d/%d' % (i, len(dataloader)))
        gt_boxes.append(box_targets.squeeze(0))
        gt_labels.append(label_targets.squeeze(0))

        loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=0.01)

        # for baidu race
        _, idx = score_preds.sort(descending=True)
        gt_label = label_preds[idx[0]].item()
        gt_score = score_preds[idx[0]].item()

        box_preds = box_preds[idx[:1000]]
        label_preds = label_preds[idx[:1000]]
        score_preds = score_preds[idx[:1000]]
        ss = label_preds.eq(gt_label)
        score_preds = score_preds / gt_score
        xx = score_preds.gt(0.01)
        box_preds = box_preds[ss | xx]
        label_preds = label_preds[ss | xx]
        score_preds = score_preds[ss | xx]
        # score_preds = score_preds[ss | xx] / gt_score


        # origin solver
        pred_boxes.append(box_preds)
        pred_labels.append(label_preds)
        pred_scores.append(score_preds)

    print (voc_eval(
        pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels, None,
        iou_thresh=0.5, use_07_metric=True))


eval(net, dataset)
