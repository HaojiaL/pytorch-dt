import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.backends.cudnn as cudnn

import os
import sys
# os.chdir('/home/hector/project/proj-pytorch/pytorch-dt/')
sys.path.insert(0, '/home/hector/project/proj-pytorch/pytorch-dt/')


from PIL import Image, ImageDraw
from torch.autograd import Variable
from datasets.transforms import resize
from datasets.detdataset import TrainDataset
from models.ssd import SSD300, SSDBoxCoder

print('Loading model..')
net = SSD300(num_classes=61)

checkpoint = torch.load('/home/hector/project/proj-pytorch/pytorch-dt/example/ssd/checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.cuda()
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
net.eval()

print('Preparing dataset..')
img_size = 300
def transform(img, boxes, labels):
    img1, _ = resize(img, boxes, size=(img_size,img_size))
    img1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img1)
    return img, img1, boxes, labels

dataset = TrainDataset(root='/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/Image/',
                      list_file='/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/train_reset_val.txt',
                      transform=None)
box_coder = SSDBoxCoder(net)

for i, (inputs, box_targets, label_targets) in enumerate(iter(dataset)):
    img_org, inputs, box_targets, label_targets = transform(inputs, box_targets, label_targets)
    img_org_gt = img_org.copy()
    x = Variable(inputs.cuda(), volatile=True)
    loc_preds, cls_preds = net(x.unsqueeze(0))

    print('Decoding..')

    boxes, labels, scores = box_coder.decode(
        loc_preds.cpu().data.squeeze(), F.softmax(cls_preds.squeeze(), dim=1).cpu().data, score_thresh=0.01, nms_thresh=0.45)
    # print(labels)
    # print(scores)

    draw = ImageDraw.Draw(img_org)
    draw_gt = ImageDraw.Draw(img_org_gt)
    _, idx = scores.sort(descending=True)

    scale = torch.Tensor([img_org.size[0], img_org.size[1],
                          img_org.size[0], img_org.size[1]])

    boxes = boxes * scale / 300
    # print(boxes)

    # take gt_label for baidu race
    gt_label = labels[idx[0]].item()
    for i in range(min(8, len(idx))):
        print(labels[idx[i]].item(), scores[idx[i]].item())
        if labels[idx[i]].item() == gt_label and scores[idx[i]].item() > 0.1:
            draw.rectangle(list(boxes[idx[i]]), outline='red')
    for i in box_targets:
        draw_gt.rectangle(list(i), outline='green')

    # for i, box in enumerate(boxes):
    #     if scores[i] > 0.5:
    #         draw.rectangle(list(box), outline='red')
    # out = utils.make_grid([img_org_gt, inputs])
    # out.show()
    img_org_gt.show()
    img_org.show()
    input("waiting...")