import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import sys
# os.chdir('/home/hector/project/proj-pytorch/pytorch-dt/')
# sys.path.insert(0, '/home/hector/project/proj-pytorch/pytorch-dt/')


from PIL import Image, ImageDraw
from torch.autograd import Variable
from models.ssd import SSD300, SSDBoxCoder

print('Loading model..')
net = SSD300(num_classes=2)

checkpoint = torch.load('./example/ssd+/checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cudnn.benchmark = True
net.to(device)

net.eval()

print('Loading image..')
img_org = Image.open('~/project/data/caltech/JPEGImages/set04_V002_I00059_usatrain.jpg')
ow = oh = 300
img = img_org.resize((ow, oh))

print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4140, 0.4265, 0.4172), (0.2646, 0.2683, 0.2751))
])
x = transform(img)
x = Variable(x.to(device), volatile=True)
loc_preds, cls_preds = net(x.unsqueeze(0))

print('Decoding..')
box_coder = SSDBoxCoder(net)
boxes, labels, scores = box_coder.decode(
    loc_preds.cpu().data.squeeze(), F.softmax(cls_preds.squeeze(), dim=1).cpu().data, score_thresh=0.01, nms_thresh=0.35)
print(labels)
print(scores)

draw = ImageDraw.Draw(img_org)
_, idx = scores.sort(descending=True)

scale = torch.Tensor([img_org.size[0], img_org.size[1],
                      img_org.size[0], img_org.size[1]])

boxes = boxes * scale / 300
print(boxes)

# take gt_label for baidu race
gt_label = labels[idx[0]].item()
for i in range(8):
    print(labels[idx[i]].item(), scores[idx[i]].item())
    if labels[idx[i]].item() == gt_label and scores[idx[i]].item() > 0.1:
        draw.rectangle(list(boxes[idx[i]]), outline='red')

# for i, box in enumerate(boxes):
#     if scores[i] > 0.5:
#         draw.rectangle(list(box), outline='red')
img_org.show()