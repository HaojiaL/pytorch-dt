import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import os
import sys
# os.chdir('/home/hector/project/proj-pytorch/pytorch-dt/')
sys.path.insert(0, '/home/hector/project/proj-pytorch/pytorch-dt/')


from PIL import Image, ImageDraw
from torch.autograd import Variable
from models.ssd import SSD300, SSDBoxCoder

print('Loading model..')
net = SSD300(num_classes=61)

checkpoint = torch.load('/home/hector/project/proj-pytorch/pytorch-dt/example/ssd/checkpoint/ckpt1.pth')
net.load_state_dict(checkpoint['net'])
net.cuda()
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
net.eval()

print('Loading image..')
img_org = Image.open('/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/Image/f2deb48f8c5494eea4de0dde24f5e0fe98257e7c.jpg')
ow = oh = 300
img = img_org.resize((ow, oh))

print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
x = transform(img)
x = Variable(x.cuda(), volatile=True)
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