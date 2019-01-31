import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from PIL import Image, ImageDraw
from torch.autograd import Variable
from models.ssd import SSD300, SSDBoxCoder

import os

print('Loading model..')
net = SSD300(num_classes=2)

checkpoint = torch.load('./example/ssd+/checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
net.to(device)
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

net.eval()
box_coder = SSDBoxCoder(net)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4140, 0.4265, 0.4172), (0.2646, 0.2683, 0.2751))
])

file_path = '/home/zwj/project/data/caltech/ImageSets/Main/test.txt'
data_dir = '/home/zwj/project/data/caltech/JPEGImages/'
result_dir = './results/'
with open(file_path) as f:
    lines = f.readlines()

al = len(lines)
file_content = {}
for num, line in enumerate(lines):
    print('{}/{}'.format(num, al))

    img_org = Image.open(data_dir+line[:-1]+'.jpg')
    ow = oh = 300
    img = img_org.resize((ow, oh))

    x = transform(img)
    x = Variable(x.cuda(), volatile=True)
    loc_preds, cls_preds = net(x.unsqueeze(0))

    boxes, labels, scores = box_coder.decode(
        loc_preds.cpu().data.squeeze(), 
        F.softmax(cls_preds.squeeze(), dim=1).cpu().data, 
        score_thresh=0.1, nms_thresh=0.45)
    print('______________')

    _, idx = scores.sort(descending=True)

    scale = torch.Tensor([img_org.size[0], img_org.size[1],
                          img_org.size[0], img_org.size[1]])

    boxes = boxes * scale / 300

    # dir
    dirs = line.split('_')[:3]
    if not dirs[0] in file_content:
    # if not file_content.has_key(dirs[0]):
        file_content[dirs[0]] = {}
    if not dirs[1] in file_content[dirs[0]]:
    # if not file_content[dirs[0]].has_key(dirs[1]):
        file_content[dirs[0]][dirs[1]] = ""
    # file_content[dirs[0]][dirs[1]]

    for i in range(min(10, len(idx))):
        xmin = int(boxes[idx[i]][0].item())
        ymin = int(boxes[idx[i]][1].item())
        xmax = int(boxes[idx[i]][2].item())
        ymax = int(boxes[idx[i]][3].item())
        score = scores[idx[i]].item()

        file_content[dirs[0]][dirs[1]] += '{},{},{},{},{},{}\n'.format(int(dirs[2][1:])+1, xmin, ymin, xmax-xmin, ymax-ymin, score)

for dir1 in file_content:
    if not os.path.exists(os.path.join(result_dir, dir1)):
        os.makedirs(os.path.join(result_dir, dir1))
    for dir2 in file_content[dir1]:
        f = open(os.path.join(result_dir, dir1, dir2)+'.txt', "w")
        f.write(file_content[dir1][dir2])
        f.close()
