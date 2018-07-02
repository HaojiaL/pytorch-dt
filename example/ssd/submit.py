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

checkpoint = torch.load('/home/hector/project/proj-pytorch/pytorch-dt/example/ssd/checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.cuda()
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
net.eval()
box_coder = SSDBoxCoder(net)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

file_path = '/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/test.txt'
with open(file_path) as f:
    lines = f.readlines()

al = len(lines)
with open(file_path[:-4]+'_submit.txt', 'w') as f:
    for num, line in enumerate(lines):
        print('{}/{}'.format(num, al))


        img_org = Image.open('/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/Image/'+line[:-1])
        ow = oh = 300
        img = img_org.resize((ow, oh))

        x = transform(img)
        x = Variable(x.cuda(), volatile=True)
        loc_preds, cls_preds = net(x.unsqueeze(0))

        boxes, labels, scores = box_coder.decode(
            loc_preds.cpu().data.squeeze(), F.softmax(cls_preds.squeeze(), dim=1).cpu().data, score_thresh=0.01, nms_thresh=0.45)
        print('______________')

        _, idx = scores.sort(descending=True)

        scale = torch.Tensor([img_org.size[0], img_org.size[1],
                              img_org.size[0], img_org.size[1]])

        boxes = boxes * scale / 300

        # take gt_label for baidu race
        gt_label = labels[idx[0]].item()
        for i in range(min(8, len(idx))):
            if labels[idx[i]].item() == gt_label and scores[idx[i]].item() > 0.1:
                f.write(line[:-1])
                f.write(' {}'.format(labels[idx[i]].item()+1))
                f.write(' {}'.format(scores[idx[i]].item()))
                f.write(' {}'.format(int(boxes[idx[i]][0].item())))
                f.write(' {}'.format(int(boxes[idx[i]][1].item())))
                f.write(' {}'.format(int(boxes[idx[i]][2].item())))
                f.write(' {}'.format(int(boxes[idx[i]][3].item())))
                f.write('\n')
