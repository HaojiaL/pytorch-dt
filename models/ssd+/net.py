import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layers = self._make_layers()

    def forward(self, x):
        y = self.layers(x)
        return y

    def _make_layers(self):
        '''VGG16 layers.'''
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)


class L2Norm(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        return scale * x


class VGG16Extractor300(nn.Module):
    def __init__(self):
        super(VGG16Extractor300, self).__init__()

        self.features = VGG16()
        self.norm4 = L2Norm(512, 20)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        # Top-down layers
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        hs = []
        h4_3 = self.features(x)
        h4_3 = self.norm4(h4_3)
        # hs.append(self.norm4(h4_3))  # conv4_3

        hp1 = F.max_pool2d(h4_3, kernel_size=2, stride=2, ceil_mode=True)

        h5_1 = F.relu(self.conv5_1(hp1))
        h5_2 = F.relu(self.conv5_2(h5_1))
        h5_3 = F.relu(self.conv5_3(h5_2))
        hp2 = F.max_pool2d(h5_3, kernel_size=3, stride=1, padding=1, ceil_mode=True)

        h6 = F.relu(self.conv6(hp2))
        h7 = F.relu(self.conv7(h6))
        # hs.append(h7)  # conv7

        h8_1 = F.relu(self.conv8_1(h7))
        h8_2 = F.relu(self.conv8_2(h8_1))
        # hs.append(h8_2)  # conv8_2

        h9_1 = F.relu(self.conv9_1(h8_2))
        h9_2 = F.relu(self.conv9_2(h9_1))
        # hs.append(h9_2)  # conv9_2

        h10_1 = F.relu(self.conv10_1(h9_2))
        h10_2 = F.relu(self.conv10_2(h10_1))
        # hs.append(h10_2)  # conv10_2

        h11_1 = F.relu(self.conv11_1(h10_2))
        h11_2 = F.relu(self.conv11_2(h11_1))
        # hs.append(h11_2)  # conv11_2

        # top_down
        p8_2 = self.toplayer(h8_2)
        p7 = self._upsample_add(p8_2, self.latlayer1(h7))
        p4_3 = self._upsample_add(p7, self.latlayer2(h4_3))

        # smooth for predict
        p7 = self.smooth1(p7)
        p4_3 = self.smooth2(p4_3)

        # append to hs
        hs.append(p4_3)
        hs.append(p7)
        hs.append(p8_2)
        hs.append(h9_2)
        hs.append(h10_2)
        hs.append(h11_2)

        return hs


class SSD300(nn.Module):
    steps = (8, 16, 32, 64, 100, 300)
    box_sizes = (30, 60, 111, 162, 213, 264, 315)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    # aspect_ratios = ((3, 4), (5, 4, 3), (5, 4, 3), (5, 4, 3), (4, 3), (3, 4))
    fm_sizes = (38, 19, 10, 5, 3, 1)

    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = (4, 6, 6, 6, 4, 4)
        # add dense
        # self.num_anchors = (8, 10, 10, 10, 8, 8)
        self.in_channels = (512, 1024, 512, 256, 256, 256)

        self.extractor = VGG16Extractor300()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i], self.num_anchors[i] * 4, kernel_size=(1, 5), padding=(0, 2))]
            self.cls_layers += [
                nn.Conv2d(self.in_channels[i], self.num_anchors[i] * self.num_classes, kernel_size=(1, 5), padding=(0, 2))]

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds