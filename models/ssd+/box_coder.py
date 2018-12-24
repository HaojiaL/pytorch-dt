import math
import torch
import itertools

from utils.box import box_iou, box_nms, change_box_order


class SSDBoxCoder:
    def __init__(self, ssd_model):
        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.default_boxes = self._get_default_boxes()

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                # add dense for cy
                # for offset_y in [0, 0.5]:
                for offset_y in [0, ]:
                    cx = (w + 0.5) * self.steps[i]
                    cy = (h + offset_y) * self.steps[i]

                    s = self.box_sizes[i]
                    boxes.append((cx, cy, s, s))

                    s = math.sqrt(self.box_sizes[i] * self.box_sizes[i + 1])
                    boxes.append((cx, cy, s, s))

                    s = self.box_sizes[i]
                    for ar in self.aspect_ratios[i]:
                        boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                        boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes, labels):
        def argmax(x):
            v, i = x.max(0)
            # j = v.max(0)[1][0]
            j = v.max(0)[1].item()
            return (i[j], j) # 第j个obj 以及第j个obj的最大anchors坐标

        default_boxes = self.default_boxes  # xywh
        default_boxes = change_box_order(default_boxes, 'xywh2xyxy')

        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]
        index = torch.LongTensor(len(default_boxes)).fill_(-1) # 与anchor匹配的boxes坐标
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i, j] < 1e-6:
                break
            index[i] = j #设置与anchor匹配度的boxes坐标
            masked_ious[i, :] = 0 # 设置设置过得roi为0，表示已经搜索过次roi， 对应于while里的条件
            masked_ious[:, j] = 0

        mask = (index < 0) & (ious.max(1)[0] >= 0.5) # 没有在第一次进行匹配到的 并且 对于每一个anchor与任何boxes的roi大于0.5的
        if mask.any(): # 如果存在
            # index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]
            index[mask] = ious[mask].max(1)[1] #设置匹配 【1】表示使用坐标位置 对应于58行

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = change_box_order(default_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        loc_xy = (boxes[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:] / variances[0]
        loc_wh = torch.log(boxes[:, 2:] / default_boxes[:, 2:]) / variances[1]
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index < 0] = 0
        return loc_targets, cls_targets # cls>0 的是正样本 其他为0 ； loc在cls=0的地方是无效值

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        variances = (0.1, 0.2)
        xy = loc_preds[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        wh = torch.exp(loc_preds[:, 2:] * variances[1]) * self.default_boxes[:, 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes - 1):
            score = cls_preds[:, i + 1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze(1)]
            score = score[mask]

            keep = box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores