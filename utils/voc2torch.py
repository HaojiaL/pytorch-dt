import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# Some config
data_dir = "/home/zwj/project/data/caltech/"
annotations_dir = "./Annotations/"
imagesets_dir = "./ImageSets/Main/"

transform_list = ["test.txt", "trainval.txt"]

CLASSES = ('person',)# from 0 to clsnum-1
class_to_ind = dict(zip(CLASSES, range(len(CLASSES))))

# Start transform
pts = ['xmin', 'ymin', 'xmax', 'ymax']
for context in transform_list:
    transformed_f = open(data_dir + imagesets_dir + context[:-4] + '_torch.txt', 'w')
    with open(data_dir + imagesets_dir + context, 'r') as list_f:
        for line in list_f:
            anno = [line.strip() + '.jpg']
            anno_file = data_dir + annotations_dir + line.strip() + '.xml'
            root = ET.parse(anno_file).getroot()
            for obj in root.iter('object'):
                name = obj.find('name').text.lower().strip()
                bbox = obj.find('bndbox')
                for pt in pts:
                    anno.append(bbox.find(pt).text)
                anno.append(str(class_to_ind[name]))
            transformed_f.write(' '.join(anno)+'\n')
            print(anno)
    transformed_f.close()
