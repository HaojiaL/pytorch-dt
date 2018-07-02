import random

def split_trainval(file_path, val_prop=0.2):
    with open(file_path) as f:
        lines = f.readlines()
        random.shuffle(lines)
        val_lines = lines[:int(val_prop * len(lines))]
        train_lines = lines[int(val_prop * len(lines)):]

    with open(file_path[:-4]+'_train.txt', 'w') as f:
        f.writelines(train_lines)
    with open(file_path[:-4]+'_val.txt', 'w') as f:
        f.writelines(val_lines)

split_trainval('/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/train_reset.txt')