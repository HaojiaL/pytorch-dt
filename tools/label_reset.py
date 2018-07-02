file_path = '/media/disk4/share/DataSet/baidu_dataset/fusai/datasets/train.txt'
with open(file_path) as f:
    lines = f.readlines()

dict = {}
for line in lines:
    file_name = line[:44]
    label = line[44:-1]
    if file_name in dict:
        dict[file_name] += label
    else:
        dict[file_name] = label

with open(file_path[:-4] + '_reset.txt', 'w') as f:
    for k, v in dict.items():
        f.write(k)
        f.write(v)
        f.write('\n')