import torch
from torchvision import datasets, transforms

# Config
data_dir = '/home/zwj/project/data/caltech/JPEGImages/'

data_transform = transforms.Compose([
    transforms.ToTensor()
    ])

dataset = datasets.ImageFolder(root=data_dir, transform = data_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
mean = torch.zeros(3)
std = torch.zeros(3)
print('==> Computing mean and std..')
for inputs, targets in dataloader:
    for i in range(3):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()
mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)