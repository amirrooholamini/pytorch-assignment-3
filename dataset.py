import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

transform = torchvision.transforms.Compose([
     torchvision.transforms.ToPILImage(mode=None),
     torchvision.transforms.Resize((224,224)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0),(1)),
])

class Dataset(Dataset):
    def __init__(self, data, targets, transform=transform):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)


def splitData(dataset, val_split=0.2, train=True):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    if train:
        return datasets['train']
    return datasets['val']