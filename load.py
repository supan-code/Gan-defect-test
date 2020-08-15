import torch.utils.data as data
from PIL import Image
import os


class DefectData(data.Dataset):
    def __init__(self, root, transform=None, train=True):
        super(DefectData, self).__init__()
        self.root = root
        self.transform = transform
        if train:
            namepath = os.path.join(root, 'train.txt')
        else:
            print('--evaluate--')
            namepath = os.path.join(root, 'val.txt')
        f = open(namepath, 'r+')
        self.names = [name.split() for name in f.readlines()]
        self.ids = len(self.names)

    def __getitem__(self, x):
        name = self.names[x][0]
        image_path = os.path.join(self.root, name+'.bmp')
        img = Image.open(image_path)
        if self.transform is None:
            return img
        else:
            image, defect, target = self.transform(img)
            return [image, defect, target]

    def __len__(self):
        return self.ids
