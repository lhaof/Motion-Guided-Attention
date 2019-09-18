import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import re 

class DAVIS(data.Dataset):
    def __init__(self, dataset='train', transform=None, return_size=False):
        self.return_size = return_size
        self.flow_visual_dir = './dataset/flow_visual/DAVIS' 
        self.data_dir = './dataset/DAVIS'
        self.images_path = list()
        self.labels_path = list()
        if dataset == 'train':
            imgset_path = self.data_dir + '/ImageSets/480p/train.txt'
        else:
            imgset_path = self.data_dir + '/ImageSets/480p/val.txt'
        imgset_file = open(imgset_path)        
        # in a line: img gt
        for line in imgset_file:
            img_path = line.strip("\n").split(" ")[0]
            gt_path = line.strip("\n").split(" ")[1]
            self.images_path.append(img_path)
            self.labels_path.append(gt_path)
        self.transform = transform

    def __getitem__(self, item):
        flow_path = self.flow_visual_dir + self.images_path[item].replace('/JPEGImages', '')
        image_path = self.data_dir + self.images_path[item]
        label_path = self.data_dir + self.labels_path[item]
        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(label_path), ('{} does not exist'.format(label_path))
        assert os.path.exists(flow_path), ('{} does not exist'.format(flow_path))

        flow = Image.open(flow_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')
        label = np.array(Image.open(label_path))
        if label.max() > 0:
            label = label / label.max()
        label = Image.fromarray(label.astype(np.uint8))

        w, h = image.size
        size = (h,w)

        sample = {'image': image, 'label': label, 'flow': flow}

        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)
        
        pos_list = [i.start() for i in re.finditer('/', label_path)]
        label_name = label_path[pos_list[-2]+1:]
        sample['label_name'] = label_name

        return sample

    def __len__(self):
        return len(self.images_path)

if __name__ == '__main__':
    print("hello world")