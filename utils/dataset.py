import os.path

import torch.utils.data as D

def normalize_image_per_channel_chw(image):
    normalized_image = np.zeros_like(image, dtype=np.float32)

    for c in range(image.shape[0]):
        channel = image[c, :, :]
        base_min = channel.min()
        base_max = channel.max()
        normalized_image[c, :, :] = (channel - base_min) / (base_max - base_min + 1)

    return normalized_image


class seg_dataset(D.Dataset):
    def __init__(self, image_paths, label_paths, mode, eval=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.eval = eval
        self.len = len(image_paths)

        self.val_transform = A.Compose([
            # A.RandomCrop(256,256),
            A.Normalize(),
            ToTensorV2()
        ])
        self.train_transform = A.Compose([
            # A.RandomCrop(256, 256),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])

    def __readTif__(self, fileName, xoff=0, yoff=0, data_width=0, data_height=0):
        dataset = gdal.Open(fileName)
        width = dataset.RasterXSize
        height = dataset.RasterYSize

        if (data_width == 0 and data_height == 0):
            data_width = width
            data_height = height
        data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)

        return data

    def __getitem__(self, index):
        image = self.__readTif__(self.image_paths[index])
        image = image.astype(np.float32)
        label = self.__readTif__(self.label_paths[index])
        label = label.astype(np.float32)
        label = label[None, :, :]

        if self.mode == "train":
            base_min = np.min(label)
            base_max = np.max(label)

            label = torch.tensor(label)

            label.clamp_(0, float('inf'))
            label = 2 * ((label - base_min) / (base_max - base_min + 10)) - 1
            image = torch.tensor(image)
            image = image/255

            return image, label,int(base_max),int(base_min)
        elif self.mode == "val":
            base_min = np.min(label)
            base_max = np.max(label)

            label = torch.tensor(label)

            label.clamp_(0, float('inf'))
            label = 2 * ((label - base_min) / (base_max - base_min + 10)) - 1
            image = torch.tensor(image)
            image = image / 255
            return image, label,int(base_max),int(base_min)
        elif self.mode == "test":
            if self.eval == True:

                label = torch.tensor(label)
                label.clamp_(0, float('inf'))
                image = torch.tensor(image)
                image = image / 255

                return image, label, os.path.split(self.label_paths[index])[1],self.image_paths[index]
            else:
                transformed_data = self.val_transform(image=image)
                image = transformed_data['image']
                return image, os.path.split(self.label_paths[index])[1]

    def __len__(self):
        return self.len



import os
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from osgeo import gdal



def get_dataloader(image_paths, label_paths, mode, batch_size, shuffle, num_workers, drop_last, eval=True):
    dataset = seg_dataset(image_paths, label_paths, mode, eval=eval)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    return dataloader


def build_dataloader(train_path, val_path, batch_size):
    train_loader = get_dataloader(train_path[0], train_path[1], "train", batch_size, shuffle=True, num_workers=2,
                                  drop_last=True)

    valid_loader = get_dataloader(val_path[0], val_path[1], "val", batch_size//4+1, shuffle=False, num_workers=0,
                                  drop_last=False)
    return train_loader, valid_loader

def build_test_dataloader(val_path, eval=True):
    test_loader = get_dataloader(val_path[0], val_path[1], "test", 1, shuffle=False, num_workers=0,
                                 drop_last=False, eval=eval)
    return test_loader
