import torch.utils.data as data
from PIL import Image

import os

from torchvision.datasets.folder import find_classes,make_dataset
from torchvision.datasets.folder import DatasetFolder,ImageFolder
from torchvision.datasets.folder import default_loader,IMG_EXTENSIONS

class ImageAugmentationDataset(ImageFolder):
    '''
        Pre_transform is applied to the source image firstly.
        Then, degrade_transform is applied to get degraded image (such as blurred image).
        Finally, both the degraded imgs and the pre-transformed images are uniformly post-transformed.
        Usually, the post-transforms are ToTensor() and Normalize()

        Degrade transform should have two arguments as input:
            1: input image to be degraded
            2: Boolean input indicates if the degrading parameters should be returned.
    '''
    def __init__(self, root, 
                pre_transform=None, 
                degrade_transform=None,
                post_transform=None,
                loader=default_loader):
        super(self.__class__, self).__init__(root, 
                            transform=None,
                            target_transform=None,
                            loader=loader)
        self.imgs = self.samples
        self.pre_transform = pre_transform
        self.degrade_transform = degrade_transform
        self.post_transform = post_transform        


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.pre_transform:
            sample = self.pre_transform(sample)
        degraded_sample = sample.copy()
        if self.degrade_transform:
            degraded_sample = self.degrade_transform(degraded_sample)

        if self.post_transform:
            sample = self.post_transform(sample)
            degraded_sample = self.post_transform(degraded_sample)

        return degraded_sample,sample