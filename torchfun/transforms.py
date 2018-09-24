# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 

from PIL import ImageFilter
import random

__doc__ = '''Transforms used for data augmentation in torchvision.datasets.*
or in torch.utils.data.Dataset'''

class RandomGaussianBlur(object):
    '''PIL image'''
    def __init__(self,kernel_ratio=0.01,random_ratio=0.005,pixel_range=None):
        '''
        Arguemet:
            kernel_ratio: radius ratio with respect to image width
            random_ratio: random_noise applied to kernel_ratio
        '''
        super(self.__class__,self).__init__()
        self.__dict__.update(locals())
        assert(random_ratio<kernel_ratio)

    def __call__(self,img):
        l,w = img.size
        avg_size = (l+w)//2

        kernel_size_ratio = (random.random()-0.5)*2*self.random_ratio \
                    + self.kernel_ratio
        kernel_size = max(int(kernel_size_ratio * avg_size),1)
        fltr = ImageFilter.GaussianBlur(kernel_size)
        return img.filter(fltr)

    def __repr__(self):
        return 'GaussianBlur with kernel_ratio:%f random_ratio:%f' %(self.kernel_ratio,self.random_ratio)
        