# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 

from __future__ import division,print_function
import torch
import numpy as np

__doc__ = '''Neural Network related layers/functions/classes
that are compatible with all pyTorch usage.
'''


def flatten(x):
    '''Flatten function
    Usage:
        out = flatten(x)
    '''
    shapes = x.shape
    dims = len(shapes)
    flatten_length = torch.prod(shapes[1:])
    return x.view(-1,flatten_length)

class Flatten(torch.nn.Module):
    '''Flatten module
    Usage:
        flat = Flatten()
        out = flat(x)
    '''
    def __init__(self):
        pass
    def forward(self,x):
        return flatten(x)

def subpixel(x,out_channels=1):
    '''Unfold channel/depth dimensions to enlarge the feature map
    Notice: Output size is deducted. 
    The size of the unfold square is automatically determined
    e.g. :
        images: 100x16x16x9.  9=3x3 square
        subpixel-out: 100x48x48x1
    Arguement:
        x: NCHW image, channel first.
        out_channels, channel number of output feature map'''
    b,c,h,w = shapes = x.shape
    out_c = out_channels
    if c%out_c != 0:
        print('input has',c,'channels, cannot be split into',out_c,'parts')
        raise Exception('subpixel inappropriate size')
    unfold_dim = c//out_c
    l = int(np.sqrt(unfold_dim))
    if l**2 != unfold_dim:
        print('remaining',unfold_dim,'digits for each channel, unable to sqrt.')
        raise Exception('subpixel inappropriate size')
    ###### start ######
    x.transpose_(1,3) # nhwc
    y = x.reshape(b,h,w,unfold_dim,out_c)
    y.transpose_(2,3)
    y = y.reshape(b,h,l,l,w,out_c)
    y = y.reshape(b,l*h,l,w,out_c) # b,l*h,l,w,outc
    y.transpose_(2,3) # b,l*h,w,l,outc
    y = y.reshape(b,l*h,l*w,out_c)
    y.transpose_(1,3) # nchw
    return y

class Subpixel(torch.nn.Module):
    '''Unfold channel/depth dimensions to enlarge the feature map
    Notice: Output size is deducted. 
    The size of the unfold square is automatically determined
    e.g. :
        images: 100x16x16x9.  9=3x3 square
        subpixel-out: 100x48x48x1
    Arguement:
        out_channels, channel number of output feature map'''
    def __init__(self,out_channels=1):
        self.out_channels = out_channels
    def forward(self,x):
        return subpixel(x,out_channels=self.out_channels)