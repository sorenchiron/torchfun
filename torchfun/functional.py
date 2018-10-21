from __future__ import division,print_function
import torch
from .torchfun import sort_args
import torch.nn.functional as F

def conv2d_dfs(x,weight,bias=None,stride=1,padding=0,dilation=1):
    '''depth fully shared conv2d.
    weight: shape shoule be : 1 x 1 x kh x kw
    bias: contains only 1 number or None
    '''
    b,c,h,w = x.shape
    x = x.contiguous().view(b*c,1,h,w,bias,stride,padding,dilation,groups=1)
    return x.contiguous().view(b,c,h,w)
