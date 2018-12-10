from __future__ import division,print_function
import torch
from .torchfun import sort_args
import torch.nn.functional as F
import itertools

__doc__ = 'mathematical functions'

def conv2d_dfs(x,weight,bias=None,stride=1,padding=0,dilation=1):
    '''depth fully shared conv2d.
    Argument:
        x: input image with size: N x C x H x W
        weight: shape shoule be : 1 x 1 x kernel-height x k-width
        bias: contains only 1 number or None
    '''
    b,c,h,w = x.shape
    x = x.contiguous().view(b*c,1,h,w)
    x = F.conv2d(x,weight,bias,stride,padding,dilation,groups=1)
    _,_,h,w = x.shape
    return x.contiguous().view(b,c,h,w)

def flatten(x):
    '''Flatten function
    Usage:
        out = flatten(x)
    '''
    #shapes = x.shape
    #dims = len(shapes)
    #flatten_length = torch.prod(shapes[1:])
    return x.contiguous().view(x.size(0),-1)

def subpixel(x,out_channels=1):
    '''Unfold channel/depth dimensions to enlarge the feature map
    Notice: Output size is deducted. 
    The size of the unfold square is automatically determined
    e.g. :
        images: 100x9x16x16.  9=3x3 square
        subpixel-out: 100x1x48x48
    Arguement:
        x: NCHW image, channel first.
        out_channels, channel number of output feature map'''
    b,c,h,w = shapes = list(x.shape)
    out_c = int(out_channels)
    if c%out_c != 0:
        print('input has',c,'channels, cannot be split into',out_c,'parts')
        raise Exception('subpixel inappropriate size')
    unfold_dim = int(c//out_c)
    l = int(unfold_dim**0.5)
    if l**2 != unfold_dim:
        print('remaining',unfold_dim,'digits for each channel, unable to sqrt.')
        raise Exception('subpixel inappropriate size %f**2 != %d' %(l,int(unfold_dim)))
    ###### start ######
    x = x.transpose(1,3) # nhwc
    y = x.reshape(b,h,w,unfold_dim,out_c)
    y = y.transpose(2,3)
    y = y.reshape(b,h,l,l,w,out_c)
    y = y.reshape(b,l*h,l,w,out_c) # b,l*h,l,w,outc
    y = y.transpose(2,3) # b,l*h,w,l,outc
    y = y.reshape(b,l*h,l*w,out_c)
    y = y.transpose(1,3) # nchw
    return y

def clip(in_tensor,max_or_min,min_or_max):
    '''limit the values in in_tensor to be within [min,max].
    values larger than max will be cut to max, respectively for mins.
    the order of max/min doesn't matter.
    the operation is not in-place, that saves you alot troubles.
    Notice: this clip is not inplace, and neither can pass backward derivatives
            when the values are clipped to min/max.
    Warning: When applied during loss-calculating in training, this clip will cause
            gradient disapperance.
    '''
    minv,maxv = torch.tensor(sorted([max_or_min,min_or_max]),dtype=in_tensor.dtype).to(in_tensor.device)
    x = torch.max(in_tensor,minv)
    x = torch.min(x,maxv)
    return x

def clip_(in_tensor,max_or_min,min_or_max):
    '''limit the values in in_tensor to be within [min,max].
    values larger than max will be cut to max, respectively for mins.
    the order of max/min doesn't matter.
    the operation is not in-place, that saves you alot troubles.
    Notice: this clip is not inplace.
            But, for clipped values, the gradients will always be passed backwards.
            This is useful when 
    '''
    minv,maxv = sorted((max_or_min,min_or_max))

    greater_pos = (in_tensor.detach()>maxv).type_as(in_tensor)
    diff = (maxv-in_tensor.detach())*greater_pos
    in_tensor = in_tensor+diff

    less_pos = (in_tensor.detach()<minv).type_as(in_tensor)
    diff = (minv-in_tensor.detach())*less_pos
    in_tensor = in_tensor+diff
    return in_tensor



def add_noise(in_tensor,noise_type='normal',noise_param=(0,1),range_limit=(-1,1)):
    '''
    Add noise to input tensor.
    Noise type can be either `normal` or `uniform`
        * for normal, (mean,std) is required as noise_param
        * for uniform (min,max) is required as noise_param
    The range of the output tensor can be limited,
      by giving `range_limit`:(min,max)
    '''
    if noise_type=='uniform':
        nmin,nmax = noise_param
        noise = torch.zeros_like(in_tensor).uniform_(nmin,nmax)
    elif noise_type=='normal':
        mean,std = noise_param
        noise = torch.zeros_like(in_tensor).normal_(mean,std)
    if range_limit is not None:
        return clip(in_tensor+noise,*range_limit)
    else:
        return in_tensor+noise


def instance_mean_std(x,num_features):
    '''NCHW'''
    b,c,h,w = x.shape
    nc_groups = c//num_features
    x = x.contiguous().view(b,nc_groups,num_features,-1)
    x = x.transpose(1,2) # b, numfeatu, ncgroup, h*w
    x = x.contiguous()
    x = x.contiguous().view(b*num_features,-1)

    mean = x.mean(1)
    std = x.std(1)

    mean = mean.contiguous().view(b,num_features) # n,feature
    std = std.contiguous().view(b,num_features) # n,feature
    return mean,std

def instance_renorm(x,mean,std,eps=1e-5):
    '''Make x have given mean and std.
    Argument:
    x: NCHW
    mean: tensor with size: N-by-num-features
    std: tensor with size: N-by-num-features
    eps: default 1e-5
    '''
    b,c,h,w = x.shape
    b,num_features = mean.shape
    nc_groups = c//num_features

    x_m,x_std = instance_mean_std(x,num_features)
    x_m,x_std = x_m.detach(),x_std.detach()
    x = x.contiguous().view(b,nc_groups,num_features,-1)
    x = x.transpose(2,3).transpose(0,2) # -1,ncgroup,b,numfeature
    x = ((x-x_m)/(x_std+eps))*(std+eps) + mean
    x = x.transpose(0,2).transpose(2,3) # b,nc_groups,num_features,-1
    x = x.contiguous().view(b,c,h,w)
    return x.contiguous()

def max_min_norm(x,to_max,to_min,eps=1e-5):
    '''scale the input x into range [to_min,to_max].
    Arguments:
        to_max: target expect max value of the output
        to_min: target expect min value of the output
    '''
    b,c,h,w = x.shape
    x = flatten(x)
    xmax,_ = x.max(1)
    xmin,_ = x.min(1)
    x = x.transpose(0,1)
    x = (x-xmin)/(xmax-xmin+eps)
    x = x * (to_max - to_min) + to_min
    x = x.transpose(0,1)
    return x.view(b,c,h,w)

### training related functional tools
def combine_parameters(*models):
    '''Combine the parameters of serveral trainable module object,
    into one unified parameter generator.
    Arguements:
            *models: any number of models.

    This utility is useful when you want several individual parts to be
    handled by one Optimizer. Parameters shall be gathered into one iterator
    first, because torch.optimizers require only one parameter-iterator as input'''
    return itertools.chain(*[m.parameters() for m in models])
