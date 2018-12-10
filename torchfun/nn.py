# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 

from __future__ import division,print_function
import torch
from . import functional
from .functional import *
from .torchfun import sort_args
from time import time
from collections import OrderedDict
import itertools

__doc__ = '''Neural Network related layers/functions/classes
that are compatible with all pyTorch usage.
'''

class DebugAgent(torch.nn.Module):
    '''wrapper around layers.
    this wrapper hooks the forward function, to measure time 
    consumption of this layer.'''
    def __init__(self,obj):
        super(DebugAgent,self).__init__()
        self.obj = obj
        self.name = obj.__class__.__name__
        self.bind(obj)
    def bind(self,obj):
        self.__dict__.update(obj.__dict__)
        self.deep_names = {name:getattr(obj,name) for name in obj.__dir__()}
        self.__dict__.update(self.deep_names)
    def __call__(self,*argv,**kw):
        start = time()
        ret = self.deep_names['forward'](*argv,**kw)
        end = time()
        print('debug:',self.name.rjust(15)+'\t:%.5f sec'%(end-start))
        return ret
    def __getattr__(self,name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.deep_names:
            return self.deep_names[name]
        else:
            return getattr(self.obj,name)


class Module(torch.nn.Module):
    '''More debugging/controlling methods with complete original
    features from torch.nn.Module
    provides:
        * debug mode: inspect running time layer by layer.
        * release mode: back to normal from debug mode.
        * freeze layers: set layers to be in-trainable
        * unfreeze: as the name implies.
        * unfreeze_all: so as the name implies.

    Notice: you can safely change the base class from torchfun module
            back to torch module, when you want to publish the model.
            the state_dict will be loaded correctly and the forward() will
            function the same.
    Hint:   Consider establishing a BaseClass global variable at the top of your 
            code. Use a argument parser to select between torchfun.nn.Module and torch.nn.Module,
            so that the following classes follows the specified base class

            Example::
                    | import torch.nn.Module as ReleaseModule
                    | import torchfun.nn.Module as DevModule    
                    | from sys import argv
                    | if argv[1] == 'develop':
                    |     Base = DevModule
                    | elif argv[1] == 'release':
                    |     Base = ReleaseModule
                    | 
                    | class MyModel(Base):
                    |     ...

    '''
    def __init__(self):
        super(Module,self).__init__()
        self._debug=False
        self._frozen_names=[]
        if hasattr(self,'forward') and self.forward.__doc__ == None:
            self.forward.__doc__ = self.__class__.__doc__

    def debug(self):
        '''turn on debug mode.
        allow detailed timing report of forward()'''
        if self._debug:
            return
        else:
            self._debug=True
            self._modules_bak=self._modules
            self.debug_modules = OrderedDict()
            for name in self._modules:
                self.debug_modules[name]=DebugAgent(self._modules[name])
            self._modules = self.debug_modules
            self._update_using_modules_dict(self.debug_modules)
    def release(self):
        if self._debug:
            del self._modules
            self._modules = self._modules_bak
            self._update_using_modules_dict(self._modules_bak)
            self._debug = False
    def _update_using_modules_dict(self,odict):
        for name in odict:
            self.__setattr__(name,odict[name])
    def freeze(self,*obj_or_name):
        for o in obj_or_name:
            self._freeze(o,True)
    def unfreeze(self,*obj_or_name):
        for o in obj_or_name:
            self._freeze(o,False)
    def unfreeze_all(self):
        self._frozen_names = []
    def _freeze(self,obj_or_name,freeze=True):
        modules = {self._modules:name for name in self._modules}
        if obj_or_name in self._modules:
            name = obj_or_name
        elif obj_or_name in modules:
            name = modules[obj_or_name]
        else:
            raise Exception('layer or name:%s not in this module'%obj_or_name)
        if freeze:
            if name not in self._frozen_names:
                self._frozen_names.append(name)
        else:
            if name in self._frozen_names:
                del self._frozen_names[name]
        del modules
    
    def train(self,mode=True):
        super(Module,self).train(mode)
        for name in self._frozen_names:
            getattr(self,name).eval()

    def parameters(self):
        if self._frozen_names:
            generators = [self._modules[name].parameters() for name in self._modules if name not in self._frozen_names]
            return itertools.chain(*generators)
        else:
            return super(Module,self).parameters()


class Flatten(torch.nn.Module):
    '''Flatten module
    Usage:
        flat = Flatten()
        out = flat(x)
    '''
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,x):
        return flatten(x)

class Subpixel(torch.nn.Module):
    '''Unfold channel/depth dimensions to enlarge the feature map
    Notice: Output size is deducted. 
    The size of the unfold square is automatically determined
    e.g. :
        images: 100x16x16x9.  9=3x3 square
        subpixel-out: 100x48x48x1
    Arguement:
        out_channels, channel number of output feature map
        stride: enlarging ratio of spatial dimensions. stride=2 outputs x4 img area. If provided, out_channels will be ignored.'''
    def __init__(self,out_channels=1,stride=None):
        super(Subpixel,self).__init__()
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        if stride:
            self.patch_pixels = int(stride**2)
        else:
            self.patch_pixels = stride
    def forward(self,x):
        if not self.stride:
            return subpixel(x,out_channels=self.out_channels)
        else:
            c = int(x.size(1)) // self.patch_pixels
            return subpixel(x,out_channels=c)

class Conv2dDepthShared(torch.nn.Conv2d):
    r"""
    Applies a 2D convolution over an input signal composed of several input
    planes.

    Share the kernel along depth/channel direction.
    Conv2dDepthShared divides input images into multiple sub-layers(trunks) along depth axis, and use shared kernel to process each depth trunk.
    
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels. 
    The `weight` and `bias` matrix of a Conv2dDepthShared are low-rank. 
    They share trunks of digits repeatitively inside their matrices. 

    Example:
        Conv(in=3,out=9,k=3,s=1) will create kernel weight with size of (9x3x5x5)
        kernel of a Depth-shared-Conv2d only has 3x1x5x5 parameters.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image, inchannels must can be divided by trunks.
        out_channels (int): Number of channels produced by the convolution. out channels must can be divided by trunks.
        trunks (int): Number of trunks a single image is divided into (along depth). All trunks inside an image share same weight/bias.
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] - \text{dilation}[0]
                        * (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] - \text{dilation}[1]
                        * (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    
    """

    def __init__(self, in_channels, out_channels, trunks, kernel_size, 
        stride=1,padding=0, 
        dilation=1, groups=1, bias=True):
        if out_channels % trunks >0:
            message = self.__class__+':'+'out channels should be dividable by trunks!'
            print(message)
            raise Exception(message)
        elif in_channels % trunks >0:
            message = self.__class__+':'+'in channels should be dividable by trunks!'
            print(message)
            raise Exception(message)
        slice_depth = in_channels // trunks
        slice_out = out_channels // trunks

        super(Conv2dDepthShared, self).__init__(slice_depth, slice_out, 
            kernel_size, 
            stride,padding, 
            dilation=dilation, groups=groups, bias=bias)

        self.trunks = trunks
        self.slice_depth = slice_depth
        self.slice_out = slice_out
        self.conv2d_fn = torch.nn.functional.conv2d
        self.in_channels = int(slice_depth * trunks)
        self.out_channels = int(slice_out * trunks)

    def forward(self, x):
        b,c,h,w = x.shape
        xslices = x.contiguous().view(b*self.trunks,
                                    self.slice_depth,h,w)
        yraw = self.conv2d_fn(xslices, self.weight, 
                            self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        _,_,h,w = yraw.shape
        y = yraw.contiguous().view(b,self.out_channels,h,w)
        return y

class Squeeze(torch.nn.Module):
    '''squeeze(dim=None) -> Tensor

    Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.

    For example, if `input` is of shape:
    :math:`(A \times 1 \times B \times C \times 1 \times D)` then the `out` tensor
    will be of shape: :math:`(A \times B \times C \times D)`.

    When :attr:`dim` is given, a squeeze operation is done only in the given
    dimension. If `input` is of shape: :math:`(A \times 1 \times B)`,
    `squeeze(input, 0)` leaves the tensor unchanged, but :func:`squeeze(input, 1)` will
    squeeze the tensor to the shape :math:`(A \times B)`.

    .. note:: As an exception to the above, a 1-dimensional tensor of size 1 will
              not have its dimensions changed.

    .. note:: The returned tensor shares the storage with the input tensor,
          so changing the contents of one will change the contents of the other.
    
    Example::

        >>> x = torch.zeros(2, 1, 2, 1, 2)
        >>> x.size()
        torch.Size([2, 1, 2, 1, 2])
        >>> y = torch.squeeze(x)
        >>> y.size()
        torch.Size([2, 2, 2])
        >>> y = torch.squeeze(x, 0)
        >>> y.size()
        torch.Size([2, 1, 2, 1, 2])
        >>> y = torch.squeeze(x, 1)
        >>> y.size()
        torch.Size([2, 2, 1, 2]) 

    '''
    def __init__(self,dim=None):
        super(Squeeze,self).__init__()
        self.dim = dim
    def forward(self,x):
        if self.dim:
            return x.squeeze_(self.dim)
        else:
            return x.squeeze_()

class AbsMax(torch.nn.Module):
    '''
    TODO: not fully implemented
    '''
    def __init__(self,dim=1):
        super(AbsMax,self).__init__()
        self.dim = dim
    def forward(self,x):
        signs = x.sign()
        absx = x.abs()

class Clip(torch.nn.Module):
    __doc__=clip.__doc__
    def __init__(self,max_or_min,min_or_max,dtype=torch.float,keep_grad=True):
        super(Clip,self).__init__()
        minv,maxv = sorted([max_or_min,min_or_max])
        minv,maxv = torch.tensor([minv,maxv],dtype=dtype)
        self.register_buffer(name='minv',tensor=minv)
        self.register_buffer(name='maxv',tensor=maxv)
        if keep_grad:
            self.forward = self.forward_keep_grad
        else:
            self.forward = self.forward_clip_grad

    def forward_clip_grad(self,x):
        return torch.min(torch.max(x,self.minv),self.maxv)
    def forward_keep_grad(self,x):
        minv,maxv = self.minv,self.maxv
        in_tensor = x
        greater_pos = (in_tensor.detach()>maxv).type_as(in_tensor)
        diff = (maxv-in_tensor.detach())*greater_pos
        in_tensor = in_tensor+diff

        less_pos = (in_tensor.detach()<minv).type_as(in_tensor)
        diff = (minv-in_tensor.detach())*less_pos
        in_tensor = in_tensor+diff
        return in_tensor

class NO_OP(torch.nn.Module):
    '''A Module that repersents NO-Operation NO-OP.
    NO-OP is needed when programmers want customizable dynamic assembling
    of models. 
    To disable some layers, instead of using multiple `if` clauses, nn-parts can be configured to be
    NO-OP class, which will make that part turned-off in all occurrence.

    Notice: NO_OP will accept any init-args, and ignore them.
    '''
    def __init__(self,*argv,**kws):
        super(NO_OP,self).__init__()
    @staticmethod
    def forward(x,*argv,**kws):
        return x
    forward.__doc__ = __doc__


class InstanceMeanStd(torch.nn.Module):
    __doc__ = instance_mean_std.__doc__
    def __init__(self,num_features):
        super(self.__class__,self).__init__()
        self.num_features=num_features
    def forward(self,x):
        return instance_mean_std(x,self.num_features)
    forward.__doc__ = __doc__


class InstanceReNorm(torch.nn.Module):
    '''re normalize input with given mean and std-dev value.'''
    __doc__ = instance_renorm.__doc__
    def __init__(self,eps=1e-5):
        super(self.__class__,self).__init__()
        self.eps=eps
        #self.register_buffer(name='eps',tensor=torch.tensor(eps))
    def forward(self,x,mean,std):
        return instance_renorm(x,mean,std,self.eps)
    forward.__doc__ = __doc__

class MaxMinNorm(Clip):
    __doc__ = max_min_norm.__doc__
    def __init__(self,max_or_min,min_or_max,eps=1e-5):
        super(MaxMinNorm,self).__init__(max_or_min,min_or_max)
        self.eps = eps
        #self.register_buffer(name='eps',tensor=eps)
    def forward(self,x):
        return max_min_norm(x,self.maxv,self.minv,self.eps)

class ReLU(torch.nn.ReLU):
    '''activation that accepts any argument and ignores them.
    useful when you want to switch between different activations
    programatically, '''
    def __init__(self,*args,**kws):
        super(ReLU,self).__init__()

class Conv2dDepthFullyShared(torch.nn.Conv2d):
    NotImplemented

class Interpolate(torch.nn.Module):
    ''' resizing/scaling multi dimensional tensors
    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear` (4D-only), `trilinear` (5D-only), `area`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (string): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
        align_corners (bool, optional): if True, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is `linear`,
            `bilinear`, or `trilinear`. Default: False

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.
'''
    def __init__(self,size=None,scale_factor=None,mode='bilinear',align_corners=None):
        super().__init__()
        self.__dict__.update(locals())
        if (size is None) and (scale_factor is None):
            print('Torchfun: warning: size and scale_factor are both None, the interpolation parmeters are not specified!')
    def forward(self,x):
        return torch.nn.functional.interpolate(x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)