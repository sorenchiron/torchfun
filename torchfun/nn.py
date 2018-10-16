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
    #shapes = x.shape
    #dims = len(shapes)
    #flatten_length = torch.prod(shapes[1:])
    return x.view(x.size(0),-1)

class Flatten(torch.nn.Module):
    '''Flatten module
    Usage:
        flat = Flatten()
        out = flat(x)
    '''
    def __init__(self):
        super(self.__class__,self).__init__()

    def forward(self,x):
        return flatten(x)

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
    x = x.transpose(1,3) # nhwc
    y = x.reshape(b,h,w,unfold_dim,out_c)
    y = y.transpose(2,3)
    y = y.reshape(b,h,l,l,w,out_c)
    y = y.reshape(b,l*h,l,w,out_c) # b,l*h,l,w,outc
    y = y.transpose(2,3) # b,l*h,w,l,outc
    y = y.reshape(b,l*h,l*w,out_c)
    y = y.transpose(1,3) # nchw
    return y

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
        super(self.__class__,self).__init__()
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
            c = x.size(1) // self.patch_pixels
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

        super(self.__class__, self).__init__(slice_depth, slice_out, 
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
        xslices = x.view(b*self.trunks,self.slice_depth,h,w)
        yraw = self.conv2d_fn(xslices, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        _,_,h,w = yraw.shape
        y = yraw.view(b,self.out_channels,h,w)
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
        super(self.__class__,self).__init__()
        self.dim = dim
    def forward(self,x):
        if self.dim:
            return x.squeeze_(self.dim)
        else:
            return x.squeeze_()

class AbsMax(torch.nn.Module):
    def __init__(self,dim=1):
        super(self.__class__,self).__init__()
        self.dim = dim
    def forward(self,x):
        signs = x.sign()
        absx = x.abs()

        
