import torch
import torchfun as tf 
from torchfun import *
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

class MaxMinNorm(Clip):
    __doc__ = max_min_norm.__doc__
    def __init__(self,max_or_min,min_or_max,eps=1e-5):
        super(MaxMinNorm,self).__init__(max_or_min,min_or_max)
        eps = torch.tensor(eps,dtype=torch.float)
        self.register_buffer(name='eps',tensor=eps)
    def forward(self,x):
        return max_min_norm(x,self.maxv,self.minv,self.eps)


x = torch.rand(1,2,3,4)
x0 = x[0]
x0min = x0.min()
x0max = x0.max()
y0 = ((x0-x0min)/(x0max-x0min+1e-5))*(1 - -1) + -1
print(tf.hash_parameters(y0))

m = MaxMinNorm(-1,1)
out = m(x)

print(tf.hash_parameters(out[0]))

print((out[0]-y0).abs().sum())