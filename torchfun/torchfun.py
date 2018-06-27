from __future__ import division,print_function
import numpy as np
import torch
import io
t = torch

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

def imshow(x,title=None,auto_close=True):
    '''only deal with torch channel-first image batch,
    title: add title to plot. (Default None)
        title can be string, or any string-able object.
    auto_close: (default True) 
        Close the pyplot session afterwards. 
        Clean the environment just like you had 
        never used matplotlib here.
        if set to False, the plot will remain in the memory for further drawings.'''
    import torchvision
    shapes = x.shape
    if len(shapes)==3:
        x = t.unsqueeze(x,dim=0)
    grid = torchvision.utils.make_grid(x)
    gridnp = grid.numpy()
    max_intensity = gridnp.max()
    min_intensity = gridnp.min()
    if min_intensity>=0 and max_intensity>1:
        # 0 - 255
        gridnp /= 255
    elif min_intensity<0 and min_intensity>=-0.5 and max_intensity>0 and max_intensity <=0.5:
        # -0.5 - 0.5
        gridnp += 0.5
    elif min_intensity<-0.5 and min_intensity>=-1 and max_intensity>0.5 and max_intensity <=1:
        # -1 - 1
        gridnp /= 2
        gridnp += 0.5

    import matplotlib.pyplot as plt
    plt.imshow(np.transpose(gridnp,(1,2,0)))
    if title:
        plt.title(title)
    plt.show()
    if auto_close:
        plt.close()
        del plt
    else:
        plt.pause(0.001)

def load(a,b):
    '''
    Load weight `a` into model `b`, or load model `b` using weight `a`
    The order of the arguments doesn't matter.
    Example:
        >load('weights.pts',model)
    or
        >load(model,'weights.pts')
    or
        >f = open('weight.pts')
        >load(f,model)
    or
        >load(model,f)

    Return value: None
    '''
    args = (a,b)
    arg_file_pos = None
    if isinstance(a,torch.nn.Module):
        model = a 
        arg_file_pos = 1
    elif isinstance(b,torch.nn.Module):
        model = b
        arg_file_pos = 0
    else:
        print('TorchFun:load(): Warning! neither of the arguments is pytorch model, abort loading.')
        return

    source = args[arg_file_pos]
    if isinstance(source,io.TextIOWrapper):
        # file handle input
        source = io.BytesIO(f.read())
    weights = torch.load(source)
    model.load_state_dict(weights)
    return