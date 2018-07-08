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
    if x.requires_grad:
        x = x.detach()
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
        print('TorchFun:imshow:Guessed pixel value range:0~255')
    elif min_intensity<0 and min_intensity>=-0.5 and max_intensity>0 and max_intensity <=0.5:
        # -0.5 - 0.5
        gridnp += 0.5
        print('TorchFun:imshow:Guessed pixel value range:-0.5~0.5')
    elif min_intensity<-0.5 and min_intensity>=-1 and max_intensity>0.5 and max_intensity <=1:
        # -1 - 1
        gridnp /= 2
        gridnp += 0.5
        print('TorchFun:imshow:Guessed pixel value range:-1~1')
    elif min_intensity>=0 and max_intensity<=1:
        print('TorchFun:imshow:Guessed pixel value range:0~1')
    else:
        print('TorchFun:imshow:Cannot speculate the value-range of this image. Please normalize the image manually before using imshow.')
        return

    import matploatlib
    if matplotlib.get_backend() == 'WebAgg':
        print('TorchFun:imshow:Warning, you are using WebAgg backend for Matplotlib. Please consider windowed display SDKs such as TkAgg backend and GTK* backends.')
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

def save(a,b):
    '''
    save weight `a` into target `b`, or save model `b` into target `a`
    The order of the arguments doesn't matter.
    Example:
        >save('weights.pts',model)
    or
        >save(model,'weights.pts')
    or
        >f = open('weight.pts')
        >save(f,model)
    or
        >save(model,f)
    or
        >save('weights.pts',state_dict)
    or
        >save(state_dict,'weights.pts')
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
    elif isinstance(a,dict):
        model = a
        arg_file_pos = 1
    elif isinstance(b,dict):
        model = b
        arg_file_pos = 0
    else:
        print('TorchFun:save(): Warning! neither of the arguments is pytorch model, abort loading.')
        return

    target = args[arg_file_pos]
    if isinstance(target,io.TextIOWrapper):
        # file handle input
        target.close()
        target = target.name
    if isinstance(model,dict):
        weights = model
    else:
        weights = model.state_dict()
    torch.save(weights,target)
    return

def count_parameters(model_or_dict):
    '''Arguements:
    model_or_dict: model or state dictionary
    Return: parameter amount in python-int
            Returns 0 if datatype not understood
    Usage:
        count_parameters(model)
        count_parameters(state_dict)
        count_parameters(weight_tensor)
        count_parameters(numpy_array)
    '''
    if isinstance(model_or_dict,torch.nn.Module):
        w = model_or_dict.state_dict()
    elif isinstance(model_or_dict,dict):
        w = model_or_dict
    elif getattr(model_or_dict,'shape',default=False):
        return np.prod(model_or_dict.shape)
    else:
        print('TorchFun:count_parameters:Warning, Unknown data type:',model_or_dict.__class__)
        return 0
    num = 0
    for k in model_or_dict:
        num += np.prod(model_or_dict[k].shape)
    return num

