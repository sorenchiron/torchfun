# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 

from __future__ import division,print_function
import numpy as np
import torch
import io
from tqdm import tqdm
t = torch

def sort_args(args_or_types,types_or_args):
    if type(args_or_types[0]) is type:
        types = args_or_types
        args = types_or_args
    else:
        types = types_or_args
        args = args_or_types

    type_arg_dict = {type(a):a for a in args}
    res=[]
    for t in types:
        res.append(type_arg_dict[t])
    return res


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
        gridnp = gridnp/255
        print('TorchFun:imshow:Guessed pixel value range:0~255')
    elif min_intensity<0 and min_intensity>=-0.5 and max_intensity>0 and max_intensity <=0.5:
        # -0.5 - 0.5
        gridnp += 0.5
        print('TorchFun:imshow:Guessed pixel value range:-0.5~0.5')
    elif min_intensity<-0.5 and min_intensity>=-1 and max_intensity>0.5 and max_intensity <=1:
        # -1 - 1
        gridnp = gridnp/2
        gridnp += 0.5
        print('TorchFun:imshow:Guessed pixel value range:-1~1')
    elif min_intensity>=0 and max_intensity<=1:
        print('TorchFun:imshow:Guessed pixel value range:0~1')
    else:
        print('TorchFun:imshow:Cannot speculate the value-range of this image. Please normalize the image manually before using imshow.')
        return

    import matplotlib
    if matplotlib.get_backend() == 'WebAgg':
        print('TorchFun:imshow:Warning, you are using WebAgg backend for Matplotlib. Please consider windowed display SDKs such as TkAgg backend and GTK* backends.')
    import matplotlib.pyplot as plt

    print("TorchFun:imshow:Notice, To close the window, focus the mouse on the figure and press `q` key.")
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


def _g():
    '''used only for getting generator_type'''
    yield from range(1)
generator_type =  type(_g())
def count_parameters(model_or_dict_or_param):
    '''Count parameter numer of a module/state_dict/layer/tensor.
    This function can also print the occupied memory of parameters in MBs
    Arguements:
    model_or_dict_or_param: model or state dictionary or model.parameter(), or numpy-array, or tensor.
    Return: parameter amount in python-int
            Returns 0 if datatype not understood
    Usage:
    1. count trainable and untrainbale params
        count_parameters(model)
    same as    
        count_parameters(state_dict)
    2. count only trainable params:
        count_parameters(model.parameters())
    3. count data matrix
        count_parameters(weight_tensor)
        count_parameters(numpy_array)


    Alias: parameters()
    '''
    dtype_size={torch.float16:2,torch.float32:4,torch.float64:8,torch.int16:2,torch.int32:4,torch.int64:8,torch.uint8:1}
    if isinstance(model_or_dict_or_param,torch.nn.Module):
        s = model_or_dict_or_param.state_dict()
        w = [s[k] for k in s]
    elif isinstance(model_or_dict_or_param,dict):
        w = [model_or_dict_or_param[i] for i in model_or_dict_or_param]
    elif isinstance(model_or_dict_or_param,generator_type):
        w = [i for i in model_or_dict_or_param]
    elif getattr(model_or_dict_or_param,'shape',False):
        w= [model_or_dict_or_param.shape]
    else:
        print('TorchFun:count_parameters:Warning, Unknown data type:',model_or_dict.__class__)
        return 0
    num = 0
    Bytes = 0
    for p in w:
        num += np.prod(p.shape)
        dtype = p.dtype if p.dtype in dtype_size else torch.float32
        Bytes += num * dtype_size[dtype]
    print('Torchfun:parameters:',Bytes/1024/1024,'MBs',num,'params')
    return num

parameters = count_parameters

def show_layers_parameters(model):
    total_params=0
    print('-----------start-----------')
    for i,l in enumerate(model,1):
        l_params = count_parameters(l)
        total_params+=l_params
        print('layer',i,l.__class__.__name__,'params:',l_params)
    print('---------------------------')
    print('total parameters:',total_params)
    print('------------end------------')


import os
module_type = type(os)

class Packsearch(object):
    '''Given an module object as input:
    > p = Packsearch(torch)
    the instance p provide p.search() method. So that you can 
    search everything inside this package
    > p.search('maxpoo')
    output:
        Packsearch: 35 results found:
        -------------results start-------------
        0        torch.nn.AdaptiveMaxPool1d
        1        torch.nn.AdaptiveMaxPool2d
        2        torch.nn.AdaptiveMaxPool3d
        3        torch.nn.FractionalMaxPool2d
        4        torch.nn.MaxPool1d
        5        torch.nn.MaxPool2d
        ...
    '''
    def __init__(self,module_object,auto_init=True):
        super(self.__class__,self).__init__()
        self.root = module_object
        self.target = self.root.__name__
        self.name_list = []
        self.name_dict = []
        if auto_init:
            print('Packsearch:','caching names in the module...')
            self.traverse(self.root)
            self.preprocess_names()
            print('Packsearch:','done, please use search().')

    def preprocess_names(self):
        self.name_list = sorted(self.name_list)
        self.name_dict = [(i.lower(),i) for i in self.name_list]

    def traverse(self,mod):
        '''gather all names and store them into a name_list'''
        if type(mod) is not module_type:
            return
        root_name = mod.__name__+'.' # make `torchvision.` != `torch.`
        root_name_len = len(root_name)
        traversed_mod_names = [root_name]
        stack=[mod]
        while stack:
            m = stack.pop()
            prefix = m.__name__
            if prefix in traversed_mod_names:
                continue
            names = dir(m)
            for name in names:
                obj = getattr(m,name)
                if type(obj) is module_type:
                    obj_path = obj.__name__
                    obj_prefix = obj_path[:root_name_len]
                    if obj_prefix == root_name:
                    # only include mods of current package.
                    # packages like io,numpy,os, will be excluded.
                        stack.insert(0,obj) # to be traversed later
                else:
                    pass # do not traverse non-module objects
                self.name_list.append(prefix+'.'+name)
            del m
            # record that we have visited this module
            traversed_mod_names.append(prefix)

    def dynamic_traverse(self,mod,query):
        '''traverse the module and simultaneously search for queried name'''
        if type(mod) is not module_type:
            return
        root_name = mod.__name__+'.' # make `torchvision.` != `torch.`
        root_name_len = len(root_name)
        traversed_mod_names = [root_name]
        stack=[mod]
        while stack:
            m = stack.pop()
            prefix = m.__name__
            if prefix in traversed_mod_names:
                continue
            names = dir(m)
            for name in names:
                obj = getattr(m,name)
                if type(obj) is module_type:
                    obj_path = obj.__name__
                    obj_prefix = obj_path[:root_name_len]
                    if obj_prefix == root_name:
                    # only include mods of current package.
                    # packages like io,numpy,os, will be excluded.
                        stack.insert(0,obj) # to be traversed later
                else:
                    pass # do not traverse non-module objects
                obj_path = prefix+'.'+name
                if query in obj_path.lower():
                    yield obj_path
            del m
            # record that we have visited this module
            traversed_mod_names.append(prefix)

    def search(self,name):
        q = name.lower()
        res = []
        for k,full_name in tqdm(self.name_dict):
            if q in k:
                res.append(full_name)
        res = sorted(res)
        if not res:
            print('Packsearch:searched',len(self.name_list),'items (instances, functions, modules).',name,'not found.')
        else:
            print('Packsearch:',len(res),'results found:')
            print('-------------results start-------------')
            for i,r in enumerate(res):
                print('%d\t'%i,r)
            print('--------------results end--------------')

    def __len__(self):
        return len(self.name_list)
    def __str__(self):
        return 'Packsearch engine applied on {}, with {} names recoreded. Powered by Torchfun.'.format(self.target,len(self))
    def __repr__(self):
        obj_desc = super(self.__class__,self).__repr__()
        return self.__str__() + os.linesep + 'location:' + obj_desc

def packsearch(module_or_str,str_or_module):
    '''Given an module object, and search pattern string as input:
    > packsearch(torch,'maxpoo')
    or
    > packsearch('maxpoo',torch)
    you can search everything inside this package
    output:
        Packsearch: 35 results found:
        -------------results start-------------
        0        torch.nn.AdaptiveMaxPool1d
        1        torch.nn.AdaptiveMaxPool2d
        2        torch.nn.AdaptiveMaxPool3d
        3        torch.nn.FractionalMaxPool2d
        4        torch.nn.MaxPool1d
        5        torch.nn.MaxPool2d
        ...
    '''
    mod,name = sort_args([module_type,str],[module_or_str,str_or_module])
    p = Packsearch(mod,auto_init=False)
    res_counter = 0
    print('Packsearch: dynamic searching start ...')
    for r in p.dynamic_traverse(mod,name.lower()):
        print(r)
        res_counter += 1
    if res_counter == 0:
        print('Packsearch: no result found')
    else:
        print('Packsearch: search done,',res_counter,'results found.')


def hash_parameters(module,use_sum=False):
    '''return the summary of all variables.
    This is used to detect chaotic changes of weights.
    You can check the sum_parameters before and after some operations, to know
    if there is any change made to the params.

    I use this function to verify gradient behaviours.'''
    means = []
    for p in module.parameters():
        if use_sum:
            v = p.sum()
        else:
            v = p.mean()
        means.append(v)
    return float(sum(means))