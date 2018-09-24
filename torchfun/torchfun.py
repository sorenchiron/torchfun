# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 

from __future__ import division,print_function
import numpy as np
import torch
import io,os,importlib,sys
from tqdm import tqdm
from .nn import *
t = torch

def sort_args(args_or_types,types_or_args):
    '''
    This is a very interesting function.
    It is used to support __arbitrary-arguments-ordering__ in TorchFun.

    Input:
        The function takes a list of types, and a list of arguments.

    Returns:
        a list of arguments, with the same order as the types-list.

    Of course, `sort_args` supports arbitrary-arguments-ordering by itself.

    '''
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
    KBytes = 0
    for p in w:
        this_num = np.prod(p.shape)
        num += this_num
        dtype = p.dtype if p.dtype in dtype_size else torch.float32
        KBytes += (this_num/1024*dtype_size[dtype])
    print('Torchfun:parameters:%.2f' % (KBytes/1024),'MBs',num,'params')
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

module_type = type(os)

class Packsearch(object):
    '''Given an module object as input:
    > p = Packsearch(torch)
    or
    > p = Packsearch('torch')
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
        if isinstance(module_object,module_type):
            self.root = module_object
        else:
            self.root = importlib.import_module(module_object)
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


def hash_parameters(model_or_statdict_or_param,use_sum=False):
    '''return the summary of all variables.
    This is used to detect chaotic changes of weights.
    You can check the sum_parameters before and after some operations, to know
    if there is any change made to the params.

    I use this function to verify gradient behaviours.

    By default, This only hash the trainable parameters!

    arguements:
    module_or_statdict_or_param: torch.nn.module, 
                    or model.state_dict(), 
                    or model.parameters().
    use_sum: return the sum instead of mean value of all params.'''
    m = model_or_statdict_or_param
    means = []
    params = []
    if isinstance(m,Torch.nn.Module):
        params = [m[k] for k in m]
    elif isinstance(m,dict):
        params = m.parameters()
    elif isinstance(m,generator_type):
        params = parameters
    else:
        print('TorchFun:hash_parameters:','input type not support:',type(m))

    for p in params:
        if use_sum:
            v = p.sum()
        else:
            v = p.mean()
        means.append(v)

    return float(sum(means))

def force_exist(dirname,verbose=True):
    '''force a directory to exist.
    force_exist can automatically create directory with any depth.
    Arguements:
        dirname: path of the desired directory
        verbose: print every directory creation. default True.
    Usage:
        force_exist('a/b/c/d/e/f')
        force_exist('a/b/c/d/e/f',verbose=False)
    '''
    
    if dirname == '' or dirname == '.':
        return True
    top = os.path.dirname(dirname)
    force_exist(top)
    if not os.path.exists(dirname):
        if verbose:
            print('creating',dirname)
        os.makedirs(dirname)
        return False
    else:
        return True


def omini_open(path):
    import subprocess
    import platform
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def whereis(module_or_string,open_gui=True):
    '''find the source file location of a module
    arguments:
        module_or_string: target module object, or it's string path like `torch.nn`
        open_gui: open the folder with default window-manager.
    returns:
        module file name, or None
    '''
    if isinstance(module_or_string,module_type):
        mod = module_or_string
    elif isinstance(module_or_string,str):
        try:
            mod = importlib.import_module(module_or_string)
        except:
            s = '.'.join(module_or_string.split('.')[:-1])
            print('TorchFun:warning:',module_or_string,'is not a module. Maybe it is a node class. TorchFun is trying to parse its father prefix:',s)
            try:
                mod = importlib.import_module(s)
            except Exception as e:
                print('TorchFun:error:','father prefix',s,'is not valid')
                raise e
    else:
        print('TorchFun:error:invalid arguement',module_or_string)
        return None
    fname = mod.__file__
    dirname = os.path.dirname(mod.__file__)
    print(fname)
    if open_gui:
        omini_open(dirname)

    return fname
    
def pil_imshow(arr):
    """
    Simple showing of an image through an external viewer.

    This function is only available if Python Imaging Library (PIL) is installed.

    Uses the image viewer specified by the environment variable
    SCIPY_PIL_IMAGE_VIEWER, or if that is not defined then `see`,
    to view a temporary file generated from array data.

    .. warning::

        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).

    Parameters
    ----------
    arr : ndarray
        Array of image data to show.

    Returns
    -------
    None

    Examples
    --------
    >>> a = np.tile(np.arange(255), (255,1))
    >>> from scipy import misc
    >>> misc.imshow(a)

    Ported and upgraded based on scipy.misc.imshow
    Open-sourced according to the license.
    """
    from PIL import Image
    import tempfile

    # to get the tempdir, or the tempdir is None by default.
    fnum,fname = tempfile.mkstemp()
    os.close(fnum)
    os.unlink(fname)

    oldpath = os.path.join(tempfile.tempdir,'torchfun_pil_imshow_tempimg_prefix.png')
    if os.path.exists(oldpath):
        print('cleaning old tmp image.')
        os.unlink(oldpath)

    im = Image.fromarray(arr)
    fnum, fname = tempfile.mkstemp('.png',prefix='torchfun_pil_imshow_tempimg_prefix')
    if im.mode != 'RGB':
        im = im.convert('RGB')

    try:
        im.save(fname)
    except Exception as e:
        print(e)
        raise RuntimeError("Error saving temporary image data.")
    os.close(fnum)
    omini_open(fname)