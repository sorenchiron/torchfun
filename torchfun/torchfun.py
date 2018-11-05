# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 

from __future__ import division,print_function
import numpy as np
import torch
import io,os,importlib,sys
from tqdm import tqdm
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

    type_arg_tuples = [(type(a),a) for a in args]
    res=[]
    for t in types:
        found = False
        for type_arg in type_arg_tuples:
            arg_type,arg_val = type_arg
            if issubclass(arg_type,t):
                found=True
                break
        if found:
            res.append(arg_val)
            type_arg_tuples.remove(type_arg)
        else:
            raise TypeError('One of the required argument is of type '+ t.__name__ + ', but none of the given arguments is of this type.')

    return res

def omini_open(path):
    '''
    Opens everything using system default viwer.

    This function can call system GUI to open folders,images,files,videos...
    '''
    import subprocess
    import platform
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def imshow(x,title=None,auto_close=True,rows=8,backend=None):
    '''only deal with torch channel-first image batch,
    title: add title to plot. (Default None)
        title can be string, or any string-able object.
    auto_close: (default True) 
        Close the pyplot session afterwards. 
        Clean the environment just like you had 
        never used matplotlib here.
        if set to False, the plot will remain in the memory for further drawings.
    rows: (default 8)
        the width of the output grid image.
    backend: None to use default gui. selections:
        WebAgg,  GTK3Agg,
        WX,      GTK3Cairo,
        WXAgg,   MacOSX,
        WXCairo, nbAgg,
        agg,     Qt4Agg,
        cairo,   Qt4Cairo,
        pdf,     Qt5Agg,
        pgf,     Qt5Cairo,
        ps,      TkAgg,
        svg,     TkCairo,
        template        
 
    Usage:
    ```python
        imshow(batch)
        imshow(batch,title=[a,b,c])
        imshow(batch,title='title')
        imshow(batch,auto_close=False) 
    ```
    Warnings:
    ```text
        TorchFun:imshow:Warning, you are using WebAgg backend for Matplotlib. 
        Please consider windowed display SDKs such as TkAgg backend and GTK* backends.
    ```
    This means your matplotlib is using web-browser for figure display. We __strongly__ recommend you to use window-based native display because browser-based backends are fragile and tend to crash. You can change the display mamanger for matplotlib each time you execute your script by:
    ```python
    import matplotlib
    matplotlib.use('TkAgg') # or GTK GTKAgg
    ```
    or permanantly by editing: `site-packages/matplotlib/mpl-data/matplotlibrc` and change backend to `TkAgg`

    A full list of available backends can be found at:
    ```python
    import matplotlib
    matplotlib.rcsetup.all_backends
    ```
    and, the TCL/TK GUI library for `tkinter` can be downloaded [here](https://www.tcl.tk/).

    Notice:
        If you use conda to manage your python versions, errors may occur when using TCL/TK.
        That's because conda secretly redirect your global python library path towards its.
        That will cause other stand-alone python versions to search from conda's lib dirs for binaries.
        To solve this, you may have to set:
        
        ```
        export TCL_LIBRARY=/usr/...pythondir.../lib/tcl8.6
        export TK_LIBRARY=/usr/...pythondir.../lib/tcl8.6
        ```
        or on windows:

        ```
        set "TCL_LIBRARY=/usr/...pythondir.../lib/tcl8.6"
        set "TK_LIBRARY=/usr/...pythondir.../lib/tcl8.6"
        ```

        '''
    import torchvision
    if x.requires_grad:
        x = x.detach()
    x = x.cpu()
    shapes = x.shape
    if len(shapes)==3:
        x = t.unsqueeze(x,dim=0)
    grid = torchvision.utils.make_grid(x,nrow=rows)
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
    if backend:
        print('using explicitly designated backend:',backend)
        matplotlib.use(backend,force=True)
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
    import PIL
    from PIL import Image
    from PIL.ImageFile import ImageFile
    import tempfile

    if isinstance(arr,torch.Tensor):
        return imshow(arr)

    # to get the tempdir, or the tempdir is None by default.
    fnum,fname = tempfile.mkstemp()
    os.close(fnum)
    os.unlink(fname)

    oldpath = os.path.join(tempfile.tempdir,'torchfun_pil_imshow_tempimg_prefix.png')
    if os.path.exists(oldpath):
        print('cleaning old tmp image.')
        os.unlink(oldpath)

    if isinstance(arr,ImageFile):
        im = arr
    else:
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
def count_parameters(model_or_dict_or_param, verbose=True):
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
    Notice:
        return value is parameter Number.

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
    if verbose:
        print('Torchfun:parameters:%.2f' % (KBytes/1024),'MBs',num,'params')
    return num

parameters = count_parameters

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
    if isinstance(m,torch.nn.Module):
        params = m.parameters()
    elif isinstance(m,dict):
        params = [m[k] for k in m]
    elif isinstance(m,generator_type):
        params = parameters
    elif isinstance(m,torch.Tensor):
        params = [m]
    else:
        print('TorchFun:hash_parameters:','input type not support:',type(m))

    for p in params:
        if use_sum:
            v = p.sum()
        else:
            v = p.mean()
        means.append(v)

    return float(sum(means))

def show_layers_parameters(model):
    total_params=0
    print('-----------start-----------')
    for i,name in enumerate(model._modules,1):
        l = model._modules[name]
        l_params = count_parameters(l,verbose=False)
        total_params+=l_params
        print('layer',str(i).ljust(3),l.__class__.__name__.rjust(15),'params:',l_params)
    print('---------------------------')
    print('total parameters:',total_params)
    print('------------end------------')

module_type = type(os)

class Packsearch(object):
    '''Search names inside a package.
    Given an module object as input:
    > p = Packsearch(torch)
    or
    > p = Packsearch('torch')
    the instance p provide p.search() method. So that you can 
    search everything inside this package
    > p.search('maxpoo')
    or simply
    > p('maxpoo')
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
    def __init__(self,module_object,auto_init=True,verbose=False):
        super(self.__class__,self).__init__()
        if isinstance(module_object,module_type):
            self.root = module_object
        else:
            try:
                self.root = importlib.import_module(module_object)
            except:
                raise TypeError('''Torchfun Packsearch: Wrong Argument!. torchfun.Packsearch is a class. 
                    it accept a torch Module or string as input. Use help(torchfun.Packsearch) to see the usage.
                    Or, you can use torch.packsearch(package,query) function.''')
        self.warned = False # to make warning appear only once.
        self.target = self.root.__name__
        self.name_list = []
        self.name_dict = []
        if auto_init:
            print('Packsearch:','caching names in the module...')
            self.traverse(self.root,search_attributes=verbose)
            self.preprocess_names()
            print('Packsearch:','done, please use search().')

    def preprocess_names(self):
        self.name_list = sorted(self.name_list)
        self.name_dict = [(i.lower(),i) for i in self.name_list]

    def traverse(self,mod,search_attributes=False):
        '''gather all names and store them into a name_list
        search_attributes: whether to include class attributes or method names
        '''
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

            names = [] 
            try:
                dir_ = dir(m)
                names.extend(dir_)
            except:
                if not self.warned:
                    print('Packsearch:warning:some objects were corrupted by their author, and they are not traversable.')
                    self.warned = True
            names.extend(list(m.__dict__.keys()))
            names = list(set(names))

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
                    # usually do not traverse non-module objects
                    if search_attributes:
                        attrs = dir(obj)
                        items = [prefix+'.'+name+'.'+attr for attr in attrs]
                        self.name_list.extend(items)
                self.name_list.append(prefix+'.'+name)
            del m
            # record that we have visited this module
            traversed_mod_names.append(prefix)

    def dynamic_traverse(self,mod,query,search_attributes=False):
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
            names = [] 
            try:
                dir_ = dir(m)
                names.extend[dir_]
            except:
                if not self.warned:
                    print('Packsearch:warning:some objects were corrupted by their author, and they are not traversable.')
                    self.warned = True
            names.extend(list(m.__dict__.keys()))
            names = list(set(names))

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
                    if search_attributes:
                        attrs = dir(obj)
                        items = [prefix+'.'+name+'.'+attr for attr in attrs]
                        for itm in items:
                            if query in itm.lower():
                                yield itm
                        del items
                    del obj
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
    def __call__(self,name):
        return self.search(name)


def packsearch(module_or_str,str_or_module,verbose=False):
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
    for r in p.dynamic_traverse(mod,name.lower(),search_attributes=verbose):
        print(r)
        res_counter += 1
    if res_counter == 0:
        print('Packsearch: no result found')
    else:
        print('Packsearch: search done,',res_counter,'results found.')




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

def whereis(module_or_string,open_gui=True):
    '''
    find the source file location of a module
    arguments:
        module_or_string: target module object, or it's string path like `torch.nn`
        open_gui: open the folder with default window-manager.
    returns:
        module file name, or None

    '''
    mors = module_or_string
    mod = None
    if isinstance(module_or_string,module_type):
        mod = module_or_string
    elif ((not isinstance(module_or_string,str)) and
     hasattr(module_or_string,'__module__')):
        module_or_string = module_or_string.__module__
    
    if isinstance(module_or_string,str):
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
         
    if not hasattr(mod,'__file__'):
        print('TorchFun:error:invalid arguement',mors)
        return None

    fname = mod.__file__
    dirname = os.path.dirname(mod.__file__)
    print(fname)
    if open_gui:
        omini_open(dirname)

    return fname
    
def tf_session(allow_growth=True):
    '''Used to create tensorflow session that does not stupidly and unresonably consume all gpu-memeory.
    returns:
        a tensorflow session consuming dynamic gpu memory.'''
    try:
        tensorflow = importlib.import_module('tensorflow')
    except Exception as e:
        print('tensorflow not installed, cannot provide tensorflow session instance.')
        raise e
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    return tensorflow.Session(config=config)


class Options(object):
    '''A simple yet effective option class for debugging use.
    key features: you can set attributes to it directly.
    like:
            o = Options()
            o.what=1
            o.hahah=123

    '''
    def __init__(self):
        super().__init__()
    def __setattr__(self,name,value):
        if hasattr(self,name):
            super().__setattr__(name,value)
        else:
            self.__dict__[name]=value
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        contents=['\t%s:%s'%(name,str(self.__dict__[name])) for name in self.__dict__]
        contents.insert(0,'Options containing:')
        return os.linesep.join(contents)

def options(*args,**kws):
    '''warpping class for Options. this function returns an option object with attributes
    set according to the input key-value arguments. 
    please refer to Option class for more information'''
    if len(args)!=0:
        print('warning:options must be keywords pairs, not anoymous value list')
    o = Options()
    o.__dict__.update(kws)
    return o

def documentation(search=None):
    ''' help documentation on Torchfun
    Argument:
        search: give None to go to the latest doc site
                give string or object to search the object
    '''
    index_page = 'https://sorenchiron.github.io/torchfun/genindex.html'
    search_page = 'https://sorenchiron.github.io/torchfun/search.html?q='
    if search is None:
        omini_open(index_page)
    else:
        if not isinstance(search,str):
            search = search.__name__
            search_url = search_page+search
            omini_open(search_url)
