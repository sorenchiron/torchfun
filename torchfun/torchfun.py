# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 

from __future__ import division,print_function
import numpy as np
import torch
import io,os,importlib,sys
from tqdm import tqdm
import types
from .types import argparse_list_type,list_of_int,list_of_float,argparse_bool_type
from .ui import *
from .utils import printf
from .utils import print_verbose
from .utils import safe_open
from .utils import sort_args
from .utils import omini_open
from .utils import force_exist

t = torch

def to_numpy(tensor,keep_dim=False):
    '''convert a NCHW tensor to NHWC numpy array.
    if the input tensor has only one image,
    then the N dimension will be deleted.

    if the input is not 4-dimensional, 
    the tensor will be simply cat to np array
    '''
    if isinstance(tensor,np.ndarray):
        return tensor 
    elif isinstance(tensor,torch.Tensor):
        tensor = tensor.cpu().detach().numpy()
        dims = len(tensor.shape)
        if dims == 4:
            n,c,h,w = tensor.shape
            tensor = tensor.transpose(0,2,3,1)
            if n==1 and not keep_dim:
                tensor = tensor.squeeze(0)
        elif dims == 3:
            tensor = tensor.transpose(1,2,0)
        return tensor

    else:
        raise Exception('unknown input type:%s'%(tensor.__class__.__name__))


def dtype(obj):
    '''
    return the data type of:

        a model

        a tensor

        or, the type() of anything else

    '''
    dtype = None
    if isinstance(obj,torch.nn.Module):
        dtype = obj.parameters().send(None).dtype
        # get param generator, then used send() to extract the first weight tensor
    elif isinstance(obj,torch.Tensor):
        dtype = obj.dtype
    else:
        dtype = type(obj)
        warn('data type of object',obj,dtype,'is not torch data-type')
    return dtype


def imread(fname,out_range=(0,1),dtype=torch.float):
    '''read jpg/png/gif/bmp/tiff... image file, and cat to tensor
    function based on imageio.

    args:
        
        fname: string of the file path
        
        out_range: tuple, output pixel value range.
        
        dtype: torch datatype

    Notice:
        the returned image is 1xCxHxW (NCHW).
    '''
    import imageio
    rmin,rmax=out_range
    img = imageio.imread(fname)
    img = (img/255)*(rmax-rmin)+rmin
    img = torch.tensor(img,dtype=dtype)
    if len(img.shape) == 3:
        img.transpose_(2,1)
        img.transpose_(0,1)
    else:
        img.unsqueeze_(0)
    img.unsqueeze_(0)
    return img

def imsave(img_or_dest,dest_or_img):
    '''save torch tensor image(s) or numpy image(s).

    img: can be numpy image, numpy image batch. 

        or single torch-tensor image

        or torch-tensor-image-batch.

        or list/tuple of numpy images

        or list/tuple of pytorch-tensor images

    Notice: images must have non-zero channels, even for grey-scale images (1-channel images).
    
    behaviour: images will be concatenated into a long single image and save into destination file-name.
            the concating will be taken horizontally.
    
    '''
    img,dest = sort_args((img_or_dest,dest_or_img),((torch.Tensor,np.ndarray,tuple,list),str))

    if isinstance(img,(tuple,list)):
        if len(img)==0:
            raise Exception('No images contained in the input list/tuple')
        if isinstance(img[0],torch.Tensor):
            libcat = torch.cat
            libstack = torch.stack
        elif isinstance(img[0],np.ndarray):
            libcat = np.concatenate
            libstack = np.stack
        else:
            raise Exception('input datatype not understood %s' % str(type(img[0])))
        ## make a batch
        if len(img[0].shape)==4:
            img = libcat(img,0)
        else:
            img = libstack(img)

    if isinstance(img,torch.Tensor):
        dimensions = len(img.shape)
        if dimensions == 4:
            # concatenate into a long image
            img = torch.cat([i for i in img],2)
        dimensions = len(img.shape)
        if dimensions == 3: #make channel last image
            img = img.cpu().numpy().transpose(1,2,0)
    img = _force_image_range(img,out_range=(0,1))
    if img is None:
        return
    import imageio
    imageio.imsave(dest,img)
imwrite = imsave

def _force_image_range(npimg,out_range=(0,1),verbose=True):
    '''input must be numpy image. input tensors will be cat to numpy arrray
    This function automatically detects the domain of the input img batch,
    and force the input to be between the out_range.

    args:

        out_range: (min,max)

    Notice:
        if work under verbose=True, images with suspicious domain will return None
        in order to issue instant debugging.
        if not, the image will be forced into the output range.
        '''
    max_intensity = npimg.max()
    min_intensity = npimg.min()
    if min_intensity>=0 and max_intensity>1 and max_intensity<=255:
        # 0 - 255
        npimg = npimg/255
        print_verbose('TorchFun:imshow:Guessed pixel value range:0~255',verbose=verbose)
    elif min_intensity<0 and min_intensity>=-0.5 and max_intensity>0 and max_intensity <=0.5:
        # -0.5 - 0.5
        npimg += 0.5
        print_verbose('TorchFun:imshow:Guessed pixel value range:-0.5~0.5',verbose=verbose)
    elif min_intensity>=0 and max_intensity<=1:
        print_verbose('TorchFun:imshow:Guessed pixel value range:0~1',verbose=verbose)
    elif min_intensity<-0.5 and min_intensity>=-1 and max_intensity>0.5 and max_intensity <=1:
        # -1 - 1
        npimg = npimg/2
        npimg += 0.5
        print_verbose('TorchFun:imshow:Guessed pixel value range:-1~1',verbose=verbose)
    elif min_intensity>=-1 and max_intensity<=1:
        npimg = npimg/2
        npimg += 0.5
        print_verbose('TorchFun:imshow:Cannot speculate the value-range of this image!, Trying -1~1.',verbose=verbose)
    else:
        msg='TorchFun:imshow:Cannot speculate the value-range of this image. Please normalize the image manually before using imshow.'
        print_verbose(msg,verbose=verbose)
        if verbose:
            return None
        else:# force range to be 0-1
            minv,maxv = npimg.min(),npimg.max()
            npimg = (npimg-minv)/(maxv-minv+1e-11)
    outmin,outmax = out_range
    npimg = npimg*(outmax-outmin) + outmin
    return npimg

def imshow(x,title=None,auto_close=True,cols=8,backend=None):
    '''only deal with torch channel-first image batch,
    
    title: add title to plot. (Default None)
        title can be string, or any string-able object.
   
    auto_close: (default True) 
        Close the pyplot session afterwards. 
        Clean the environment just like you had 
        never used matplotlib here.
        if set to False, the plot will remain in the memory for further drawings.
    
    cols: columns(default 8)
        the width of the output grid, aka, number of images per row.
   
    backend: None to use default gui. options are:
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
    
    If your are using Unix-like systems such as MacOSX, you can create ~/.matplotlib/matplotlibrc and add a line: `backend:TkAgg` to it. 

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
    if isinstance(x,(list,tuple)):
        x = torch.cat(x,0)
    elif isinstance(x,np.ndarray):
        x = x.transpose((2,0,1))
        x = torch.tensor(x).unsqueeze(0)
    if x.requires_grad:
        x = x.detach()
    x = x.cpu()
    shapes = x.shape
    if len(shapes)==3:
        x = t.unsqueeze(x,dim=0)
    grid = torchvision.utils.make_grid(x,nrow=cols)
    gridnp = grid.numpy()
    gridnp = _force_image_range(gridnp,out_range=(0,1))
    if gridnp is None:
        return

    import matplotlib
    if backend:
        print('using explicitly designated backend:',backend)
        matplotlib.use(backend,force=True)
    else:
        matplotlib.use('TkAgg')
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

def imresize(tensor_or_size,size_or_tensor):
    '''stretch pytorch image NCHW, into given shape

    arguments

            tensor: NCHW tensor

            size: tuple or list of [height,width], or a scale factor
    '''
    x,siz = sort_args([tensor_or_size,size_or_tensor],[torch.Tensor,(tuple,list,int,float)])
    if isinstance(siz,(int,float)):
        return torch.nn.functional.interpolate(x,scale_factor=siz,mod='bilinear',align_corners=False)
    else:
        h,w = siz
        return torch.nn.functional.interpolate(x,size=(h,w),mode='bilinear',align_corners=False)

def imcrop_center(tensor_or_size,size_or_tensor):
    '''crop a center patch from pytorch image NCHW

    arguments

            tensor: NCHW tensor

            size: tuple or list of [height,width], or a scale factor
                    given a tuple of height,width, this returns a scaled patch of the 
                    largest center crop.
                    given a factor of scale, this will return a scaled square maximum center crop
    '''
    x,siz = sort_args([tensor_or_size,size_or_tensor],[torch.Tensor,(tuple,list,int,float)])
    *_,h,w = x.shape
    if isinstance(siz,(float,int)):
        out_h = out_w = min(h,w)*siz
        siz = (out_h,out_w)
    else:
        out_h,out_w = siz
    imgratio = h/w
    cropratio = out_h/out_w
    if cropratio>=imgratio:# height filled first
        H = h
        W = int(H / cropratio)
        Hstart = 0
        Wstart = (w - W)//2
    else:
        W = w
        H = W * cropratio
        Wstart = 0
        Hstart = (h-H)//2
    Hend = Hstart+H
    Wend = Wstart+W
    Hstart,Hend,Wstart,Wend = int(Hstart),int(Hend),int(Wstart),int(Wend)
    patch = x[:,:,Hstart:Hend,Wstart:Wend]
    with torch.no_grad():
        return torch.nn.functional.interpolate(patch,size=siz,mode='bilinear',align_corners=False)

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

    Behaviour: the loaded state-dict will be transformed to the same device as model,
            so that torch won't complain about CUDA-memory-insufficient when you just want to load
            weights from disk directly to cpu-model.
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

    model,source = sort_args((a,b),(torch.nn.Module,(str,io.TextIOWrapper)))
    if isinstance(source,io.TextIOWrapper):
        # file handle input
        source = io.BytesIO(f.read())
    model_device = model.state_dict()[list(model.state_dict().keys())[0]].device
    weights = torch.load(source,map_location=model_device)
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
    elif isinstance(model_or_dict_or_param,types.GeneratorType):
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
    elif isinstance(m,types.GeneratorType):
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

def vectorize_parameter(model_or_statdict_or_param):
    '''return the vectorized form of all variables.
    This is used to detect chaotic changes of weights.

    arguements:

    module_or_statdict_or_param: torch.nn.module, 

                    or model.state_dict(), 

                    or model.parameters().
   
    '''
    m = model_or_statdict_or_param
    means = []
    params = []
    if isinstance(m,torch.nn.Module):
        params = m.parameters()
    elif isinstance(m,dict):
        params = [m[k] for k in m]
    elif isinstance(m,types.GeneratorType):
        params = parameters
    elif isinstance(m,torch.Tensor):
        params = [m]
    else:
        print('TorchFun:vectorize_parameter:','input type not support:',type(m))
    ps=[]
    for p in params:
        ps.append(p.view(-1,1))

    return torch.cat(ps,0)

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

def show(net,input_shape=(1,3,32,32),logdir='tensorboard',port=8888):
    '''print the network architecture on web-browser, using tensorboardX and tensorboard.
    tensoboard must be install to use this tool.
    this tool will create a noise data according to given input_shape,
    and feed it directly to net, in order to probe its structure.
    network strctures descriptions will be written to logdir.
    a tensorboard daemon will be launched to read the logdir and start a web server
    on given port.

    Notice: 

        input shape must be NCHW, following pytorch style.

        This program overwrites the system argument lists (sys.argv)
    '''
    try:
        from tensorboard.main import run_main
    except Exception as e:
        raise Exception(str(e)+'\nError importing tensorboard. Maybe your tensorboard is not installed correctly.\
            usually, tensorboard should come with tensorflow. stand-alone tensorboard packages are not stable enough.')
    import tensorboardX as tb
    import sys,shutil,re
    net,input_shape,logdir,port = sort_args([net,input_shape,logdir,port],[torch.nn.Module,(tuple,list),str,int])
    shutil.rmtree(logdir,ignore_errors=True)
    imgs = torch.rand(*input_shape)
    w = tb.SummaryWriter(logdir)
    try:
        w.add_graph(net,imgs)
    except Exception as e:
        raise Exception(str(e)+'\nYour network has problems dealing with input data.\
         It is usually due to wrong input shape or problematic network implementation.\
         Please check your network code for more information.')
    finally:
        w.close()
    args = ['tensorboard','--logdir',logdir,'--port',str(port),'--host','127.0.0.1']
    sys.argv = args
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    print('You may have to delete',logdir,'folder manully.')
    omini_open('http://127.0.0.1:%d'%port)
    try:
        run_main()
    except Exception as e:
        print(e)

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
        if isinstance(module_object,types.ModuleType):
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
        if type(mod) is not types.ModuleType:
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
                if type(obj) is types.ModuleType:
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
        if type(mod) is not types.ModuleType:
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
                if type(obj) is types.ModuleType:
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
    mod,name = sort_args([types.ModuleType,str],[module_or_str,str_or_module])
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

#def force_exist(dirname,verbose=True):
#    '''force a directory to exist.
#    force_exist can automatically create directory with any depth.
#    Arguements:
#        dirname: path of the desired directory
#        verbose: print every directory creation. default True.
#    Usage:
#        force_exist('a/b/c/d/e/f')
#        force_exist('a/b/c/d/e/f',verbose=False)
#    '''
#    
#    if dirname == '' or dirname == '.':
#        return True
#    top = os.path.dirname(dirname)
#    force_exist(top)
#    if not os.path.exists(dirname):
#        if verbose:
#            print('creating',dirname)
#        os.makedirs(dirname)
#        return False
#    else:
#        return True

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
    if isinstance(module_or_string,types.ModuleType):
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

class Options(dict):
    '''A simple yet effective option class for debugging use.
    key features: you can set attributes to it directly.
    like:
            o = Options()
            o.what=1
            o.hahah=123

    '''
    def __init__(self,*args,**kws):
        super().__init__(*args,**kws)
    def __setattr__(self,name,value):
        self[name]=value
    def __getattr__(self,name):
        if name in self:
            return self[name]
        else:
            return self[name]
    def __getitem__(self,name):
        if name in self:
            return super().__getitem__(name)
        else:
            self[name] = Options()
            return self[name]
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        contents=['\t%s:%s'%(name,str(self[name])) for name in self]
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

class Unimodel(torch.nn.Module):
    '''this class is used to gather your multiple models, so that
    you can save/load them together.

    usage:

        unimodel = Unimodel(resnet1,resnet2,resnet3)

        unimodel.save('xxxx.pth')

        unimodel.load('ssss.pth')

        resnet1 = unimodel.resnet1

        ...

    '''
    def __init__(self,*models,**named_models):
        super().__init__()
        for i,m in enumerate(models):
            self.__setattr__('model%d'%i,m)
        for name in named_models:
            self.__setattr__(name,named_models[name])
    def save(self,path):
        '''same as torchfun save()'''
        save(self,path)
    def load(self,path):
        '''same as torchfun load()'''
        load(self,path)


















# try not to override default bool type
# TODO: change bool, or somehow it will 
# overwrite all occurrence of bool above
from .types import Bool,TorchEasyList
List = TorchEasyList

# try not to override default list type
##################### Last function ####################


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
        else:
            search = search
        search_url = search_page+search
        omini_open(search_url)

