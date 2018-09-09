name = 'torchfun'
from .torchfun import flatten,imshow
from .torchfun import *
import os

__folder__ = os.path.dirname(__file__)

__ver_fname__ = os.path.join(__folder__,'version')

__version__ = open(__ver_fname__).read()

__all__ = locals()

del_list=[
'os',
'io',
'np',
'torch',
'tqdm',
'print_function']

for del_item in del_list:
    del __all__[del_item]
