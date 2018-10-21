# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 

from .torchfun import *
from . import transforms
from . import nn
from . import utils
from . import datasets
from . import functional
from .transforms import *
from .nn import *
from .utils import *

import os

name = 'torchfun'

__folder__ = os.path.dirname(__file__)

__ver_fname__ = os.path.join(__folder__,'version')

__version__ = open(__ver_fname__).read()

__all__ = list(locals())

__doc__ = '''`TorchFun` project was initiated long ago and was published in 2018-6-13. 

TorchFun is motivated by the one author loves, who sometimes encounter inconvenience using PyTorch at her work. The author published the project as a python package `TorchFun` on PyPi, so that his beloved could access it whenever and wheresoever needed.

The purposed of TorchFun is to provide functions which are important and convenient, but absent in PyTorch, like some layers and visualization utils.

This project has been undergoing in secret so that the author could credit TorchFun to his beloved as a little supprise of help, one day when this project is well-enriched or gets famous.

Interestingly, The original project name given by the author, is `Torchure`. That is because he was always multi-tasking trivial affairs in school and got scorched, and when he was learning this new framework in hope to help his beloved, he found plenty of issues/missing-functionalities. He felt that this was totally a torture. So, this project was named "Torchure" to satirize the lame PyTorch (you can still found Torchure in PyPi). And, the author hoped, by developing Torchure, his beloved could feel ease even when encountering the crappy-parts of PyTorch.

This history-of-project was appended recently, because his adorable little beloved wants a supprise immediately, or she will keep rolling on the floor.

To c71b05bf46d8772e4488335085a2e7fd.

'''

del_list=[
'os',
'io',
'np',
'torch',
'tqdm',
'division',
'print_function',
'importlib',
'sys']

for del_item in del_list:
    if del_item in __all__:
        __all__.remove(del_item)

del del_list
