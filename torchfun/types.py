import os
import torch

module_type = type(os)

def _g():
    '''used only for getting generator_type'''
    yield from range(1)
generator_type =  type(_g())

def argparse_list_type(element_type=int):
    '''used to put into argparse add_argument(type=?),
    in order to parse lists like:
    --gpu-ids=1,2,3,4

    usage:
        floatlist = argparse_list_type(float)
        parser.add_argument('--cuda-devs',type=flaotlist,default=[1,2])

    args:
        element_type: type of each element in the list,
                eache element will be cat to this type.

    Notice: Instances must be used, instead of this class itself.
    '''
    def list_type(option_string):
        a = option_string
        a = a.strip().strip('[]()\{\}')
        sep = ' '
        if ',' in a:
            sep = ','
        elems = [i.strip() for i in a.split(sep)]

        return [element_type(i) for i in elems]
    return list_type

list_of_int = argparse_list_type()

list_of_float = argparse_list_type(float)

def argparse_bool_type(option_string):
    '''used to put into argparse add_argument(type=torchfun.bool),
    in order to parse bool switch values like:
    false False true True 0 1'''
    a = option_string.strip().lower()
    if a in ('true','1'):
        return True
    elif a in ('false','0'):
        return False
    else:
        raise Exception('un-parsable argument value %s for bool-type'%option_string)

bool=argparse_bool_type
Bool=argparse_bool_type

class TorchEasyList(list):
    def cat(self,dim=0):
        return torch.cat(self,dim)
    def push_head(self,element):
        self.insert(0,element)
    def push_back(self,element):
        return self.append(element)
    def __add__(self,element):
        if isinstance(element,list):
            return TorchEasyList(super().__add__(element))
        else:
            newlist = TorchEasyList(self)
            newlist.push_head(element)
            return newlist