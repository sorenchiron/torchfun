import logging
from logging import *
__doc__='''tools for user-interface interactions. 
(mostly for console-debugging)'''

# def info(*args):
#     header = '[info]\t'
#     print(header,*args)
# 
# def warning(*args):
#     header = '[warn]\t'
#     print(header,*args)

def error(*args):
    '''
    throw error and exit the program
    '''
    header = '[error]\t'
    print(header,*args)
    exit(-1)
    
def wait(*args):
    '''
    wait and show a message, until a key is stroked.
    '''
    header = '[wait]\t'
    print(header,*args)
    print('\t\tpress Enter to continue')
    input()

info = logging.info
warn = warning = logging.warning
debug = logging.debug
basicConfig = logging.basicConfig

def input_or(prompt,default='y'):
    '''
    get input from command-line,
    if  the input is ommited by an enter key, then the default values is returned.
    '''
    v = input(prompt+'(default:%s):'%(str(default)))
    v = v or default
    dtype = type(default)
    return dtype(v)
