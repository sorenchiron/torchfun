
__doc__='''tools for user-interface interactions. 
(mostly for console-debugging)'''

def info(*args):
    header = '[info]\t'
    print(header,*args)
    
def error(*args):
    header = '[error]\t'
    print(header,*args)
    exit(-1)

def warning(*args):
    header = '[warn]\t'
    print(header,*args)
warn = warning

def wait(*args):
    header = '[wait]\t'
    print(header,*args)
    print('\t\tpress Enter to continue')
    input()

def input_or(prompt,default='y'):
    v = input(prompt+'(default:%s):'%(str(default)))
    v = v or default
    dtype = type(default)
    return dtype(v)
