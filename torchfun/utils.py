# LICENSE: MIT
# Author: CHEN Si Yu.
# Date: 2018
# Appended constrains: If the content in this project is used in your source-codes, this author's name must be cited at the begining of your source-code. 


import time as time_pkg
#from .torchfun import force_exist
import os


__doc__='utilities for pytorch experiment. Non-mathematical tools.'


def printf(format_str,*args):
    '''works like C printf.
    '''
    print(format_str % tuple(args))

def print_verbose(*args,verbose=True):
    '''programmable print function. with an option to control
    if the inputs are really printed.
    used to control verbose levels of a function.
    verbose: default True.
    '''
    if verbose:
        print(*args)

def safe_open(*args,encoding=None,return_encoding=False,verbose=True,**kws):
    '''automatically determine the encoding of the file.
    so that there will not be so many stupid encoding errors occuring during coding.

    Note: the file needs to be fully loaded into RAM to examine the encodings. 
        OOM(Out Of Memory) exception may be raised when encountering large text files.'''
    encs = ['ascii','utf_8','gb2312','gbk','utf_16','utf_16_be','utf_16_le','utf_7','gb18030','big5','big5hkscs','cp037','cp424','cp437','cp500','cp737','cp775','cp850','cp852','cp855','cp856','cp857','cp860','cp861','cp862','cp863','cp864','cp865','cp866','cp869','cp874','cp875','cp932','cp949','cp950','cp1006','cp1026','cp1140','cp1250','cp1251','cp1252','cp1253','cp1254','cp1255','cp1256','cp1257','cp1258','euc_jp','euc_jis_2004','euc_jisx0213','euc_kr','hz','iso2022_jp','iso2022_jp_1','iso2022_jp_2','iso2022_jp_2004','iso2022_jp_3','iso2022_jp_ext','iso2022_kr','latin_1','iso8859_2','iso8859_3','iso8859_4','iso8859_5','iso8859_6','iso8859_7','iso8859_8','iso8859_9','iso8859_10','iso8859_13','iso8859_14','iso8859_15','johab','koi8_r','koi8_u','mac_cyrillic','mac_greek','mac_iceland','mac_latin2','mac_roman','mac_turkish','ptcp154','shift_jis','shift_jis_2004','shift_jisx0213',]
    if encoding not in encs:
        encs.append(encoding)
    for enc in encs:
        try:
            f = open(*args,encoding=enc,**kws)
            f.readline()
            f.read()
            f.close()
        except:
            print_verbose('safe_open:','trying other encodings.',verbose=verbose)
        else:
            f = open(*args,encoding=enc,**kws)
            if return_encoding:
                return f,enc
            else:
                return f
    print_verbose('tried encodings:',','.join(encs),'. None of them are supported.',verbose=verbose)
    if return_encoding:
        return None,None
    else:
        return None


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
    if (
            (type(args_or_types[0]) is type) # one type
            or 
            ( # type list
                isinstance(args_or_types[0],(list,tuple)) and 
                (type(args_or_types[0][0]) is type)
            )
        ):
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
    else:# linux
        if path[:4] == 'http': # browser
            command = 'firefox'
        else:
            command = 'xdg-open'
        subprocess.Popen([command, path])

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

def record_experiment(exp_dir='not-specified',record_top_dir='records',logfilename='record.txt',comment=''):
    '''Arguments:
        * exp_dir : directory of this experiment output files, 
                    recorded so that it would be easier for you to find the outcome files 
                    of this experiment later.
        * record_top_dir : create a dir to save all kinds of record logs. default(records). 
                    set to empty string('') to save all record in the current dir.
        * logfilename : filename of this log, usually `train_record` `evaluation_record` etc.
        * comment :string comment added to log paragraph. default is empty string.
    '''
    from sys import argv
    try:
        force_exist(record_top_dir)
        logfilepath = os.path.join(record_top_dir,logfilename)
        if not os.path.exists(logfilepath):
            f = open(logfilepath,'w')
            f.write('科研实验开始::\n')
            f.close()

        with open(logfilepath,'a') as f:
            t = time_pkg.localtime()
            timestr = 'y:%d m:%d d:%d %d:%d:%d' %(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)
            command = ' '.join(argv)
            template = \
            '''
-----------------
时间：%s
    命令：%s
    目录名：%s
    效果与结果备注：%s
    原因分析：
    示例：

            ''' % (timestr,command,exp_dir,comment)
            f.write(template)
    except Exception as e:
        print('record_experiment:智能实验助理记录出错了。',e)
    else:
        print('record_experiment:智能实验助理已经帮您记录此次试验。')

def cpu_memory(print_on_screen=True):
    '''total CPU memory usage of current python session.
    returns: (RSS,VMS) In bytes!
            RSS is resident set size, 
            VMS is virtual memory size.
    Notice: 
            return values are in bytes.
            printed values are in MBs.
    '''
    import psutil
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss
    vms = p.memory_info().vms
    if print_on_screen:
        rssmb = rss/1024/1024
        vmsmb = vms/1024/1024
        print('RAM:%.2fMB Disk:%.2fMB'%(rssmb,vmsmb))
    return rss,vms


def change_suffix(fpath,new_suffix):
    '''
    change the suffix of fpath to new_suffix.
    if the original path fpath has no suffix, then new_suffix will be
    directly applied at the end of the fpath string, joint by separator `.`

    In : change_suffix('xxx/ddd/www.x','aa')
    Out: 'xxx/ddd/www.aa'

    In : change_suffix('xxx/ddd/www','aa')
    Out: 'xxx/ddd/www.aa'
    '''
    fname = os.path.basename(fpath)
    if '.' in fname:
        # xxx/xxx/xxx.suffix
        path_with_name = '.'.join(fpath.split('.')[:-1])
    else:
        path_with_name = fpath
    new_path = path_with_name+'.'+new_suffix
    return os.path.normpath(new_path) 


def change_fname(fpath,new_fname):
    '''
    change the name of file in the given fpath.
    the suffix (if exists) separated by `.` will be kept.

    
    In : change_fname('xxx/ddd/www','aa')
    Out: 'xxx\\ddd\\aa'

    In : change_fname('xxx/ddd/www.txt','aa')
    Out: 'xxx\\ddd\\aa.txt'    
    '''
    fname = os.path.basename(fpath)
    if '.' in fname:
        # xxx/xxx/xxx.suffix
        suffix = fpath.split('.')[-1]
    else:
        suffix = None
    
    if suffix is not None:
        fname_suffix = new_fname+'.'+suffix
    else:
        fname_suffix = new_fname

    topdir = os.path.dirname(fpath)
    new_path = os.path.join(topdir,fname_suffix)
    return os.path.normpath(new_path)

_named_time_table={}
_anonymous_time = [None]
def time(message=None,name=None,named_time_table=_named_time_table,anonymous_time=_anonymous_time):
    '''show running time
    dict-key locating costs 1e-6 second = 1 micro-second
    context switching costs 5e-5 second.

    Usage 1:
        tf.time()
        ...
        elapsed = tf.time()

    Usage 2:
        tf.time()
        ...
        tf.time('elapsed:')
        out: elapsed: 0.02 sec

    Usage 3:
        tf.time(name='clock1')
        ...
        tf.time('elapsed',name='clock1')
        out: elapsed 0.02 sec
        '''
    enter_time = time_pkg.time()
    assign_exit_time_to_anonymous = False
    assign_exit_time_to_named = False
    if name is None:
        if anonymous_time[0] is None:
            assign_exit_time_to_anonymous = True # assign time at the end
        else:
            elapsed = enter_time - anonymous_time[0]
            anonymous_time[0] = None
            if message is not None:
                print(message,elapsed,'sec')
            return elapsed
    else:
        if name not in named_time_table:
            assign_exit_time_to_named = True
        else:
            elapsed = enter_time - named_time_table[name]
            del named_time_table[name]
            if message is not None:
                print(message,elapsed,'sec')
            return elapsed

    if assign_exit_time_to_anonymous:
        anonymous_time[0] = time_pkg.time()
        return anonymous_time[0]

    if assign_exit_time_to_named:
        named_time_table[name] = time_pkg.time()
        return named_time_table[name]

def reset_timer(named_time_table=_named_time_table,anonymous_time=_anonymous_time):
    named_time_table.clear()
    anonymous_time.clear()







