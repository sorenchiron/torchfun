import time as time_pkg
from .torchfun import force_exist
import os


__doc__='utilities for pytorch experiment. Non-mathematical tools.'

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







