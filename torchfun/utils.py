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























