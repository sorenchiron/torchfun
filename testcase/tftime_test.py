import torchfun as tf

import time
import time as time_pkg





def usage1():
    print('testing usage1')
    tftime()
    time.sleep(1.1)
    elapsed = tftime()
    print('elapsed',elapsed)

def usage2():
    print('testing usage2')
    tftime()
    time.sleep(1.2)
    tftime('elapsed:')

def usage3():
    print('testing usage3')
    tftime(name='clock1')
    time.sleep(1.3)
    elapsed = tftime(name='clock1')
    print('elapsed',elapsed)

def usage4():
    print('testing usage4')
    tftime(name='clock1')
    time.sleep(1.4)
    tftime('elapsed',name='clock1')


usage1()
usage2()
usage3()
usage4()