import time
def time_counter(func):
    def inner(*args,**kwargs):
        time0 = time.time()
        func(*args,**kwargs)
        time1 = time.time()
        print('>>>time cost:%.4fs'%(time1-time0))
    return inner