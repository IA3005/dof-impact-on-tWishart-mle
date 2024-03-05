import numpy as np

def mean_list(l):
    s=0
    for x in l:
        s+=x
    return s/len(l)

def std_list(l):
    s=0
    m= mean_list(l)
    for x in l:
        s+=(x-m)**2
    return np.sqrt(s/len(l))
