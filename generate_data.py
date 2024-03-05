import numpy as np
import numpy.linalg as la

from scipy.stats import wishart, beta , gengamma,ortho_group, norm


import scipy
import math
from scipy.linalg import sqrtm


def generate_random_sdp(p,cond):
    d = np.zeros(p)
    d[:p-2] = uniform.rvs(loc=1/np.sqrt(cond),scale=np.sqrt(cond)-1/np.sqrt(cond),size=p-2)
    d[p-2] = 1/np.sqrt(cond)
    d[p-1] = np.sqrt(cond)
    U =ortho_group.rvs(dim=p) #random orthogonal matrix
    return U@np.diag(d)@U.T


def t_wishart_rvs(scale,n,df,size):
    p,_=scale.shape
    L = la.cholesky(scale)
    ws = wishart.rvs(scale=np.eye(p),df=n,size=size)
    qs = beta.rvs(a=df/2,b=n*p/2,size=size)
    vec = df*(1/qs-1)/np.trace(ws,axis1=-1,axis2=-2)
    return np.einsum('...,...ij->...ij',vec,L@ws@L.T) 



