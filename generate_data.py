import numpy as np
import numpy.linalg as la

from scipy.stats import wishart, beta , gengamma


import scipy
import math
from scipy.linalg import sqrtm



def t_wishart_rvs(scale,n,df,size):
    p,_=scale.shape
    L = la.cholesky(scale)
    ws = wishart.rvs(scale=np.eye(p),df=n,size=size)
    qs = beta.rvs(a=df/2,b=n*p/2,size=size)
    vec = df*(1/qs-1)/np.trace(ws,axis1=-1,axis2=-2)
    return np.einsum('...,...ij->...ij',vec,L@ws@L.T) 



def kotz_wishart_rvs(scale,n,a,b,r,size):
    p,_ = scale.shape
    L = la.cholesky(scale)
    ws = wishart.rvs(scale=np.eye(p),df=n,size=size)
    qs = math.pow(r,-1/b)*gengamma.rvs(a=(a-1+0.5*n*p)/b,c=b,size=size)
    vec = qs/np.trace(ws,axis1=-1,axis2=-2)
    return np.einsum('...,...ij->...ij',vec,L@ws@L.T)