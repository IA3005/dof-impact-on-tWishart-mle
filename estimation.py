import numpy as np
import numpy.linalg as la
from functools import partial
from tqdm import tqdm

import pymanopt
from pymanopt import Problem
from pymanopt.optimizers import SteepestDescent, ConjugateGradient
from manifold import SPD


def ellipt_wish_cost(R,S,n,logh):
    k, p, _ = S.shape
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    return 1/2 * np.log(la.det(R)) - np.sum(logh(a))/n/k


def ellipt_wish_egrad(R,S,n,u):
    k, p, _ = S.shape
    # psi
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    psi = np.einsum('k,kij->ij',u(a),S)
    #
    return la.solve(R,la.solve(R.T,((R  - psi/n/k) /2).T).T)


# student
def t_logh(t,df,dim):
    return -(df+dim)/2*np.log(1+t/df) 

def t_u(t,df,dim):
    return (df+dim)/(df+t)

def t_wish_cost(R,S,n,df):
    p, _ = R.shape
    return ellipt_wish_cost(R,S,n,partial(t_logh,df=df,dim=n*p))

def t_wish_egrad(R,S,n,df):
    p, _ = R.shape
    return ellipt_wish_egrad(R,S,n,partial(t_u,df=df,dim=n*p))

def wishart_t_est(S,n,df,manifold, algo="RCG"):
    @pymanopt.function.numpy(manifold)
    def cost(R):
        return t_wish_cost(R,S,n,df)
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(R):
        return t_wish_egrad(R,S,n,df)
    #
    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient)
    init = np.eye(S.shape[-1])
    assert algo in ["RCG","RSG"],"Wrong Algorithm Name"
    if algo=="RCG": #conjugate gradient
        optimizer = ConjugateGradient(verbosity=0)
    else:
        optimizer = SteepestDescent(verbosity=0)

    return optimizer.run(problem, initial_point=init).point

def mle_RCG(samples,n,df):
    if df==np.inf:
        return np.mean(samples,axis=0)/n
    _,p,_=samples.shape
    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p,alpha,beta)
    return wishart_t_est(samples,n,df,manifold,algo=algo)
