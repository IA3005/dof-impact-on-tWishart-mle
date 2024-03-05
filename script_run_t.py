import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy.stats import wishart, multivariate_normal

import pymanopt
from pymanopt import Problem
from pymanopt.optimizers import SteepestDescent, ConjugateGradient

from manifold import SPD
from ellipt_wishart_estim import wishart_est, t_wish_cost, t_wish_egrad
from generate_data import t_wishart_rvs


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


def wishart_t_simulate_data(p,n,kmax,df,nIt):
    # manifold
    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p,alpha,beta)
    # random Wishart parameter
    R = manifold.random_point(cond=100)
    # simulated data with maximum number of samples
    Ss = t_wishart_rvs(scale=R,n=n,df=df,size=(nIt,kmax))
    return R, Ss, manifold

def wishart_t_perform_estim(R,Ss,n,ks,df,manifold):
    nIt,_,p,_ = Ss.shape
    nK = np.size(ks)
    # init error measures
    err_wish = np.zeros((nIt,nK))
    err_wish_t = np.zeros((nIt,nK))
    # Monte-Carlo estimation loop
    for kix in range(nK):
        k = ks[kix]
        print(f"k={k}")
        for it in tqdm(range(nIt)):
            # select data
            S = Ss[it][:k]
            # compute Wishart estimator
            wish_est = wishart_est(S,n)
            # compute Wishart t estimator
            wish_t_est = wishart_t_est(S,n,df,manifold)
            # compute errors
            err_wish[it,kix] = manifold.dist(R,wish_est)**2
            err_wish_t[it,kix] = manifold.dist(R,wish_t_est)**2
    return err_wish, err_wish_t

def wishart_t_write_results(ks,err_wish,err_wish_t,df):
    # compute median errors in dB
    err_wish_median = 10*np.log10(np.median(err_wish,axis=0))
    err_wish_t_median = 10*np.log10(np.median(err_wish_t,axis=0))
    # compute quantiles 0.1 and 0.9 errors in dB
    err_wish_qmin = 10*np.log10(np.quantile(err_wish,q=0.05,axis=0))
    err_wish_qmax = 10*np.log10(np.quantile(err_wish,q=0.95,axis=0))
    err_wish_t_qmin = 10*np.log10(np.quantile(err_wish_t,q=0.05,axis=0))
    err_wish_t_qmax = 10*np.log10(np.quantile(err_wish_t,q=0.95,axis=0))

    # plot
    fig, ax = plt.subplots()

    ax.fill_between(ks,err_wish_qmin,err_wish_qmax,alpha=.5,linewidth=0)
    wish_line, = ax.plot(ks,err_wish_median,linewidth=2)
    wish_line.set_label("Wishart estimator")

    ax.fill_between(ks,err_wish_t_qmin,err_wish_t_qmax,alpha=.5,linewidth=0)
    wish_t_line, = ax.plot(ks,err_wish_t_median,linewidth=2)
    wish_t_line.set_label(f"Wishart t df={df} estimator")

    ax.legend()

    ax.set_xscale('log')
    plt.show()
    
    # write in file
    filename = f"result_wish_t_df{df}.txt"
    meth_name = ["wish","wish_t"]
    nM = np.size(meth_name)
    with open(filename,'w') as f:
        f.write("k")
        for mix in range(nM):
            f.write(f",{meth_name[mix]}_median,{meth_name[mix]}_qmin,{meth_name[mix]}_qmax")
        f.write("\n")
        for kix in range(nK):
            f.write(f"{ks[kix]},{err_wish_median[kix]},{err_wish_qmin[kix]},{err_wish_qmax[kix]},{err_wish_t_median[kix]},{err_wish_t_qmin[kix]},{err_wish_t_qmax[kix]}\n")


if __name__ == '__main__':
    
    # simulation parameters
    p = 16
    n = 100
    ks = [30,70,100,300,500]
    kmax = np.max(ks)
    nK = np.size(ks)
    # number of Monte-Carlo
    nIt = 200
    # Student degree of freedom
    df = 100
    
    # simulate data and get manifold    
    R, Ss, manifold = wishart_t_simulate_data(p,n,kmax,df,nIt)
    
    # perform estimation
    err_wish, err_wish_t = wishart_t_perform_estim(R,Ss,n,ks,df,manifold)

    # analyze results    
    wishart_t_write_results(ks,err_wish,err_wish_t,df)   