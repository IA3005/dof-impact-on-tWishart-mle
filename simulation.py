import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm

import sklearn
from sklearn.metrics import confusion_matrix
from scipy.linalg import sqrtm,pinv
from manifold import SPD
from generate_data import  t_wishart_rvs, generate_random_sdp
from estimation import mle_RCG

if __name__ == '__main__':

    n = 100
    p= 50
    cond = 10000
    K = 1000
    df0 = 5
    MC= 50
    dfs = [1,3,5,10,50,100,200,500,1000,5000,1e4,1e5,1e7,1e8,1e10,1e12]
    
    center = generate_random_sdp(p,cond)
    
    print("Generate samples:")
    if n<p:
        vec_samples = multivariate_t(shape=np.kron(np.eye(n),center),df=df0).rvs(K*MC)
        X_samples = np.zeros((K*MC,p,n))
        for k in range(K*MC):
            X_samples[k] = vec_samples[k].reshape((n,p)).T #(size,p,n)
        #X_samples = vec_samples.reshape((n,p,K*MC)).T
        samples = np.einsum("kil,kjl->kij",X_samples,X_samples)
    else:
        if K*MC<1e4:
            samples = t_wishart_rvs(center,n,df0,K*MC)
        else:
            samples = []
            for _ in tqdm(range(K*MC)):
                samples.append(t_wishart_rvs(center,n,df0,1)[0])
    
    centers_estim = {df: None for df in dfs}
    
    samples_traces = [[np.trace(samples[j*K:(j+1)*K][i]) for i in range(K)] for j in range(MC)]
    ##samples_traces[j][i] = the trace of ith sample of the jth monte carlo set
    samples_inverse_traces = [[np.trace(pinvh(samples[j*K:(j+1)*K][i])) for i in range(K)] for j in range(MC)]
    samples_eigenvals=[[np.linalg.eigvalsh(samples[j*K:(j+1)*K][i]) for i in range(K)] for j in range(MC)]
    ##samples_eigenvals[j][i] = array of eigenvalues of the ith sample of the jth monte carlo set
    samples_max_eigenval=[[max(samples_eigenvals[j][i]) for i in range(K)] for j in range(MC)]
    ##samples_max_eigenval[j][i]= largest eigenvalue of the ith sample of the jth monte carlo set
    samples_min_eigenval=[[min(samples_eigenvals[j][i]) for i in range(K)] for j in range(MC)]
    samples_conds = [[np.linalg.cond(samples[j*K:(j+1)*K][i]) for i in range(K)] for j in range(MC)]
    samples_conds_mean = np.mean(np.asarray(samples_conds))#mean of all samples (from all monte carlo sets)

    print("Compute MLE estimators with RCG:")
    for df in tqdm(dfs):
        centers_estim[df]=Parallel(n_jobs=-1)(delayed(mle_RCG)(np.asarray(samples[j*K:(j+1)*K]).reshape((K,p,p)),n,df) for j in (range(MC)))
    
    #centers bounds
    centers_traces = {df: [] for df in dfs}
    ##centers_traces[df][j]=trace of the center estimated with df on the jth monte carlo set
    centers_inverse_traces = {df:[] for df in dfs}
    centers_eigenvals = {df:[] for df in dfs}
    for df in dfs:
        for j in range(MC):
            center = centers_estim[df][j]
            centers_traces[df].append(np.trace(center))
            centers_inverse_traces[df].append(np.trace(pinvh(center)))
            centers_eigenvals[df].append(np.linalg.eigvalsh(center))
            
    errors_estim = {df:[np.linalg.norm(centers_estim[df][j]-center) for j in range(MC)] for df in dfs}
    errors_estim_means = [mean_list(errors_estim[df]) for df in dfs]
    errors_estim_stds = [std_list(errors_estim[df]) for df in dfs]
    
    ##########################################################################
    ##figure 1: L2 norm between estimated centers and real center
    ##########################################################################
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    y1= plt.errorbar(dfs,errors_estim_means,errors_estim_stds)
    
    df_hyp_1 = np.asarray([n*p*(max(samples_traces[j])*max(samples_inverse_traces[j])/p-1) for j in range(MC)]) 
    ymax=max(errors_estim_means)+max(errors_estim_stds)
    plt.vlines(np.mean(df_hyp_1),0,ymax,'r',label="hyp on traces")
    plt.vlines(np.mean(df_hyp_1)-np.std(df_hyp_1),0,ymax,'r','dashed')
    plt.vlines(np.mean(df_hyp_1)+np.std(df_hyp_1),0,ymax,'r','dashed')
 
    df_hyp_2 = np.asarray([n*p*(p*(max(samples_max_eigenval[j])/min(samples_min_eigenval[j]))-1) for j in range(MC)]) 
    plt.vlines(np.mean(df_hyp_2),0,ymax,'g',label="hyp on eigenvalues")
    plt.vlines(np.mean(df_hyp_2)-np.std(df_hyp_2),0,ymax,'g','dashed')
    plt.vlines(np.mean(df_hyp_2)+np.std(df_hyp_2),0,ymax,'g','dashed')

  
    plt.title(f"n={n}, p={p}, df={df0}, K={K}, MC={MC}, center_cond={np.round(cond,3)} \n  samples_cond={np.round(samples_conds_mean,3)}")#" \n df_pop={np.round(np.mean(dfs_pop),3)}+/- {np.round(np.std(dfs_pop),3)}")
    plt.legend(loc="upper left")
    plt.xlabel("degree of freedom used for center estimation")
    plt.ylabel("L2 norm of diff between estimated center \n and real center")
    plt.show()
    
    ##########################################################################
    ##figure 2:  Traces of estimated centers (+theorical upper bound)
    ##########################################################################
    figbis = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    bound_trace={df: [(max(samples_traces[j])/df)*(df/n+p-1) for j in range(MC)] for df in dfs}
    ##bound_trace[df][j]= upper bound of the trace of estimated center with df based on the samples of the jth monte carlo set 
    bound_trace_means = [mean_list(bound_trace[df]) for df in dfs]
    bound_trace_stds = [std_list(bound_trace[df]) for df in dfs]
    trace_means = [mean_list(centers_traces[df]) for df in dfs]
    trace_stds = [std_list(centers_traces[df]) for df in dfs]
    y2= plt.errorbar(dfs,bound_trace_means,bound_trace_stds,label="theorical")
    y2bis= plt.errorbar(dfs,trace_means,trace_stds,label="empirical")
    plt.title(f"n={n}, p={p}, df={df0}, K={K}, MC={MC}, center_cond={np.round(cond,3)} \n  samples_cond={np.round(samples_conds_mean,3)}")#" \n df_pop={np.round(np.mean(dfs_pop),3)}+/- {np.round(np.std(dfs_pop),3)}")
    plt.legend(loc="upper left")
    plt.xlabel("degree of freedom used for center estimation")
    plt.ylabel("trace of the computed centers")
    plt.show()

    ##########################################################################
    ##figure 3: Traces of the inverse of estimated centers (+theorical upper bound)
    ##########################################################################
    figtris = plt.figure()
    respect_df_hyp = True
    plt.xscale('log')
    plt.yscale('log')
    bound_inv_trace={df: [max(samples_inverse_traces[j])*df/(df/n+p-max(samples_traces[j])*max(samples_inverse_traces[j])) for j in range(MC)] for df in dfs}
    ##bound_inv_trace[df][j]=trace of the inverse of the estimated center with df based on samples of the jth monte carlo set
    bound_inv_trace_means = [mean_list(bound_inv_trace[df]) for df in dfs]
    bound_inv_trace_stds = [std_list(bound_inv_trace[df]) for df in dfs]
    inv_trace_means = [mean_list(centers_inverse_traces[df]) for df in dfs]
    inv_trace_stds = [std_list(centers_inverse_traces[df]) for df in dfs]
    y3= plt.errorbar(dfs,bound_inv_trace_means,bound_inv_trace_stds,label="theorical")
    y3bis= plt.errorbar(dfs,inv_trace_means,inv_trace_stds,label="empirical")
    ymax=max(max(inv_trace_means)+max(inv_trace_stds),max(bound_inv_trace_means)+max(bound_inv_trace_stds))
    plt.vlines(np.mean(df_hyp_1),0,ymax,'g',label="hyp on traces")
    plt.vlines(np.mean(df_hyp_1)-np.std(df_hyp_1),0,ymax,'g','dashed')
    plt.vlines(np.mean(df_hyp_1)+np.std(df_hyp_1),0,ymax,'g','dashed')
    if respect_df_hyp:
        plt.xlim([np.mean(df_hyp_1)-1.01*np.std(df_hyp_1),max(dfs)])
    plt.title(f"n={n}, p={p}, df={df0}, K={K}, MC={MC}, center_cond={np.round(cond,3)} \n  samples_cond={np.round(samples_conds_mean,3)}")#" \n df_pop={np.round(np.mean(dfs_pop),3)}+/- {np.round(np.std(dfs_pop),3)}")
    plt.legend(loc="upper left")
    plt.xlabel("degree of freedom used for center estimation")
    plt.ylabel("trace of the inverse of computed centers")
    plt.show()
    
 
    ##########################################################################
    ##figure 4: Largest eigenvalue of estimated centers (+theorical upper bound)
    ##########################################################################
    figbisbis = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    bound_max_eigenval={df: [(1/df)*((df/n+p)*max(samples_max_eigenval[j]) - min(samples_min_eigenval[j])) for j in range(MC)] for df in dfs}
    bound_max_eigenval_means = [mean_list(bound_max_eigenval[df]) for df in dfs]
    bound_max_eigenval_stds = [std_list(bound_max_eigenval[df]) for df in dfs]
    max_eigenval={df:[max(centers_eigenvals[df][j]) for j in range(MC)] for df in dfs}
    max_eigenval_means = [mean_list(max_eigenval[df]) for df in dfs]
    max_eigenval_stds = [std_list(max_eigenval[df]) for df in dfs]
    y4= plt.errorbar(dfs,bound_max_eigenval_means,bound_max_eigenval_stds,label="theorical")
    y4bis= plt.errorbar(dfs,max_eigenval_means,max_eigenval_stds,label="empirical")
    plt.title(f"n={n}, p={p}, df={df0}, K={K}, MC={MC}, center_cond={np.round(cond,3)} \n  samples_cond={np.round(samples_conds_mean,3)}")#" \n df_pop={np.round(np.mean(dfs_pop),3)}+/- {np.round(np.std(dfs_pop),3)}")
    plt.legend(loc="upper left")
    plt.xlabel("degree of freedom used for center estimation")
    plt.ylabel("Largest eigenvalue of the computed centers")
    plt.show()

 
    ##########################################################################
    ##figure 4: Smallest eigenvalue of estimated centers (+theorical lower bound)
    ##########################################################################  
    figtrisbis = plt.figure()
    respect_df_hyp = True
    plt.xscale('log')
    plt.yscale('log')
    bound_min_eigenval={df: [(1/df)*((df/n+p)*min(samples_min_eigenval[j])-p*p*max(samples_max_eigenval[j])) for j in range(MC)] for df in dfs}
    bound_min_eigenval_means = [mean_list(bound_min_eigenval[df]) for df in dfs]
    bound_min_eigenval_stds = [std_list(bound_min_eigenval[df]) for df in dfs]
    min_eigenval={df:[min(centers_eigenvals[df][j]) for j in range(MC)] for df in dfs}
    min_eigenval_means = [mean_list(min_eigenval[df]) for df in dfs]
    min_eigenval_stds = [std_list(min_eigenval[df]) for df in dfs]
    y5= plt.errorbar(dfs,bound_min_eigenval_means,bound_min_eigenval_stds,label="theorical")
    y5bis= plt.errorbar(dfs,min_eigenval_means,min_eigenval_stds,label="empirical")
    ymax=max(max(min_eigenval_means)+max(min_eigenval_stds),max(bound_min_eigenval_means)+max(bound_min_eigenval_stds))
    plt.vlines(np.mean(df_hyp_2),0,ymax,'g',label="hyp on traces")
    plt.vlines(np.mean(df_hyp_2)-np.std(df_hyp_2),0,ymax,'g','dashed')
    plt.vlines(np.mean(df_hyp_2)+np.std(df_hyp_2),0,ymax,'g','dashed')
    if respect_df_hyp:
        plt.xlim([np.mean(df_hyp_2)-1.01*np.std(df_hyp_2),max(dfs)])
    plt.title(f"n={n}, p={p}, df={df0}, K={K}, MC={MC}, center_cond={np.round(cond,3)} \n  samples_cond={np.round(samples_conds_mean,3)}")#" \n df_pop={np.round(np.mean(dfs_pop),3)}+/- {np.round(np.std(dfs_pop),3)}")
    plt.legend(loc="upper left")
    plt.xlabel("degree of freedom used for center estimation")
    plt.ylabel("Smallest eigenvalue of computed centers")
    plt.show()
    
        
