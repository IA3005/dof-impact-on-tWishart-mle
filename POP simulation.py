import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ortho_group, norm, uniform
from generate_data import t_wishart_rvs
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix

from tWDAbis import tWDA_,tWDA_pop

import mpld3
import streamlit.components.v1 as components

from scipy.linalg import sqrtm,pinv

from manifold import SPD
from ellipt_wishart_estim import wishart_est, t_wish_cost, t_wish_egrad
from script_run_t import wishart_t_est

def generate_random_sdp(p):
    diagonal = np.random.normal(size=p)**2
    U =ortho_group.rvs(dim=p) #random orthogonal matrix
    return U@np.diag(diagonal)@U.T


def mle_RCG(samples,n,df):
    if df==np.inf:
        return np.mean(samples,axis=0)/n
    _,p,_=samples.shape
    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p,alpha,beta)
    return wishart_t_est(samples,n,df,manifold)


def pop(samples,n,maxiter=10,threshold=1e-2):
    K,p,_ = samples.shape
    center_wishart = np.mean(samples,axis=0)/n
    traces = np.einsum("kij,ji->k",samples,pinv(center_wishart))
    kappa = (np.mean(traces**2)/(n*p*(n*p+2)))-1 #(E(Q²)/E(Q)²)*(np/(np+2))-1
    if kappa ==0:
        df_old = np.inf
    else:
        df_old = 2/kappa+4 # kappa = 2/(df-4)
    dfs = [df_old]
    t= 0
    error =np.inf
    while (t<maxiter) and (error>threshold):
        inverses_cov =np.zeros((K,p,p))
        for i in range(K):
            index_i= list(range(0,i))+list(range(i+1,K))
            cov_i = mle_RCG(samples[index_i],n,df_old) #simplify
            inverses_cov[i,:,:] = pinv(cov_i)
        
        theta = np.einsum("kij,kji->",samples,inverses_cov)/(n*K*p)
        df_new = 2*theta/(theta-1)
        error = np.abs(df_new-df_old)/df_old
        df_old = df_new
        dfs.append(df_new)
        t +=1
    return dfs
                

    
        

#Main
simul1 = False
if simul1:

    exp='2'
    
    n = 1280
    p = 24
    center = generate_random_sdp(p)
    print("The center is :",center)
    #np.save(f"pop/exp{exp}_p={p}_center.npy",center)
    #center = np.load("pop/exp{exp}_p={p}_center.npy")
    
    Ks = [5,10,20,50,100]
    df = 5
    MC = 50
    
    #samples= t_wishart_rvs(center,n,df,max(Ks)*MC)
    #print("All samples are generated!")
    
    dfs_all_ = []
    
    df_pops_ = []
    mse_pops_ =[]
    iter_pops_ =[]
    
    df_inits_ = []
    mse_inits_ =[]
    
    for K in Ks:
        print("K=",K)
        samples= t_wishart_rvs(center,n,df,K*MC)
        print("All samples are generated! Shape=",samples.shape)
        df_pops = np.zeros(MC)
        df_inits = np.zeros(MC)
        iter_pops = np.zeros(MC)
        
            
        dfs_all = Parallel(n_jobs=-1)(delayed(pop)(samples[j*K:(j+1)*K],n) for j in tqdm(range(MC)))
        dfs_all_.append(dfs_all)
        
        for j in range(MC):
            df_pops[j] = dfs_all[j][-1]
            df_inits[j] = dfs_all[j][0]
            iter_pops[j] = len(dfs_all[j])
        
        df_pops_.append(df_pops)
        mse_pops_.append(np.abs(df_pops-df)**2)
        iter_pops_.append(iter_pops)
    
        df_inits_.append(df_inits)
        mse_inits_.append(np.abs(df_inits-df)**2)
        
        df_pop = np.mean(df_pops)
        iter_pop = np.mean(iter_pops)
        df_init = np.mean(df_inits)
    
        print("The (mean) POP dof is estimated", df_pop," after an average of ",np.mean(iter_pop)," iterations")
        print("The (mean) dof init is estimated!", df_init)
        
        center_pop = mle_RCG(samples,n,df_pop)
        center_estim_df_known = mle_RCG(samples,n,df)
        err1 = np.linalg.norm(center_pop-center)
        err2 = np.linalg.norm(center_estim_df_known-center)
        
        print(err1)
        print(err2)
    
    data = { "K":[], "pop":[],"init":[]}
    for k in range(len(Ks)):
        for j in range(MC):
            data["K"].append(Ks[k])
            data["pop"].append(mse_pops_[k][j])
            data["init"].append(mse_inits_[k][j])
    data_ = pd.DataFrame(data)
    data_.to_csv(f"pop/exp{exp}_p={p}_n={n}_MC={MC}_df={df}.csv")
    plt.figure()
    sns.lineplot(x="K",y="pop",data=data_,label="pop")
    sns.lineplot(x="K",y="init",data=data_,label="init")
    plt.legend(loc="best")
    plt.title(f"MSE of estimated dof for p={p}, n={n}, df={df} and MC={MC}")
    plt.savefig(f"pop/exp{exp}.png")
    plt.show()


######################################################"
simul2=False
if simul2:
    
    nb_classes = 1
    n = 1280
    p=24
    dfs = [4]
    centers = np.zeros((nb_classes,p,p))
    nb_train_per_class = 50
    nb_test_per_class = 4
    train_samples = np.zeros((nb_classes*nb_train_per_class,p,p))
    train_labels = np.zeros(nb_classes*nb_train_per_class)
    test_samples = np.zeros((nb_classes*nb_test_per_class,p,p))
    test_labels = np.zeros(nb_classes*nb_test_per_class)
    
    #generate centers
    # for c in range(nb_classes):
    #     centers[c] = generate_random_sdp(p)    
    centers[0] = generate_random_sdp(p)
    epsilon0 = min(np.linalg.eigvalsh(centers[0]))*1e-3
    
    """
    centers[1] = centers[0] + epsilon0*np.eye(p)
    
    centers[2] = generate_random_sdp(p)
    epsilon2 = min(np.linalg.eigvalsh(centers[2]))*1e-3
    centers[3] = centers[2] + epsilon2*np.eye(p)
    """
    for c in range(nb_classes):
        for l in range(c+1,nb_classes):
            print(f"Diff of centers of class {c} and {l} = ",np.linalg.norm(centers[c]-centers[l]))
    
    #generate data
    for c in range(nb_classes):
        train_samples[c*nb_train_per_class:(c+1)*nb_train_per_class] = t_wishart_rvs(centers[c],n,dfs[c],nb_train_per_class)
        train_labels[c*nb_train_per_class:(c+1)*nb_train_per_class] = c
        test_samples[c*nb_test_per_class:(c+1)*nb_test_per_class] = t_wishart_rvs(centers[c],n,dfs[c],nb_test_per_class)
        test_labels[c*nb_test_per_class:(c+1)*nb_test_per_class] = c
        
    #classify
    twda = tWDA_(n,dfs=100,n_jobs=-1)
    twda_pop = tWDA_pop(n,n_jobs=-1)
    
    twda.fit(train_samples,train_labels)
    twda_pop.fit(train_samples,train_labels)
    
    twda_center= twda.centers[0]
    twda_pop_center = twda_pop.centers[0]
    
    
    
    """
    train_preds_twda = twda.predict(train_samples)
    train_preds_twda_pop = twda_pop.predict(train_samples)    
    
    test_preds_twda = twda.predict(test_samples)
    test_preds_twda_pop = twda_pop.predict(test_samples)   

    #evaluate scores
    train_score_twda = len(train_preds_twda[train_preds_twda==train_labels])/(nb_classes*nb_train_per_class)   
    train_score_twda_pop = len(train_preds_twda_pop[train_preds_twda_pop==train_labels])/(nb_classes*nb_train_per_class)   
    
    test_score_twda = len(test_preds_twda[test_preds_twda==test_labels])/(nb_classes*nb_test_per_class)   
    test_score_twda_pop = len(test_preds_twda_pop[test_preds_twda_pop==test_labels])/(nb_classes*nb_test_per_class)

    conf_twda = confusion_matrix(test_labels, test_preds_twda)
    conf_twda_pop = confusion_matrix(test_labels, test_preds_twda_pop)
    
    print("Dofs of tWDA_pop: ",twda_pop.dfs)
    
    print("Train Scores:")
    print(">> For tWDA = ",train_score_twda)
    print(">> For tWDA_pop = ",train_score_twda_pop)
        
    print("Test Scores:")
    print(">> For tWDA = ",test_score_twda)
    print(">> For tWDA_pop = ",test_score_twda_pop)
    
    
    print("Test confusion matrices:")
    print(">> For tWDA = ",conf_twda)
    print(">> For tWDA_pop = ",conf_twda_pop)
    """
    
    
simul3=True
if simul3:
    

    
    n = 1280
    p=24
    K = 50
    df0 = 100
    MC= 100
    
    dfs = [3,5,10,50,100,200,500,1000,5000,1e4,1e5,1e7,1e8,1e10]
    
    center = generate_random_sdp(p)
    samples = t_wishart_rvs(center,n,df0,K*MC)
    
    centers_estim = np.zeros((len(dfs),MC,p,p))
    dfs_pop = np.zeros(MC)
    dfs_pop_niters = np.zeros(MC)

    for i in tqdm(range(len(dfs))):
        centers_estim[i]=np.asarray(Parallel(n_jobs=-1)(delayed(mle_RCG)(samples[j*K:(j+1)*K],n,dfs[i]) for j in (range(MC))))
        
    dfs_pop_all = Parallel(n_jobs=-1)(delayed(pop)(samples[j*K:(j+1)*K],n) for j in tqdm(range(MC)))
    for j in range(MC):
        dfs_pop[j] = dfs_pop_all[j][-1]
        dfs_pop_niters[j] = len(dfs_pop_all[j])
        
    centers_pop=np.asarray(Parallel(n_jobs=-1)(delayed(mle_RCG)(samples[j*K:(j+1)*K],n,dfs_pop[j]) for j in (range(MC))))
    
    
    errors_estim = np.linalg.norm(centers_estim-np.tile(center,(len(dfs),MC,1,1)),axis=(2,3))
    errors_pop = np.linalg.norm(centers_pop-np.tile(center,(MC,1,1)),axis=(1,2))
    
    
    fig = plt.figure()
    plt.xscale('log')
    plt.errorbar(dfs,np.mean(errors_estim,axis=1),yerr=np.std(errors_estim,axis=1),ecolor="r")
    plt.hlines(np.mean(errors_pop),dfs[0],dfs[-1],'b',label="pop")
    plt.hlines(np.mean(errors_pop)-np.std(errors_pop),dfs[0],dfs[-1],'b','dashed')
    plt.hlines(np.mean(errors_pop)+np.std(errors_pop),dfs[0],dfs[-1],'b','dashed')
    plt.title(f"n={n},p={p},df={df0},K={K},MC={MC} \n cond={np.round(np.linalg.cond(center),3)} \n df_pop={np.round(np.mean(dfs_pop),3)}+/- {np.round(np.std(dfs_pop),3)}")
    plt.legend(loc="upper left")
    plt.xlabel("degree of freedom used for center estimation")
    plt.ylabel("L2 norm of diff between estimated center \n and real center")
    plt.show()
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=600)
    
    
    
        