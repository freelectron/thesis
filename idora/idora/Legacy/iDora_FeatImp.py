# General
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback
from operator import itemgetter
from functools import reduce   
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
# Mutual Info sklearn
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import chi2
# Custom
import settings
# KDE
from numpy import trapz
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate

# FUNCTIONS
                                          #  ------ Mutual Information  ------
def mi_class(data,target,features): 
    """ Normalized MI for classification (labeling: treats each values as a distinct class) """
    mi_norm_lst = []
    for feat in features: 
        name = feat 
        mi_n = metrics.normalized_mutual_info_score(data[feat], data[target])
        mi_norm_lst.append((name,mi_n))
    mi_norm = sorted(mi_norm_lst,key=itemgetter(0))
    df_mi_norm = pd.DataFrame(mi_norm, columns=['feature', 'Norm_MI_Class'])
    df_mi_norm.sort_values(by = 'feature', inplace = True)
    
    return df_mi_norm

def mi_cont(df,target,features=[]):
    """ Mutual Information Estimation where the target variable is continuous """
    mi_skl_lst = [] 
    # ^ Fix when too lazy to give feature list
    if features == []:
        features = list( df.describe().columns )
        features.remove(target)
    # ^ end
    for feat in features: 
        y = df.loc[:,[feat, target]].values  
        X = y[:,0].reshape( (-1,1) )
        y = y[:,1] 
        d = mutual_info_regression(X=X,y=y)              
        mi_skl_lst.append( (feat,d[0]) ) 
    mi_lst = sorted(mi_skl_lst,key=itemgetter(0))
    df_mi_lst = pd.DataFrame(mi_lst, columns=['feature', 'MI_Cont'])
    df_mi_lst.sort_values(by = 'feature', inplace = True)

    return df_mi_lst

def mi_disc(df,target,features=[]):
    """ Mutual Information Estimation where the target variable is discrete """
    mi_skl_lst = [] 
    # ^ A quick fix for the case when too lazy to give feature list
    if features == []:
        features = list( df.describe().columns )
        features.remove(target)
    # ^ end
    for feat in features: 
        y = df.loc[:,[feat, target]].values  
        X = y[:,0].reshape( (-1,1) )
        y = y[:,1] 
        d = mutual_info_classif(X=X,y=y)              
        mi_skl_lst.append( (feat,d[0]) ) 
    mi_lst = sorted(mi_skl_lst,key=itemgetter(0))
    df_mi_lst = pd.DataFrame(mi_lst, columns=['feature', 'MI_Disc'])
    df_mi_lst.sort_values(by = 'feature', inplace = True)

    return df_mi_lst

def mi_cont_brp(df,target,features,N_samples=100,sample_size=1000):
    """" NEEEDS AN ACADEMIC PROOVE ! """
    # Produce MI scores with Bootstrapping
    mi_skl_lst = [] 
    for feat in features: 
        data = df.loc[:,[feat, target]].values 
        temp_lst = []
        for j in range(N_samples):
            mat = data[np.random.choice(data.shape[0], sample_size, replace=False)]
            X = mat[:,0].reshape( (-1,1) )
            y = mat[:,1] 
            d = mutual_info_regression(X=X,y=y)              
            temp_lst.append( (d[0]) )
        temp_avg = sum(temp_lst) / float(len(temp_lst))
        mi_skl_lst.append( (feat,temp_avg) )
    mi_lst = sorted(mi_skl_lst,key=itemgetter(0))
    df_mi_lst_brp = pd.DataFrame(mi_lst, columns=['feature', 'MI_Cont_Brp'])
    df_mi_lst_brp.sort_values(by = 'feature', inplace = True)

    return df_mi_lst_brp

def mi_disc_brp(df,target,features,N_samples=100,sample_size=1000):
    """" NEEEDS AN ACADEMIC PROOVE ! """
    # Produce MI scores with Bootstrapping
    mi_skl_lst = [] 
    for feat in features: 
        data = df.loc[:,[feat, target]].values 
        temp_lst = []
        for j in range(N_samples):
            mat = data[np.random.choice(data.shape[0], sample_size, replace=False)]
            X = mat[:,0].reshape( (-1,1) )
            y = mat[:,1] 
            d = mutual_info_classif(X=X,y=y)
            temp_lst.append( (d[0]) )
        temp_avg = sum(temp_lst) / float(len(temp_lst))
        mi_skl_lst.append( (feat,temp_avg) )
    mi_lst = sorted(mi_skl_lst,key=itemgetter(0))
    df_mi_lst_brp = pd.DataFrame(mi_lst, columns=['feature', 'MI_Disc_Brp'])
    df_mi_lst_brp.sort_values(by = 'feature', inplace = True)

    return df_mi_lst_brp

                                          #  ------  Correlation ------
def pearson_corr(df,target,features):
    """ Mutual Information Estimation where the target variable is discrete """
    corr_lst = []
    for i in features:
        corr_ = np.corrcoef(df[i],df[target])
        coeff = corr_[0][1]
        corr_lst.append( (i,coeff) ) 
    corr_lst = sorted(corr_lst,key=itemgetter(0))
    df_corr_lst = pd.DataFrame(corr_lst, columns=['feature', 'P_Corr'])
    df_corr_lst.sort_values(by = 'feature', inplace =True) 
    
    return df_corr_lst

def pearson_corr_abs(df,target,features):
    """ Mutual Information Estimation where the target variable is discrete """
    corr_lst = []
    for i in features:
        corr_ = np.corrcoef(df[i],df[target])
        coeff = corr_[0][1]
        if coeff<0:
            coeff = abs(coeff)
        corr_lst.append( (i,coeff) )
    corr_lst = sorted(corr_lst,key=itemgetter(0))
    df_corr_lst = pd.DataFrame(corr_lst, columns=['feature', 'P_Corr_Abs'])
    df_corr_lst.sort_values(by = 'feature', inplace =True) 
    return df_corr_lst

def distance_corr(df,target,features):
    """ Distance Correlation """
    corr_lst = []
    for i in features:
        corr_ = compute_distance_corr(df[i],df[target])
        coeff = corr_
        corr_lst.append( (i,coeff) )
    corr_lst = sorted(corr_lst,key=itemgetter(0))
    df_corr_lst = pd.DataFrame(corr_lst, columns=['feature', 'D_Corr'])
    df_corr_lst.sort_values(by = 'feature', inplace =True) 
    
    return df_corr_lst
                        
def distance_corr_abs(df,target,features):
    """ Distance Correlation Absolute Values"""
    corr_lst = []
    for i in features:
        corr_ = compute_distance_corr(df[i],df[target])
        coeff = corr_
        if coeff<0:
            coeff = abs(coeff)
        corr_lst.append( (i,coeff) )
    corr_lst = sorted(corr_lst,key=itemgetter(0))
    df_corr_lst = pd.DataFrame(corr_lst, columns=['feature', 'D_Corr_abs'])
    df_corr_lst.sort_values(by = 'feature', inplace =True) 
    
    return df_corr_lst
    
                                              #  ------  Other ------
def x2_scores(df,target,features,bins=11):
    """ Produce Chi-squared values for each feature """
    x2_lst = []
    for feat in features: 
        name = feat 
        mat = None
        if ( len( df[feat].unique() ) > settings.cat_unique_th ): 
            mat = pd.DataFrame(pd.qcut(df[feat],bins,labels=False))
        else: 
            mat = df[feat].to_frame()
        chi_val,p = chi2(mat.values, df[target])
        x2_lst.append((name,chi_val[0]))
    x2_lst_ = sorted(x2_lst,key=itemgetter(0))
    df_x2 = pd.DataFrame(x2_lst_, columns=['feature', 'X2_Value'])
    df_x2.sort_values(by = 'feature', inplace = True)
    
    return df_x2

def d_auc(df,var_lst,target,k=1):
    kde_funcs = [kde_statsmodels_u, kde_scipy, kde_sklearn]
    d_auc_lst = [] 
    for feat in var_lst:
        x_pos = df[feat][df[target]==1].values
        x_neg = df[feat][df[target]==0].values
        x = df[feat].values
        h1 = x_pos.std() / k
        h2 = x_neg.std() / k
        h3 = x.std() / k
        bins3 = int( (- min(x) + max(x)) / h3 )
        bins1 = int( (- min(x_pos) + max(x_pos)) / h1 ) 
        bins2 = int( (- min(x_neg) + max(x_neg)) / h2 )
        x_grid = np.linspace(min(x), max(x), bins3)
        x_grid_pos = np.linspace(min(x_pos), max(x_pos),bins1)
        x_grid_neg = np.linspace(min(x_neg), max(x_neg),bins2)
        pdf_pos = kde_funcs[2](x_pos, x_grid, bandwidth=h3)
        pdf_neg = kde_funcs[2](x_neg, x_grid, bandwidth=h3)
        y = [min(x1,x2) for x1,x2 in zip(pdf_pos,pdf_neg)]
        area3 = trapz(y, dx=h3)
        y = [x for x in pdf_pos]
        area1 = trapz(y, dx=h3)
        y = [x for x in pdf_neg]
        area2 = trapz(y, dx=h3)
        D_AUC = ( area3 / (min(area1,area2)) )
        d_auc_lst.append((feat,D_AUC))
    d_auc = sorted(d_auc_lst,key=itemgetter(0))
    d_auc= pd.DataFrame(d_auc, columns=['feature', 'D_AUC'])
    d_auc.sort_values(by = 'feature', inplace = True)
    
    return d_auc

def d_auc_auto(df,target,k=1,var_lst=None):
    kde_funcs = [kde_statsmodels_u, kde_scipy, kde_sklearn]
    if var_lst is None:
        var_lst = []
        print('Using automatically identified discrete variable')
        for i in list(df.columns):
            if ( (len(df[i].unique()) <= settings.cat_unique_th) and (i!=target) ):
                pass # var_lst.append((i,1))
            elif (len(df[i].unique()) > settings.cat_unique_th):
                var_lst.append((i))   
    d_auc_lst = [] 
    for feat in var_lst:
        x_pos = df[feat][df[target]==1].values
        x_neg = df[feat][df[target]==0].values
        x = df[feat].values
        h1 = x_pos.std() / k
        h2 = x_neg.std() / k
        h3 = x.std() / k
        bins3 = int( (- min(x) + max(x)) / h3 )
        bins1 = int( (- min(x_pos) + max(x_pos)) / h1 ) 
        bins2 = int( (- min(x_neg) + max(x_neg)) / h2 )
        x_grid = np.linspace(min(x), max(x), bins3)
        x_grid_pos = np.linspace(min(x_pos), max(x_pos),bins1)
        x_grid_neg = np.linspace(min(x_neg), max(x_neg),bins2)
        pdf_pos = kde_funcs[2](x_pos, x_grid, bandwidth=h3)
        pdf_neg = kde_funcs[2](x_neg, x_grid, bandwidth=h3)
        y = [min(x1,x2) for x1,x2 in zip(pdf_pos,pdf_neg)]
        area3 = trapz(y, dx=h3)
        y = [x for x in pdf_pos]
        area1 = trapz(y, dx=h3)
        y = [x for x in pdf_neg]
        area2 = trapz(y, dx=h3)
        D_AUC = ( area3 / (min(area1,area2)) )
        d_auc_lst.append((feat,D_AUC))
    d_auc = sorted(d_auc_lst,key=itemgetter(0))
    d_auc= pd.DataFrame(d_auc, columns=['feature', 'D_AUC'])
    d_auc.sort_values(by = 'feature', inplace = True)  
    
    return d_aucs

### ---- Get to MI inspection ----
def numeric_check(df,flag=True):
    var_lst = []
    for i in list(df.columns):
        if not ( is_numeric_dtype(df[i]) ):
            flag = False
            var_lst.append(i)
            
    return flag,var_lst

def to_numeric(df,var_lst):
    for i in var_lst:
        df[i] = df[i].astype(float)
    
    return df

def MI_threshold_check(result,Reg=True):
    result.columns = map(str.lower, result.columns) 
    i = result
    if not Reg:           
        interest = ('mi_disc','mi_disc_brp')
        if interest[1] in list(i.columns):
            temp = i[interest[1]] 
            print(f'Starting with {len(temp)} features.' )
            temp = temp[temp > 0.01]
            print(f'Only {len(temp)} features not fully independent of target.' )
            for j in range(7,10,1):
                j = j/10
                print('j = ', j)
                q_0 = (temp.quantile(q=j)) 
                q_1 = (temp.quantile(q=j+0.1)) 
                vals_inter = [ i for i in temp.values if ( (i>=q_0) & (i<=q_1) ) ] 
                print(f'number of values in the {int(j*10 +1)}th quantile is ', len(vals_inter) )
                try: 
                    avg = sum(vals_inter) / ( float(len(vals_inter)) )
                except:
                    avg = 0.0           
                print('mean value is ', avg)
                print('Proportion of values in that quantile is ',len(vals_inter)/len(temp))
        else:
            temp = i[interest[0]] 
            print(f'Starting with {len(temp)} features.' )
            temp = temp[temp > 0.01]
            print(f'Only {len(temp)} features not fully independent of target.' )
            for j in range(7,10,1):
                j = j/10
                print('j = ', j)
                q_0 = (temp.quantile(q=j)) 
                q_1 = (temp.quantile(q=j+0.1)) 
                vals_inter = [ i for i in temp.values if ( (i>=q_0) & (i<=q_1) ) ] 
                print(f'number of values in the {int(j*10 +1)}th quantile is ', len(vals_inter) )
                try: 
                    avg = sum(vals_inter) / (float(len(vals_inter)))
                except:
                    avg = 0.0           
                print('mean value is ', avg)
                print('Proportion of values in that quantile is ',len(vals_inter)/len(temp))
    else:
        interest = ('mi_cont','mi_cont_brp')
        if interest[1] in list(i.columns):
            temp = i[interest[1]] 
            print(f'Starting with {len(temp)} features.' )
            temp = temp[temp > 0.01]
            print(f'Only {len(temp)} features not fully independent of target.' )
            for j in range(7,10,1):
                j = j/10
                print('j = ', j)
                q_0 = (temp.quantile(q=j)) 
                q_1 = (temp.quantile(q=j+0.1)) 
                vals_inter = [ i for i in temp.values if ( (i>=q_0) & (i<=q_1) ) ] 
                print(f'number of values in the {int(j*10 +1)}th quantile is ', len(vals_inter) )
                try: 
                    avg = sum(vals_inter) / ( float(len(vals_inter)) )
                except:
                    avg = 0.0           
                print('mean value is ', avg) 
                print('Proportion of values in that quantile is ',len(vals_inter)/len(temp))
        else:
            temp = i[interest[0]] 
            print(f'Starting with {len(temp)} features.' )
            temp = temp[temp > 0.01]
            print(f'Only {len(temp)} features not fully independent of target.' )
            for j in range(7,10,1):
                j = j/10
                print('j = ', j)
                q_0 = (temp.quantile(q=j)) 
                q_1 = (temp.quantile(q=j+0.1)) 
                vals_inter = [ i for i in temp.values if ( (i>=q_0) & (i<=q_1) ) ] 
                print(f'number of values in the {int(j*10 +1)}th quantile is ', len(vals_inter) )
                try: 
                    avg = sum(vals_inter) / ( float(len(vals_inter)) )
                except:
                    avg = 0.0           
                print('mean value is ', avg)
                print('Proportion of values in that quantile is ',len(vals_inter)/len(temp))
    
    return

### ---- Compute function: distance correlation  ----
def compute_distance_corr(X, Y):
    """ Compute the distance correlation function """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    
    return dcor

### ---- KDE helping functions  ----
def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)











