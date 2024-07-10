# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:57:44 2024

@author: tbora
"""

import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score, v_measure_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import KFold
from sklearn.metrics.cluster import contingency_matrix
import seaborn as sns
from scipy import stats

#%%
def read_data(dataset, wall_info=False, drop_nan=True):
    
    if dataset == 'shm-ufjf':
        X = pd.read_csv('shm-ufjf/all_cury_approach.csv')
        y = pd.read_csv('shm-ufjf/target.csv')
        
        X.drop(columns = ['mfcc_1', 'mfcc_2', 'mfcc_3',
         'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10',
         'mfcc_11', 'mfcc_12', 'mfcc_13'], inplace=True)
        
        X.drop(columns='Unnamed: 0', inplace=True)
        y.drop(columns='Unnamed: 0', inplace=True)
        
        column_names = list(X.columns)
        C = []
        for aux in range(1,5):
            for name in column_names:
                C.append(name+'--'+str(aux))
    
        X2 = np.zeros(( X.shape[0]//4, X.shape[1]*4 ))
        lin = 0

        for row in range(0,X.shape[0],4):
            X2[lin,:] = X.iloc[row:row+4,:].values.ravel()
            lin+=1;

        y = pd.DataFrame(y['Damage Identification'][:X2.shape[0]])
        y.columns = ['Damage Level']
        X = pd.DataFrame(X2.copy())
        X.columns = C
        
    elif dataset == 'vibwall-anomaly':
        X = pd.read_csv('vibwall_data/all_vibwall.csv')
        y = pd.read_csv('vibwall_data/target_vibwall.csv')
        
        X.drop(columns='Unnamed: 0', inplace=True)
        y.drop(columns='Unnamed: 0', inplace=True)
        
        if wall_info == False:
            X.drop(columns='wall', inplace=True)
            
        idx_to_drop = []
        for row in range(X.shape[0]):
            if np.isnan(X.iloc[row,:]).any():
                idx_to_drop.append(row)
                 
        X.drop(index=idx_to_drop, inplace=True)
        y.drop(index=idx_to_drop, inplace=True)
        
        
    elif dataset == 'vibwall-severity':
        X = pd.read_csv('vibwall_data/build_wall/all_all_freq_as_sample.csv')
        y = pd.read_csv('vibwall_data/build_wall/target.csv')
        
        X.drop(columns='Unnamed: 0', inplace=True)
        y.drop(columns='Unnamed: 0', inplace=True)
        
    elif dataset == 'Z24':
        X = pd.read_csv('./Z24 Bridge/all.csv')
        y = pd.read_csv('./Z24 Bridge/Target.csv')
             
        X.drop(columns='Unnamed: 0', inplace=True)
        y.drop(columns='Unnamed: 0', inplace=True)
        
    if drop_nan:
        idx_to_drop = []
        for row in range(X.shape[0]):
            if np.isnan(X.iloc[row,:]).any():
                idx_to_drop.append(row)
                
        X.drop(index=idx_to_drop, inplace=True)
        y.drop(index=idx_to_drop, inplace=True)
        
    return X, y
    

#%%
def find_best_model(model, X, N=None, score_metric='silhouette'):

    if type(model).__name__ == 'AgglomerativeClustering':
        
        p1 = [ 'ward', 'complete', 'average', 'single' ]
        p2 = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine',]# 'precomputed']

    elif type(model).__name__ == 'KMeans':
        
        p1 = [ 'k-means++', 'random' ]
        p2 = ['lloyd', 'elkan', 'auto', 'full']    

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    best_score = 0;
    
    for par1 in p1:
        for par2 in p2:
            for n in range_n_clusters:
                
                if type(model).__name__ == 'AgglomerativeClustering':
                    
                    if par1 == 'ward':
                        par2 = 'euclidean'
                    model.linkage = par1
                    model.metric = par2
                    if not N:
                        model.n_clusters = n 
                    else:
                        model.n_clusters = N
                
                elif type(model).__name__ == 'KMeans':
                    model.init = par1
                    model.algorithm = par2
                    if not N:
                        model.n_clusters = n 
                    else:
                        model.n_clusters = N
                
                model.fit(X)
                cluster_labels = model.labels_
                
                if score_metric == 'silhouette':
                    score = silhouette_score(X, cluster_labels)
                    
                elif score_metric == 'calinski_harabasz':
                    score = calinski_harabasz_score(X, cluster_labels)
                elif score_metric == 'davies_bouldin':
                    score = 1/davies_bouldin_score(X, cluster_labels)
                    
                if score > best_score:
                    best_score = score
                    best_params = model.get_params()

    return best_score, best_params


#%%
def boxplot_feature_selection(X, y):
    
    n_groups = y.nunique()[0]
    
    q1 = np.zeros( (n_groups, X.shape[1] ))
    q3 = np.zeros( (n_groups, X.shape[1] ))
    maxs = np.zeros( (n_groups, X.shape[1] ))
    mins = np.zeros( (n_groups, X.shape[1] ))
    iqr  = np.zeros( (n_groups, X.shape[1] ))
    
    features = pd.DataFrame( np.zeros(X.shape[1]) ).T
    features.columns = X.columns
    
    for col, feature in enumerate(X.columns):
        for group in range(n_groups):
            g_idx = np.where(y == group)[0]
            X_aux = X.iloc[g_idx,col]
            
            q1[group,col] = np.percentile(X_aux, 25)
            q3[group,col] = np.percentile(X_aux, 75)
            iqr[group,col] = 1.5 * stats.iqr(X_aux)
            mins[group,col] = np.min(X_aux[X_aux >= (q1[group,col] - iqr[group,col])])
            maxs[group,col] = np.max(X_aux[X_aux <= (q3[group,col] + iqr[group,col])])          
            
    for col, feature in enumerate(X.columns):
        
        if (iqr[:,col] == 0).any():
            continue
        else:
            for i in range(n_groups):
                for j in range(n_groups):
                    if i != j:                  
                        if mins[i, col] >= maxs[j, col]:
                            features[feature] += 2
                            break;
                        elif q1[i, col] > q3[j, col]:
                            features[feature] += 1
                            break;
                    else:
                        continue
                    
    return features


#%%
def plot_feature_analysis(X, y, fs, selection_threshold, dataset):
      
    cols = ['red' if x < selection_threshold else 'yellowgreen' for x in fs.values[0]]
    black_patch = mpatches.Patch(color='red', label='Discarded Feature')
    blue_patch = mpatches.Patch(color='yellowgreen', label='Selected Feature')

    # pl.figure(figsize=(45,15), dpi=300)
    # # pl.rc('xtick', fontsize=35) 
    # # pl.rc('ytick', fontsize=35) 
    # sns.barplot(fs, palette=cols)
    # pl.axhline(y=selection_threshold, c='r')
    # # pl.xlabel('Features')
    # pl.ylabel('Selection Occurrences', fontsize=35)
    # pl.legend(handles=[black_patch, blue_patch], fontsize=35)
    # pl.xticks(rotation='vertical')
    # pl.savefig('images/'+dataset+'/Selected_Features.pdf', format="pdf")
    # pl.show()
    
    pl.figure(figsize=(40,45), dpi=300)
    pl.rc('xtick', labelsize=30) 
    pl.rc('ytick', labelsize=30) 
    sns.barplot(x=fs.iloc[0,:], y = fs.columns, palette=cols,)
    pl.axvline(x=selection_threshold, c='r')
    # pl.ylabel('Features')
    pl.xlabel('Selection Occurrences', fontsize=35, rotation=0)
    pl.legend(handles=[black_patch, blue_patch], fontsize=35)
    pl.savefig('images/'+dataset+'/Selected_Features.pdf', format="pdf")
    pl.show()
    
    var_names = X.columns
        
    X = pd.DataFrame(X)
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    X = pd.concat([X, y], axis=1)
    
    for i in range(X.shape[1]-1): 
        pl.figure(dpi=500, figsize = (10,8))
        pl.rc('xtick', labelsize=25) 
        pl.rc('ytick', labelsize=25) 
        sns.boxplot(x = (X.iloc[:,-1]).astype(int), y=X.iloc[:,i], palette ="Paired" )
        pl.title('Feature: ' + str((var_names[i])), fontsize=25)
        pl.xlabel('Predicted Clusters', fontsize=25)
        # pl.ylabel(var_names[i], fontsize=20)
        pl.ylabel('', fontsize=20)
        # pl.savefig(str(var_names[i]))
        pl.savefig('images/'+dataset+'/'+str((var_names[i]))+'.pdf', format="pdf")
        pl.show()


#%%
def corr_filter(X, threshold=0.90):
    
    corr_matrix = X.corr()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index and column name of features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(abs(upper[column]) >= threshold)]
    X.drop(columns = to_drop, inplace=True)
    
    return X, to_drop


#%%
def arruma_classes(y_true, y_pred, t_index, Y_cont, contador):
    
    classes = pd.unique(y_true.values.ravel())
    N = len(classes)
    y_pred = y_pred + 11
    
    for c in (classes[::-1]):
        true_label = int(c)
        true_idx = np.where(y_true == true_label)[0]
        
        pred_label = y_pred.iloc[true_idx].value_counts().index[0][0]
        pred_idx = np.where(y_pred == pred_label)[0]
        
        y_pred.iloc[pred_idx] = (true_label*(-1))
        
    
    for id_p , p in enumerate(pd.unique(y_pred.values.ravel())):
        if p >= 11:
            true_label = int(N)
    
            pred_idx = np.where(y_pred == p)[0]
            y_pred.iloc[pred_idx] = (true_label*(-1))
            N +=1
            
    Y_cont[t_index, contador] = abs(y_pred.values.ravel())
    
    return abs(y_pred), Y_cont


#%%
def plot_selected_features(fs, selection_threshold, dataset):
    # %matplotlib inline   
    cols = ['red' if x < selection_threshold else 'yellowgreen' for x in fs.values[0]]
    black_patch = mpatches.Patch(color='red', label='Discarded Feature')
    blue_patch = mpatches.Patch(color='yellowgreen', label='Selected Feature')

    pl.figure(figsize=(40,46), dpi=300)
    pl.rc('xtick', labelsize=35) 
    pl.rc('ytick', labelsize=35) 
    sns.barplot(x=fs.iloc[0,:], y = fs.columns, palette=cols,)
    pl.axvline(x=selection_threshold, c='r')
    # pl.ylabel('Features')
    pl.xlabel('Selection Occurrences', fontsize=40, rotation=0)
    pl.legend(handles=[black_patch, blue_patch], fontsize=45)
    # pl.xticks(rotation='horizontal')
    pl.savefig('images/'+dataset+'/Selected_Features-shm-ufjf.pdf', 
               bbox_inches='tight', format="pdf")
    pl.show()













