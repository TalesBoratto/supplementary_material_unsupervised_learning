# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:31:43 2024

@author: tbora
"""

import numpy as np
import pandas as pd
import pylab as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score, v_measure_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics.cluster import contingency_matrix
import seaborn as sns
from scipy import stats
import pickle

from defined_functions import *

#%%
def scatter_3d(X, y, y_pred):
    %matplotlib auto
    
    structural_state = {}
    color = ['b', 'm', 'r', 'c', 'k']
    if len(y.value_counts()) == 2:
        label = ['No Damage', 'Anomaly']
    else:
        label = ['No Damage', 'Anomaly 1', 'Anomaly 2', 'Anomaly 3', 'Anomaly 4']
    
    for i, _ in enumerate(y.value_counts()):
        structural_state[i] = pd.DataFrame(X[np.where(y == i)[0]])
        
    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    
    for case in structural_state.keys():
        ax.scatter(structural_state[case].iloc[:,0], structural_state[case].iloc[:,1], structural_state[case].iloc[:,2], c=color[case], label=label[case])  
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    pl.title('TRUE DAMAGES')
    pl.legend()
    pl.show()
        
    predicted_state = {}
    color = ['c', 'm', 'r', 'k', 'b', 'g', 'y', 'navy']
    label ={0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 
            3: 'Cluster 3', 4: 'Cluster 4', 5: 'Cluster 5',
            6: 'Cluster 6', 7: 'Cluster 7', 8: 'Cluster 8' }
    #label = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7']
    
    for i, idx in enumerate(y_pred.value_counts().index):
        group = idx[0]
        predicted_state[group] = pd.DataFrame(X[np.where(y_pred == group)[0]])
        
    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    
    for i, case in enumerate(predicted_state.keys()):
        ax.scatter(predicted_state[case].iloc[:,0], predicted_state[case].iloc[:,1], predicted_state[case].iloc[:,2], c=color[i], label=label[case]) 
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    pl.title('PREDICTED DAMAGES')
    pl.legend()
    pl.show()
    
    # %matplotlib inline

#%% Defining Problem Parameters

#'shm-ufjf', 'vibwall-anomaly', 'vibwall-severity', 'Z24'
dataset = 'Z24' 
use_wall_info = False

th = 0.90
select_th = 0.16

#'calinski_harabasz' #'silhouette
scorer = 'silhouette' 

# AgglomerativeClustering() // KMeans()
model = AgglomerativeClustering()

cv = 5
N_iter = 5
n_pca = 3

homogeneity = np.zeros(cv*N_iter)
completeness = np.zeros(cv*N_iter)
adj_rand_score = np.zeros(cv*N_iter)
v_measure = np.zeros(cv*N_iter)

homogeneity_fs = np.zeros(cv*N_iter)
completeness_fs = np.zeros(cv*N_iter)
adj_rand_score_fs = np.zeros(cv*N_iter)
v_measure_fs = np.zeros(cv*N_iter)

scores = []
params = []
scores_fs = []
params_fs = []

#%% Damage Detection

X_orig, Y_orig = read_data(dataset, wall_info=use_wall_info, drop_nan = True)

X_orig, dropeed = corr_filter(X_orig, th)
X_orig = pd.DataFrame(X_orig)

scaler = MinMaxScaler()
dim_reduction = PCA(n_components=n_pca) 

fs = pd.DataFrame( np.zeros(X_orig.shape[1]) ).T
fs.columns = X_orig.columns

cont = 0
for n in range(N_iter):
    kf = KFold(n_splits=cv, random_state=n, shuffle=True)
    kf.get_n_splits(X_orig)

    for i, (train_index, test_index) in enumerate(kf.split(X_orig)):

        X_train = X_orig.iloc[train_index,:]
        y_train = Y_orig.iloc[train_index]        

        X_train = scaler.fit_transform(X_train)
        X_train = dim_reduction.fit_transform(X_train)        

        s, p = find_best_model(model, X_train, score_metric=scorer)
        scores.append(s)
        params.append(p)
        
        best_params = p 
        
        if type(model).__name__ == 'AgglomerativeClustering':
            model.linkage = best_params['linkage']
            model.metric = best_params['metric']
            model.n_clusters = best_params['n_clusters']
        
        elif type(model).__name__ == 'KMeans':
            model.init = best_params['init']
            model.algorithm = best_params['algorithm']
            model.n_clusters = best_params['n_clusters']

        print(model.n_clusters)

        y_pred = pd.DataFrame( model.fit_predict(X_train) )

        homogeneity[cont] = homogeneity_score(y_train.iloc[:,0], y_pred.iloc[:,0])
        completeness[cont] = completeness_score(y_train.iloc[:,0], y_pred.iloc[:,0])
        adj_rand_score[cont] = adjusted_rand_score(y_train.iloc[:,0], y_pred.iloc[:,0])
        v_measure[cont] = v_measure_score(y_train.iloc[:,0], y_pred.iloc[:,0])

        cont += 1;
        
        fs += boxplot_feature_selection(pd.DataFrame(X_orig.iloc[train_index,:]), y_pred) 

# scatter_3d(X_train,y_train,y_pred)
print('Homogeneity: ' + str(np.round(homogeneity.mean(),4)) + '(' + str(np.round(homogeneity.std(),4)) + ')' )
print('Completeness: ' + str(np.round(completeness.mean(),4)) + '(' + str(np.round(completeness.std(),4)) + ')' )
print('Adj Rand Score: ' + str(np.round(adj_rand_score.mean(),4))+ '(' + str(np.round(adj_rand_score.std(),4)) + ')' )
print('V_Measure Score: ' + str(np.round(v_measure.mean(),4))+ '(' + str(np.round(v_measure.std(),4)) + ')' )


pd.DataFrame(homogeneity).to_csv(dataset+'_Homogeneity.csv', header=None, index=None)
pd.DataFrame(completeness).to_csv(dataset+'_Completeness.csv', header=None, index=None)
pd.DataFrame(adj_rand_score).to_csv(dataset+'_Adj_Rand_Score.csv', header=None, index=None)
pd.DataFrame(v_measure).to_csv(dataset+'_v_measure.csv', header=None, index=None)
pd.DataFrame(fs).to_csv(dataset+'_feature_importances.csv', index=None)

with open(dataset+'_scores.pickle', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(dataset+'_params.pickle', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%

selection_threshold = select_th*fs.values.max()

plot_selected_features(fs, selection_threshold, dataset)

selected_features = np.where(fs[:] >= selection_threshold)[1]

X_new = X_orig.iloc[:,selected_features]

scaler = MinMaxScaler()
dim_reduction = PCA(n_components=n_pca) 


Y = pd.DataFrame()
Y_cont = np.ones(( X_new.shape[0], cv*N_iter )) * (-1)

cont = 0
for n in range(N_iter):
    kf = KFold(n_splits=cv, random_state=n, shuffle=True)
    kf.get_n_splits(X_new)

    for i, (train_index, test_index) in enumerate(kf.split(X_new)):

        X_train = X_new.iloc[train_index,:]
        y_train = Y_orig.iloc[train_index]        

        X_train = scaler.fit_transform(X_train)
        X_train = dim_reduction.fit_transform(X_train)        

        s, p = find_best_model(model, X_train, score_metric=scorer)
        scores_fs.append(s)
        params_fs.append(p)
        
        best_params = p
        
        if type(model).__name__ == 'AgglomerativeClustering':
            model.linkage = best_params['linkage']
            model.metric = best_params['metric']
            model.n_clusters = best_params['n_clusters']
        
        elif type(model).__name__ == 'KMeans':
            model.init = best_params['init']
            model.algorithm = best_params['algorithm']
            model.n_clusters = best_params['n_clusters']

        print(model.n_clusters)

        y_pred = pd.DataFrame( model.fit_predict(X_train) )
        
        Y_PRED, Y_cont = arruma_classes(y_train, y_pred, train_index, Y_cont, cont)
        Y[cont] = Y_PRED

        homogeneity_fs[cont] = homogeneity_score(y_train.iloc[:,0], y_pred.iloc[:,0])
        completeness_fs[cont] = completeness_score(y_train.iloc[:,0], y_pred.iloc[:,0])
        adj_rand_score_fs[cont] = adjusted_rand_score(y_train.iloc[:,0], y_pred.iloc[:,0])
        v_measure_fs[cont] = v_measure_score(y_train.iloc[:,0], y_pred.iloc[:,0])
    
        cont +=1
        
print('Homogeneity: ' + str(np.round(homogeneity_fs.mean(),4)) + '(' + str(np.round(homogeneity_fs.std(),4)) + ')' )
print('Completeness: ' + str(np.round(completeness_fs.mean(),4)) + '(' + str(np.round(completeness_fs.std(),4)) + ')' )
print('Adj Rand Score: ' + str(np.round(adj_rand_score_fs.mean(),4))+ '(' + str(np.round(adj_rand_score_fs.std(),4)) + ')' )
print('V_Measure Score: ' + str(np.round(v_measure_fs.mean(),4))+ '(' + str(np.round(v_measure_fs.std(),4)) + ')' )


pd.DataFrame(homogeneity_fs).to_csv(dataset+'_Homogeneity_fs.csv', header=None, index=None)
pd.DataFrame(completeness_fs).to_csv(dataset+'_Completeness_fs.csv', header=None, index=None)
pd.DataFrame(adj_rand_score_fs).to_csv(dataset+'_Adj_Rand_Score_fs.csv', header=None, index=None)
pd.DataFrame(v_measure_fs).to_csv(dataset+'_v_measure_fs.csv', header=None, index=None)


with open(dataset+'_scores_fs.pickle', 'wb') as handle:
    pickle.dump(scores_fs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(dataset+'_params_fs.pickle', 'wb') as handle:
    pickle.dump(params_fs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# scatter_3d(X_train,y_train,y_pred)

#%% Voting-Based Cluster Definition

Y_cont = pd.DataFrame(Y_cont)

voting = np.ones(( Y_cont.shape[0], 2 ))*(-1)
for sample in range(Y_cont.shape[0]):
    aux_idx = Y_cont.iloc[sample,:].value_counts().index[0]
    aux_contagem = Y_cont.iloc[sample,:].value_counts().values[0]
    voting[sample, :] = [aux_idx, aux_contagem]
            
y_predict = pd.DataFrame(voting[:,0])

X = scaler.fit_transform(X_new)
X = dim_reduction.fit_transform(X)     

# plot_feature_analysis(X_new, y_predict)
# scatter_3d(X,Y_orig,y_predict)

vonting_based_performance = []

print('Vonting Based Performance Metrics: ')
vonting_based_performance.append(homogeneity_score(Y_orig.iloc[:,0], y_predict.iloc[:,0]))
print(homogeneity_score(Y_orig.iloc[:,0], y_predict.iloc[:,0]))
vonting_based_performance.append(completeness_score(Y_orig.iloc[:,0], y_predict.iloc[:,0]))
print(completeness_score(Y_orig.iloc[:,0], y_predict.iloc[:,0]))
vonting_based_performance.append(adjusted_rand_score(Y_orig.iloc[:,0], y_predict.iloc[:,0]))
print(adjusted_rand_score(Y_orig.iloc[:,0], y_predict.iloc[:,0]))
vonting_based_performance.append(v_measure_score(Y_orig.iloc[:,0], y_predict.iloc[:,0]))
print(v_measure_score(Y_orig.iloc[:,0], y_predict.iloc[:,0]))

Y_cont.to_csv(dataset+'_Y_clustering_count.csv', index=None)
y_predict.to_csv(dataset+'_target_voting-based.csv', index=None)
pd.DataFrame(vonting_based_performance).to_csv(dataset+'_voting-based-performance.csv', header=None, index=['Homogeneity', 'Completeness', 'Adj_rand_score','V-Measure'])
