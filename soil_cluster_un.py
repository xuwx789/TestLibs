#!/usr/bin/python3
# coding: utf-8
from __future__ import division
from time import time
import random
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.feature_selection import SelectKBest, f_regression, chi2, mutual_info_regression, f_classif
from sklearn import preprocessing
from sklearn import metrics
import math
from openpyxl import Workbook
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
########################################################################################################################
def plot_embedding(x, fig_name, path, color, text, title=None):  # text should be in str type; color in vale type
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)
    plt.ion()
    for index in range(x.shape[0]):
        plt.text(x[index, 0], x[index, 1],
                 text[index, 0],
                 color=plt.cm.tab20(color[index, 0]/(np.max(color)+1)),
                 fontdict={'weight': 'bold', 'size': 8})
    if title is not None:
        plt.title(title, fontsize=10)
    plt.savefig(path + '\\' + fig_name + '.png', dpi=1000, bbox_inches='tight')
    plt.pause(10)
    plt.close()

########################################################################################################

#保存文件
def save(data,path):
    wb = Workbook()
    ws = wb.active # 激活 worksheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        row = []
        for j in range(l):
            row.append(data[i,j])
        ws.append(row)
    wb.save(path)
########################################################################################################################
data_file_path = r'E:\Soil'
menu_guide_path = r'E:\Soil\Menu_Guide.xlsx'
save_path = r'E:\Soil\result'
choose_element = 'Ag'
choose_type = 'Soil_Type'
file_list = 'Spectra_Name'
sample_list = 'Spectra_Name'
perplexities = [5,15]
data = pd.read_excel(menu_guide_path, dtype={'Spectra_Name': str})  # 'Ag': np.float64
file_names = np.array(data[file_list])
concentration = np.array(data[choose_element])
sample_type = np.array(data[choose_type])
sample_name_unique = np.array(data[sample_list])
train_test_set = np.array(data['Train']).tolist()
type_list = data[choose_type].drop_duplicates(keep='first').tolist()
flag_tr = False
flag_te = False
########################################################################################################################
# train list
train_position = [i for i, j in enumerate(train_test_set) if j == 1]
# test list
test_position = [i for i, j in enumerate(train_test_set) if j == 0]
########################################################################################################################
# Input Data
for index_i in train_position:
    temp_path = os.path.join(data_file_path, file_names[index_i])
    temp_sample_type = type_list.index(sample_type[index_i])
    temp_sample_name = sample_name_unique[index_i]
    temp_concentration = concentration[index_i]
    print('i:')
    print(index_i)
    for index_j in os.listdir(temp_path):
        filename = os.path.join(temp_path, index_j)
        data = np.loadtxt(filename, skiprows=0)
        print(index_j)
        if flag_tr:
            X_tr = np.row_stack((X_tr, data[:, 1].T))
            y_tr = np.row_stack((y_tr, temp_concentration))
            type_tr = np.row_stack((type_tr, temp_sample_type))
            sample_name_tr = np.row_stack((sample_name_tr, temp_sample_name))
            group_tr = np.row_stack((group_tr, index_i))
        else:
            X_tr = data[:, 1].T
            y_tr = temp_concentration
            type_tr = temp_sample_type
            group_tr = index_i
            sample_name_tr = temp_sample_name
            wavelength = data[:, 0]
            flag_tr = True
########################################################################################################################
# feature selection
skb = SelectKBest(f_classif,  # mutual_info_regression,
    #f_regression,
    k=X_tr.shape[1])
X_tr = skb.fit_transform(X_tr, type_tr)
score = skb.scores_
score = score.tolist()
score_temp_order = sorted(score, reverse=True)
######################################################
wavelength = wavelength.tolist()
wavelength_pick = [445.48633,280.10086]
pick_pixel_position = [i for i, j in enumerate(wavelength) if j in wavelength_pick]
X1 = X_tr[:, pick_pixel_position]
########################################################################################################################
# Initialization
save_result = sample_name_tr
headline = ['Sample_Name']
color_index =type_tr # group_tr
text_index = sample_name_tr
########################################################################################################################
# TSNE using different perplexity
print("Computing t-SNE clustering")
for i, perplexity in enumerate(perplexities):
    tsne0 = manifold.TSNE(n_components=2, init='random',
                         random_state=0, perplexity=perplexity)
    X_tsne1 = tsne0.fit_transform(X1)
    save_result = np.column_stack((save_result, X_tsne1))
    headline.append('p' + str(perplexity) + '_x')
    headline.append('p' + str(perplexity) + '_y')
    fig_name_temp = 'tsne_' + 'p' + str(perplexity)
    plot_embedding(X_tsne1, fig_name_temp, save_path, color_index, text_index)
########################################################################################################
#利用kmeans判断分几类最好
#db = DBSCAN(eps=0.3, min_samples=1).fit(X_tsne1)
estimators = [('k_means_iris_2', KMeans(n_clusters=2)),('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_4', KMeans(n_clusters=4)),('k_means_iris_5', KMeans(n_clusters=5)),
              ('k_means_iris_6',KMeans(n_clusters=6)),('k_means_iris_7',KMeans(n_clusters=7))]
for name,est in estimators:
    est.fit(X_tsne1)
    labels = est.labels_
    print(name,metrics.silhouette_score(X_tsne1, labels))

score2=[]
scores2=[]
for k in range(2,8):
    labels=KMeans(n_clusters=k).fit(X1).labels_
    score2=metrics.silhouette_score(X1, labels)
    scores2.append(score2)
print(scores2)
########################################################################################################################
# TSNE PCA using different perplexity
print("Computing t-SNE PCA clustering")
tsne = manifold.TSNE(n_components=2, init='pca')
X_tsne = tsne.fit_transform(X1)
fig_name_temp = 'tsne_' + 'pca'
headline.append(fig_name_temp + '_x')
headline.append(fig_name_temp + '_y')
save_result = np.column_stack((save_result, X_tsne))
plot_embedding(X_tsne, fig_name_temp, save_path, color_index, text_index)
########################################################################################################################
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=5)
X_projected = rp.fit_transform(X1)
fig_name_temp = 'Random_2D_projection'
headline.append(fig_name_temp + '_x')
headline.append(fig_name_temp + '_y')
save_result = np.column_stack((save_result, X_projected))
plot_embedding(X_projected, fig_name_temp, save_path, color_index, text_index)

########################################################################################################################
# MDS embedding of the digits data set
print("Computing MDS")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
X_mds = clf.fit_transform(X1)
print("Done. Stress: %f" % clf.stress_)
fig_name_temp = 'MDS_embedding'
headline.append(fig_name_temp + '_x')
headline.append(fig_name_temp + '_y')
save_result = np.column_stack((save_result, X_mds))
plot_embedding(X_mds, fig_name_temp, save_path, color_index, text_index)
########################################################################################################################
# save the results
save_result_df = pd.DataFrame(save_result)
save_result_df.columns = headline
save_result_df.to_excel(save_path + r'\result.xls')
