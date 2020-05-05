# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:41:38 2020

@author: lsfer
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def prepare_acp(db, n_components=None, svd_solver='auto'):
    X = db.values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X)
    return X_pca

# =============================================================================
# def subset_acp(X, y, n=None, frac=None, keepdims=None):
#     # print(X.shape, y.shape)
#     if type(keepdims) == list and max(keepdims) < X.shape[1]:
#         X = X[:,keepdims]
#     if type(n) == int:
#         subset = np.random.choice(np.arange(X.shape[0]), size=n, replace=False)
#         return X[subset], y[subset]
#     if type(frac) == int:
#         subset = np.random.choice(np.arange(X.shape[0]), size=int(frac*X.shape[0]), replace=False)
#         return X[subset], y[subset]
#     if type(n) == list:
#         subset = np.concatenate([np.random.choice(np.where(y==i)[0], size=n[i], replace=False) for i in [0, 1]])
#         return X[subset], y[subset]
#     if type(frac) == list:
#         subset = np.concatenate([np.random.choice(np.where(y==i)[0], size=int(frac[i]*np.where(y==i)[0].shape[0]), replace=False) for i in [0, 1]])
#         return X[subset], y[subset]
#     else:
#         return X, y
# =============================================================================

def plot_acp(X_pca, path_save, s=30, alpha=0.3, c='red', marker ='*'):
    Xax = X_pca[:,0]
    Yax = X_pca[:,1]
    if X_pca.shape[1] == 3:
        Zax = X_pca[:,2]
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection=('3d' if X_pca.shape[1]==3 else 'rectilinear'))
    fig.patch.set_facecolor('white')
    kwargs = {'c':c, 's':s, 'alpha':alpha, 'marker':marker}
    args = [Xax, Yax]
    if X_pca.shape[1] == 3:
        args.append(Zax)
    ax.scatter(*args, **kwargs)
    plt.xlabel("First Principal Component", fontsize=14)
    plt.ylabel("Second Principal Component", fontsize=14)
    plt.savefig(path_save)
    plt.show()