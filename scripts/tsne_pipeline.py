import os
import matplotlib.pyplot as plt
from dataloader import DataLoader
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
import random


def sample_data(X, y, sample_ratio=0.1):
    idx_list = np.arange(len(data))
    random.shuffle(idx_list)
    X = X[idx_list][:int(sample_ratio*len(X))]
    y = y[idx_list][:int(sample_ratio*len(y))]
    return X, y


if __name__=="__main__":
    # data_path = "/home/avirmani/lab/data/"
    data_path = "/home/kasun/aim_hi_project_kasun/data/"
    cph_paths = ["DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004.npy", "DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.npy",
             "DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.npy", "DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.npy"]
    naive_paths = ["DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.npy", "DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006.npy",
                   "DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009.npy", "DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.npy"]

    msi_cph = DataLoader([os.path.join(data_path, cph) for cph in cph_paths], 1)
    msi_naive = DataLoader([os.path.join(data_path, naive) for naive in naive_paths], 1)

    nmf_comp = 5
    cph_data = np.vstack([file.imzml_2d_array for file in msi_cph.train_data])[:, :500]
    naive_data = np.vstack([file.imzml_2d_array for file in msi_naive.train_data])[:, :500]
    data = np.vstack([cph_data, naive_data])
    # cph = 1 | naive = 0
    labels = np.hstack([np.ones(len(cph_data)), np.zeros(len(naive_data))])
    data, labels = sample_data(data, labels)

    ## perform NMF
    nmf = NMF(n_components=nmf_comp, init='random', random_state=0, max_iter=5000)
    nmf.fit(data)
    X_nmf = nmf.transform(data)

    ## select 10% of data randomly
    # X_nmf = sample_data(X_nmf, labels)

    ## perform TSNE
    tsne = TSNE(n_components=2, init='pca', n_jobs=-1, perplexity=2000)
    X_tsne = tsne.fit_transform(X_nmf)

    ## plot 10% of random data
    # X_tsne , labels = sample_data(X_tsne, labels)

    ## plot visualization
    red = labels == 1
    green = labels == 0
    plt.scatter(X_tsne[red, 0], X_tsne[red, 1], c="r")
    plt.scatter(X_tsne[green, 0], X_tsne[green, 1], c="g")
    plt.title('TSNE Visualization in 2D')
    plt.legend(['CPH', 'Naive'])
    plt.show()
    pass

