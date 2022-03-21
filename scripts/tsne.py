import os
import matplotlib.pyplot as plt
import numpy as np
# from openTSNE import TSNE
# from sklearn.manifold import TSNE
from tsnecuda import TSNE
from sklearn.decomposition import PCA
# from skcuda.linalg import PCA
import random

def plot_tsne(data, labels, n_comp=2, filename=None):
    # tsne = TSNE(n_components=n_comp, init='pca', n_jobs=-1, n_iter=1000, learning_rate=500)
    tsne = TSNE(perplexity=2000, learning_rate=3000, n_iter=10000)
    X_embedded = tsne.fit_transform(data)
    # tsne = np.load('tsne_nmf_25_2_objects.npy', allow_pickle=True)[()]
    # X_embedded = tsne.embedding_
    # np.save('tsne_nmf_25_2_objects.npy', tsne)
    # print('KL divergenc: ', tsne.kl_divergence_)
    red = labels == 1
    green = labels == 0
    # a = np.arange(len(red))
    # random.shuffle(a)
    # red[a[:int(len(red)/2)]] = False
    # green[a[int(len(red)/2):]] = False
    if n_comp == 2:
        plt.scatter(X_embedded[red, 0], X_embedded[red, 1], c="r")
        plt.scatter(X_embedded[green, 0], X_embedded[green, 1], c="g")
        plt.title('TSNE Visualization in 2D')
        plt.legend(['CPH', 'Naive'])
    elif n_comp == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_embedded[red, 0], X_embedded[red, 1], X_embedded[red, 1], c="r")
        ax.scatter(X_embedded[green, 0], X_embedded[green, 1], X_embedded[green, 1], c="g")
        ax.set_title('TSNE Visualization in 3D')
        # plt.legend(['CPH', 'Naive'])
    # can choose to comment this and do: plt.savefig(filename)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
    return tsne


if __name__ == "__main__":
    # TSNE
    dataset = np.load('tsne_nmf_25_input.npy', allow_pickle=True)
    data = dataset[()]['data']
    # cph = 1 | naive = 0
    labels = dataset[()]['labels']
    # data = PCA(n_components=2).fit_transform(data)
    tsne_obj_2 = plot_tsne(data, labels, 2)
    # tsne_obj_3 = plot_tsne(data, labels, 3)
    pass

