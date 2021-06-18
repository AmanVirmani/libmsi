import libmsi
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.cluster import KMeans


class DataLoader:
    def __init__(self, filenames=[]):
        self.filenames = filenames
        self.files = self.load_files()
        self.preprocess('TIC')
        self.train_data, self.val_data = self.data_split()

    def load_files(self):
        files = []
        for file in self.filenames:
            files.append(libmsi.Imzml(file, 1))
        return files

    def data_split(self, split_ratio=0.75):
        random.shuffle(self.files)
        train_data = self.files[:int(split_ratio * len(self.files))]
        val_data = self.files[int(split_ratio * len(self.files)):]
        return train_data, val_data

    def preprocess(self, normalization='TIC'):
        for file in self.files:
            file.normalize_data(method=normalization)

    # Dimensionality Reduction
    def perform_pca(self, n_components=15):
        self.pca = PCA(n_components=n_components)
        data = np.vstack([file.imzml_2d_array for file in self.train_data])
        self.pca.fit(data)

    def perform_ica(self, n_components=15):
        self.ica = FastICA(n_components=n_components, random_state=0)
        data = np.vstack([file.imzml_2d_array for file in self.train_data])
        self.ica.fit(data)

    def perform_nmf(self, n_components=15, type='all'):
        data_list = []
        if type == 'all':
            self.nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=20000)
            data = np.vstack([file.imzml_2d_array for file in self.train_data])
            self.nmf.fit(data)
            data_list.append(self.nmf.transform(data))
        elif type == 'each':
            for file in self.files:
                X_projected, _ = file.performNMF(n_components=n_components)
                data_list.append(X_projected)
        return data_list


    def perform_kmeans(self, data_list, n_clusters = 3):
        labels_list = []
        for data in data_list:
            labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(data)
            labels_list.append(labels)
        return labels_list

    def plot_clusters(self, data_list, labels_list, fn_prefix='cph_'):
        colors = ['red', 'green', 'blue']
        for i, (data, labels) in enumerate(zip(data_list, labels_list)):
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
            cb = plt.colorbar()
            loc = np.arange(0, max(labels), max(labels)/float(len(colors)))
            cb.set_ticks(loc)
            cb.set_ticklabels(colors)
            plt.xlim(0, 0.1)
            plt.ylim(0, 0.1)
            plt.savefig(fn_prefix+'cluster_plot_'+str(i)+'.png')
            # plt.show()
            pass



if __name__=='__main__':
   cph_filenames = ['./dmsi0008.npy', './dmsi0011.npy']
   naive_filenames = ['./dmsi0009.npy', './dmsi0012.npy']
   cph_loader = DataLoader(cph_filenames)
   naive_loader = DataLoader(naive_filenames)

   cph_data_list = cph_loader.perform_nmf(n_components=20, type='each')
   cph_labels_list = cph_loader.perform_kmeans(cph_data_list, 3)
   cph_loader.plot_clusters(cph_data_list, cph_labels_list, 'cph_20')

   naive_data_list = naive_loader.perform_nmf(n_components=20, type='each')
   naive_labels_list = naive_loader.perform_kmeans(naive_data_list, 3)
   naive_loader.plot_clusters(naive_data_list, naive_labels_list, 'naive_20')
   pass