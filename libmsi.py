from pyimzml.ImzMLParser import ImzMLParser as read_msi
from pyimzml.ImzMLWriter import ImzMLWriter as write_msi
from pyimzml.ImzMLParser import getionimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE
import pandas as pd
import random
from tqdm import tqdm, trange


class Imzml:

    def __init__(self, filename="", bin_size=1):
        self.filename = filename
        self.binSize = bin_size
        if self.filename.split('.')[-1] == "imzML":
            self.imzml = self.readImzml()
            self.coordinates = self.imzml.coordinates
            self.min_mz = self.imzml.getspectrum(0)[0].min()
            self.max_mz = self.imzml.getspectrum(0)[0].max()
            self.cols = self.imzml.imzmldict["max count of pixels x"]
            self.rows = self.imzml.imzmldict["max count of pixels y"]
            self.imzml_array = self.convertImzml2Array(min_mz=self.min_mz, max_mz=self.max_mz)

        elif self.filename.split('.')[-1] == "npy":
            data = self.readImzmlArray()
            self.coordinates = data[()]['coordinates']
            self.min_mz = data[()]['min_mz']
            self.max_mz = data[()]['max_mz']
            self.imzml_array = data[()]['imzml_array']
            self.rows, self.cols, _ = np.shape(self.imzml_array)

        self.imzml_2d_array = self.create2dArray()

    def readImzml(self):
        return read_msi(self.filename)

    def readImzmlArray(self):
        return np.load(self.filename, allow_pickle=True)

    def saveImzmldata(self, filename=None):
        data = {'min_mz': self.min_mz, 'max_mz': self.max_mz, 'imzml_array': self.imzml_array, 'coordinates': self.coordinates}
        np.save(filename, data)

    def convertImzml2Array(self, min_mz, max_mz):
        x_list = []
        min_mz -= min_mz % self.binSize
        max_mz -= max_mz % self.binSize
        max_mz += self.binSize

        n_ch = int((max_mz - min_mz) / self.binSize)
        X = np.zeros([self.rows, self.cols, n_ch])

        print('\nConverting imzml format to binned 3d array with binSize' + str(self.binSize) + '\n')
        for i, (x, y, z) in enumerate(tqdm(self.imzml.coordinates)):
            mzA, intA = self.imzml.getspectrum(i)
            current_bin = 600
            Y = {}
            for j, mz in enumerate(mzA):
                if mz > max_mz:
                    break
                elif current_bin + self.binSize <= mz:
                    current_bin += self.binSize
                mz = current_bin
                if mz in Y.keys():
                    if Y[mz] < intA[j]:
                        Y[mz] = intA[j]
                else:
                    Y[mz] = intA[j]
            X[y - 1, x - 1] = pd.DataFrame([Y]).values
        return X

    def create2dArray(self):
        return self.imzml_array[~np.all(self.imzml_array == 0, axis=-1)]

    def reconstruct_3d_array(self, x_2d):
        x_3d = np.zeros([self.rows, self.cols, x_2d.shape[-1]])
        for i, (x, y, z) in enumerate(self.coordinates):
           x_3d[y-1, x-1] = x_2d[i]
        return x_3d

    def normalize_data(self, method='TIC'):
        if method == 'TIC':
            for (x, y, z) in self.coordinates:
                tic = sum(self.imzml_array[y-1, x-1])
                self.imzml_array[y-1, x-1] /= tic
        self.imzml_2d_array = self.create2dArray()

    # Dimensionality Reduction
    def performPCA(self, n_components=15):
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(self.imzml_2d_array)
        X_projected = self.pca.inverse_transform(X_train_pca)
        X_projected = self.reconstruct_3d_array(X_projected)
        return X_train_pca, X_projected

    def performICA(self, n_components=15):
        self.ica = FastICA(n_components=n_components, random_state=0)
        X_train_ica = self.ica.fit_transform(self.imzml_2d_array)
        X_projected = self.ica.inverse_transform(X_train_ica)
        X_projected = self.reconstruct_3d_array(X_projected)
        return X_train_ica, X_projected

    def performNMF(self, n_components=15):
        self.nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=20000)
        X_train_nmf = self.nmf.fit_transform(self.imzml_2d_array)
        X_projected = self.nmf.inverse_transform(X_train_nmf)
        X_projected = self.reconstruct_3d_array(X_projected)
        return X_train_nmf, X_projected

    # Visualization
    def plotMSI(self, mz, filename=None):
        plt.figure()
        plt.imshow(self.imzml_array[:, :, mz]).set_interpolation('nearest')
        plt.colorbar()
        plt.title('MS image for ' + str(mz) + ' mz value')
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def plotSpectra(self, index, filename=None):
        plt.plot(self.imzml_2d_array[index], 'b-')
        plt.title('MS spectra at ' + str(index) + 'index ')
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def plotTSNE(self, X_projected, n_components=2, filename=None):
        X_embedded = TSNE(n_components=2).fit_transform(X_projected)
        plt.plot(X_embedded[:, 0], X_embedded[:, 1], 'b.')
        plt.title('TSNE Visualization in 2D')
        # can choose to comment this and do: plt.savefig(filename)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()
        return X_embedded

    def plotSpectraReprojected(self, X, X_reprojected, filename=None):
        for i in random.sample(range(len(X_reprojected)), 100):
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(X[i], 'b-')
            ax1.set_title('original')
            ax2.plot(X_reprojected[i], 'r-')
            ax2.set_title('projected')
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename)
                plt.close()

    def plotMSIReprojection(self, X_3d, X_projected_3d, mz, filename=None):
        f, axes = plt.subplots(1, 2, sharey=True)
        im = axes[0].imshow(X_3d[:, :, mz])  # .set_interpolation('nearest')
        axes[0].set_title('Original MSI')
        axes[1].imshow(X_projected_3d[:, :, mz])  # .set_interpolation('nearest')
        axes[1].set_title('Reconstructed MSI')
        # axes[0].cax.colorbar(im)
        f.colorbar(im, ax=axes.ravel().tolist())
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    # Peak Processing
    def get_peaks(self, plot_peaks=True, filename=None):
        X = self.imzml_2d_array.copy()
        X_max = int(X.max() + 0.5)
        X_min = int(X.min())

        data = []
        delta = 1e-3
        #
        th = int((X_min + X_max) / 2)
        while (True):
            if 1 - np.sum(X < th) / X.size < delta:
                new_th = (X_min + th) / 2
                X_max = th
            else:
                new_th = (th + X_max) / 2
                X_min = th
            if th == int(new_th):
                break
            else:
                th = int(new_th)

        # for th in range(X_min, X_max):
        #     if 1-np.sum(X < th)/X.size < delta:
        #         break

        X[X < th] = 0

        # List of peak values
        self.peak_mzs = np.nonzero(np.mean(X, axis=0))  # + self.min_mz

        if plot_peaks:
            print('Plotting peaks for ' + self.filename.split('/')[-1] + '......')
            plt.figure()
            for i in trange(len(X)):
                plt.plot(X[i, :], 'b')
                # data.append([i, X])
            # data = np.array(data)
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename)
                plt.close()
        return X
