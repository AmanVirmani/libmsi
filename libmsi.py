"""
@package libmsi
This module is designed to read and process msi data.

Capabilities:

1. Preprocessing: Binning and Normalization
2. Dimensionality Reduction (PCA/ICA/NMF)
3. Clustering (KMeans)
"""
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
import pickle


class Imzml:
    """
    @brief Imzml class is designed to load, process, and visualize MS spectra and Image data
    """

    def __init__(self, filename="", bin_size=1, store=None):
        """
        @brief The constructor for Imzml class
        @param filename input file to be read
        @param bin_size bin size to be used for binning
        """
        self.filename = filename
        self.binSize = bin_size
        self.imzml_2d_array = None
        if self.filename.split(".")[-1] == "imzML":
            self.imzml = self.readImzml()
            self.coordinates = self.imzml.coordinates
            self.min_mz = self.imzml.getspectrum(0)[0].min()
            self.max_mz = self.imzml.getspectrum(0)[0].max()

            self.min_mz -= self.min_mz % self.binSize
            self.max_mz -= self.max_mz % self.binSize

            self.cols = self.imzml.imzmldict["max count of pixels x"]
            self.rows = self.imzml.imzmldict["max count of pixels y"]
            self.imzml_array = self.convertImzml2Array(
                min_mz=self.min_mz, max_mz=self.max_mz
            )

        elif self.filename.split(".")[-1] == "npy":
            data = self.readImzmlArray()
            print(type(data))
            # self.coordinates = data["coordinates"]
            # self.min_mz = data["min_mz"]
            # self.max_mz = data["max_mz"]
            # self.imzml_array = data["imzml_array"]
            self.coordinates = data[()]["coordinates"]
            self.min_mz = data[()]["min_mz"]
            self.max_mz = data[()]["max_mz"]
            self.imzml_array = data[()]["imzml_array"]
            self.rows, self.cols = np.shape(self.imzml_array)[:2]

        # elif self.filename.split('.')[-1] == "pickle":
        else:
            data = self.readImzmlArray()
            self.coordinates = data["coordinates"]
            self.min_mz = data["min_mz"]
            self.max_mz = data["max_mz"]
            self.imzml_array = data["imzml_array"]  # [:,:,:10000]
            self.rows, self.cols = np.shape(self.imzml_array)[:2]

        self.create2dArray(store)
        self.normalize_data()
        # self.saveImzmlData()
        print("loaded ", self.filename)

    def readImzml(self):
        """
        @brief Method to read data in .imzML format
        """
        return read_msi(self.filename)

    def readImzmlArray(self):
        """
        @brief Method to read data in .npy format
        """
        if "npy" in self.filename:
            return np.load(self.filename, allow_pickle=True)
        with open(self.filename, "rb") as fh:
            data = pickle.load(fh)
        return data

    def saveImzmlData(self, filename=None):
        """
        @brief Method to save processed data in .npy format
        @param filename output filename to save the data in
        """
        # TODO: put binsize here as well
        data = {
            "min_mz": self.min_mz,
            "max_mz": self.max_mz,
            "imzml_array": self.imzml_2d_array,
            "coordinates": self.coordinates,
        }
        if filename is None:
            filename = self.filename.replace(
                ".imzML", "_" + str(self.binSize).replace(".", "_") + ".pickle"
            )
        # np.save(filename, data)
        print("saving: ", filename.split("/")[-1])
        with open(filename, "wb") as fh:
            pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
        print("saved: ", filename.split("/")[-1])

    def convertImzml2Array(self, min_mz, max_mz):
        """
        @brief Method to bin the imzml data to a 3d array format
        @param min_mz minimum mz value to be used for binning
        @param max_mz maximum mz value to be used for binning
        """
        x_list = []
        min_mz -= min_mz % self.binSize
        max_mz -= max_mz % self.binSize
        # max_mz += self.binSize

        # n_ch = int((max_mz - min_mz) / self.binSize) + 1 # ceiling value
        n_ch = int((max_mz - min_mz) / self.binSize)  # ceiling value
        X = np.zeros([self.rows, self.cols, n_ch])

        print(
            "\nConverting imzml format to binned 3d array with binSize"
            + str(self.binSize)
            + "\n"
        )
        for i, (x, y, z) in enumerate(tqdm(self.imzml.coordinates)):
            mzA, intA = self.imzml.getspectrum(i)
            current_bin = min_mz
            Y = {}
            for j, mz in enumerate(mzA):
                if mz > max_mz:
                    break
                else:
                    ## inserting while loop to skip missing peaks in that bin
                    while current_bin + self.binSize <= mz:
                        current_bin += self.binSize
                mz = current_bin
                ## following part is buggy
                # if mz in Y.keys():
                #     if Y[mz] < intA[j]:
                #         Y[mz] = intA[j]
                # else:
                #     Y[mz] = intA[j]
                ## Alternate Way
                bin = round((mz - min_mz) / self.binSize)
                if X[y - 1, x - 1, bin] < intA[j]:
                    X[y - 1, x - 1, bin] = intA[j]

            # X[y - 1, x - 1] = pd.DataFrame([Y]).values
        return X

    def create2dArray(self, store=None):
        """
        @brief Method to get a list of valid pixel spectra in 2d array format
        """
        self.imzml_2d_array = self.imzml_array[~np.all(self.imzml_array == 0, axis=-1)]
        if store is not None:
            self.imzml_2d_array = self.imzml_2d_array[:, :store]
        # offload 3d array from memory
        del self.imzml_array

    def reconstruct_3d_array(self, x_2d):
        """
        @brief Method to reconstruct the 3d image array from 2d valid pixel spectra
        @param x_2d 2D array with valid pixel spectra
        """
        x_3d = np.zeros([self.rows, self.cols, x_2d.shape[-1]])
        for i, (x, y, z) in enumerate(self.coordinates):
            x_3d[y - 1, x - 1] = x_2d[i]
        return x_3d

    def normalize_data(self, method="TIC"):
        """
        @brief Method to normalize imzml data
        @param method the normalization method to be used.
        """
        if method == "TIC":
            # print("saving TIC binned data in", self.filename)
            for i in range(len(self.imzml_2d_array)):
                ## TODO update it to use 2d array for better runtimes
                tic = sum(self.imzml_2d_array[i])
                if tic == 0:
                    continue
                self.imzml_2d_array[i] /= tic
        # self.imzml_array = None

    # Dimensionality Reduction
    def performPCA(self, n_components=15):
        """
        @brief Method to perform Principal Component Analysis on the Imzml data
        @param n_components number of components to keep in the Imzml data
        """
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(self.imzml_2d_array)
        X_projected = self.pca.inverse_transform(X_train_pca)
        X_projected = self.reconstruct_3d_array(X_projected)
        return X_train_pca, X_projected

    def performICA(self, n_components=15):
        """
        @brief Method to perform Independent Component Analysis on the Imzml data
        @param n_components number of components to keep in the Imzml data
        """
        self.ica = FastICA(n_components=n_components, random_state=0)
        X_train_ica = self.ica.fit_transform(self.imzml_2d_array)
        X_projected = self.ica.inverse_transform(X_train_ica)
        X_projected = self.reconstruct_3d_array(X_projected)
        return X_train_ica, X_projected

    def performNMF(self, n_components=15):
        """
        @brief Method to perform Non-Negative Matrix Factorization on the Imzml data
        @param n_components number of components to keep in the Imzml data
        """
        self.nmf = NMF(
            n_components=n_components, init="random", random_state=0, max_iter=20000
        )
        X_train_nmf = self.nmf.fit_transform(self.imzml_2d_array)
        X_projected = self.nmf.inverse_transform(X_train_nmf)
        X_projected = self.reconstruct_3d_array(X_projected)
        return X_train_nmf, X_projected

    # Visualization
    def plotMSI(self, mz, filename=None):
        """
        @brief Method to plot and save MS Image
        @param mz m/z value for the desired MS Image
        @param filename filename to save the plot of MS Image (optional)
        """
        idx = round((mz - self.min_mz) / self.binSize)
        print("MS image for {0} mz value at {1}", mz, idx)
        plt.figure()
        plt.imshow(self.imzml_array[:, :, idx]).set_interpolation("nearest")
        plt.colorbar()
        # plt.title("MS image for {0} mz value at {1}", mz, idx)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def plotSpectra(self, index, filename=None):
        """
        @brief Method to plot and save MS spectrum
        @param index index of the desired spectra in valid pixels list
        @param filename filename to save the plot of MS Image (optional)
        """
        plt.plot(self.imzml_2d_array[index], "b-")
        plt.title("MS spectra at " + str(index) + "index ")
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def plotTSNE(self, X_projected, n_components=2, filename=None):
        """
        @brief Method to plot and save TSNE
        @param X_projected 2D Imzml data (valid pixels, no. of features)
        @param n_components no. of components to keep
        @param filename filename to save the plot of MS Image (optional)
        """
        X_embedded = TSNE(n_components=2).fit_transform(X_projected)
        plt.plot(X_embedded[:, 0], X_embedded[:, 1], "b.")
        plt.title("TSNE Visualization in 2D")
        # can choose to comment this and do: plt.savefig(filename)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()
        return X_embedded

    def plotSpectraReprojected(self, X, X_reprojected, filename=None):
        """
        @brief Method to plot and save the original and reprojected spectra(randomly selected 100 spectra)
        @param X original Imzml spectral data
        @param X_reprojected reprojected Imzml spectral data
        @param filename filename to save the plot of MS spectra (optional)
        """
        for i in random.sample(range(len(X_reprojected)), 100):
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(X[i], "b-")
            ax1.set_title("original")
            ax2.plot(X_reprojected[i], "r-")
            ax2.set_title("projected")
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename.split(".")[0] + "_" + str(i) + ".png")
                plt.close()

    def plotMSIReprojection(self, X_3d, X_projected_3d, mz, filename=None):
        """
        @brief Method to plot and save the original and reprojected images(analysis output)
        @param X_3d original 3D Imzml data
        @param X_projected_3d reprojected 3D Imzml data
        @param mz m/z value to be used for comparison
        @param filename filename to save the plot of MS Image (optional)
        """
        f, axes = plt.subplots(1, 2, sharey=True)
        im = axes[0].imshow(X_3d[:, :, mz])  # .set_interpolation('nearest')
        axes[0].set_title("Original MSI")
        axes[1].imshow(X_projected_3d[:, :, mz])  # .set_interpolation('nearest')
        axes[1].set_title("Reconstructed MSI")
        # axes[0].cax.colorbar(im)
        f.colorbar(im, ax=axes.ravel().tolist())
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    # Peak Processing
    def get_peaks(self, plot_peaks=True, filename=None):
        """
        @brief Method to calculate dominant peaks in the MSI dataset
        @param plot_peaks bool value to decide whether to plot the prominent peaks in Imzml data
        @param filename filename to save the plot of MS peak m/z spectra (optional)
        @return list of m/z values corresponding to prominent peaks in the Imzml data
        """
        X = self.imzml_2d_array.copy()
        X_max = int(X.max() + 0.5)
        X_min = int(X.min())

        data = []
        delta = 1e-3
        th = int((X_min + X_max) / 2)
        while True:
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

        X[X < th] = 0

        # List of peak values
        self.peak_mzs = np.nonzero(np.mean(X, axis=0))  # + self.min_mz

        if plot_peaks:
            print("Plotting peaks for " + self.filename.split("/")[-1] + "......")
            plt.figure()
            for i in trange(len(X)):
                plt.plot(X[i, :], "b")
                # data.append([i, X])
            # data = np.array(data)
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename)
                plt.close()
        return self.peak_mzs
