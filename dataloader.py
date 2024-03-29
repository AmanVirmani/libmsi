"""@package dataloader
This module is designed to load and process multiple msi datasets.

Capabilities:
1. Preprocessing: Binning and Normalization
2. Dimensionality Reduction (PCA/ICA/NMF)
3. Clustering (KMeans)
"""

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearnex import patch_sklearn
import libmsi


patch_sklearn()


class DataLoader:
    """
    @brief DataLoader class is designed to load and process mulitple msi datasets
    """

    def __init__(self, filenames=[], binSize=1, split_ratio=0.75, store=None):
        """
        @brief DataLoader class constructor
        @param filenames list of filenames of the datasets
        """
        self.filenames = filenames
        self.binSize = binSize
        self.files = self.load_files(store)
        # self.align_dataset()

        # self.preprocess('TIC')

        self.train_data, self.val_data = self.data_split(split_ratio)
        # print("train_data shape: ", self.train_data.shape)

    def load_files(self, store):
        """
        @brief method to load the datasets
        """
        files = []
        for file in self.filenames:
            files.append(libmsi.Imzml(file, self.binSize, store))
        return files

    def align_dataset(self):
        min_mz = sorted(self.files, key=operator.attrgetter("min_mz"))[0].min_mz
        min_mz -= min_mz % self.binSize
        max_mz = sorted(self.files, key=operator.attrgetter("max_mz"), reverse=True)[
            0
        ].max_mz
        max_mz -= max_mz % self.binSize
        # max_mz += self.binSize

        for file in self.files:
            file.min_mz -= file.min_mz % self.binSize
            file.max_mz -= file.max_mz % self.binSize
            pad_before = round((file.min_mz - min_mz) / self.binSize)
            pad_after = round((max_mz - file.max_mz) / self.binSize)
            file.imzml_array = np.pad(
                file.imzml_array, ((0, 0), (0, 0), (pad_before, pad_after)), "constant"
            )
            file.imzml_2d_array = file.create2dArray()
            file.min_mz = min_mz
            file.max_mz = max_mz
            # file.saveImzmlData()

    def saveImzmlData(self):
        for file in self.files:
            file.saveImzmlData()

    def data_split(self, split_ratio=0.75):
        """
        @brief method to split the data into training and validation set
        @param split_ratio ratio of the length of training data and validation data
        @return train_data list of Imzml objects for the training data
        @return val_data list of Imzml objects for the validation data
        """
        # random.shuffle(self.files)
        train_data = self.files[: int(split_ratio * len(self.files))]
        val_data = self.files[int(split_ratio * len(self.files)) :]
        return train_data, val_data

    def preprocess(self, normalization="TIC"):
        """
        @brief method to preprocess the data
        @param normalization normalization method to be used
        """
        for file in self.files:
            file.normalize_data(method=normalization)

    # Dimensionality Reduction
    def perform_pca(self, n_components=15):
        """
        @brief method to perform pca on all datasets together
        @param n_components no. of components to keep
        """
        self.pca = PCA(n_components=n_components)
        data = np.vstack([file.imzml_2d_array for file in self.train_data])
        self.pca.fit(data)

    def perform_ica(self, n_components=15):
        """
        @brief method to perform ica on all datasets together
        @param n_components no. of components to keep
        """
        self.ica = FastICA(n_components=n_components, random_state=0)
        data = np.vstack([file.imzml_2d_array for file in self.train_data])
        self.ica.fit(data)

    def generate_2d_dataset(self, data_set="train"):
        data_list = []
        if data_set == "train":
            dataset = self.training_data
        elif data_set == "val":
            dataset = self.val_data
        # find overall min_mz, max_mz
        max_mz = 0
        min_mz = -1
        for data in dataset:
            if max_mz < data.max_mz:
                max_mz = data.max_mz

            if min_mz == -1:
                min_mz = data.min_mz
            elif min_mz > data.min_mz:
                min_mz = data.min_mz
        self.min_mz = min_mz
        self.max_mz = max_mz
        # go to each data in dataset and append/prepend zeros in the 2d and 3d spectra of each pixel
        for data in dataset:
            zeros_to_prepend = (data.min_mz - min_mz) / self.binSize
            zeros_to_append = (max_mz - data.max_mz) / self.binSize

            for spectra in data.imzml_2d_array:
                data_list.append(
                    np.pad(spectra, (zeros_to_prepend, zeros_to_append), constant=0)
                )
        return np.array(data_list)

    def perform_nmf(self, n_components=15, type="all", input_data=None, n_iter=50):
        """
        @brief method to perform nmf on all datasets together
        @param n_components no. of components to keep
        @param type string value 'all' for combined data, 'each' for individual data
        @return data_list list of datasets in reduced dimensional space
        """
        data_list = []
        self.nmf = None
        if type == "all":
            data = (
                np.vstack([file.imzml_2d_array for file in self.files])
                if input_data is None
                else input_data
            )
            # self.nmf = newNMF()
            # X_transformed, components, info = self.nmf.run(data, n_components,
            #                                                max_iter=n_iter)

            self.nmf = NMF(
                n_components=n_components,
                init="random",
                random_state=0,
                max_iter=n_iter,
                verbose=1,
            )
            X_transformed = self.nmf.fit_transform(data)
            components = self.nmf.components_
            print("NMF done with error:", self.nmf.reconstruction_err_)
            start_idx = 0
            for i in range(len(self.files)):
                data_list.append(
                    X_transformed[
                        start_idx : start_idx + len(self.files[i].imzml_2d_array)
                    ]
                )
                start_idx += len(self.files[i].imzml_2d_array)
        elif type == "each":
            for file in self.files:
                X_projected, _ = file.performNMF(n_components=n_components)
                data_list.append(X_projected)
        return {"nmf_data_list": data_list, "nmf_component_spectra": components}

    def perform_kmeans(self, data_list, n_clusters=3):
        """
        @brief method to perform k means clustering
        @param data_list list of data sets to cluster
        @param n_clusters number of clusters to identify
        @return cluster_list list of cluster labels and centroids for each dataset
        """
        cluster_list = []
        for data in data_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
            cluster_list.append(
                {"labels": kmeans.labels_, "clusters": kmeans.cluster_centers_}
            )
        return cluster_list

    def align_clusters(self, cluster_lists):
        """
        @brief method to align indexes of the identified clusters
        @param cluster_lists list of cluster labels and centroids for each dataset
        @return cluster_lists list of cluster labels and centroids for each dataset
        """
        n_means = len(cluster_lists[0]["clusters"])
        data = np.vstack(
            [cluster_lists[i]["clusters"] for i in range(len(cluster_lists))]
        )
        km = KMeans(n_clusters=n_means, random_state=0).fit(data)
        for i in range(len(cluster_lists)):
            # first fix the cluster means
            clusters = np.empty_like(cluster_lists[i]["clusters"])
            labels = cluster_lists[i]["labels"].copy()
            for j in range(n_means):
                clusters[km.labels_[n_means * i + j]] = cluster_lists[i]["clusters"][j]
                cluster_lists[i]["labels"][labels == j] = km.labels_[n_means * i + j]
            cluster_lists[i]["clusters"] = clusters
        return cluster_lists

    def plot_tsne(self, data, labels, filename=None):
        """
        @brief Method to plot and save TSNE
        @param X_projected 2D Imzml data (valid pixels, no. of features)
        @param n_components no. of components to keep
        @param filename filename to save the plot of MS Image (optional)
        """
        tsne = TSNE(n_components=2, init="pca", n_jobs=-1)
        X_embedded = tsne.fit_transform(data)
        np.save("tsne_nmf_25_2_objects.npy", tsne)
        print("KL divergenc: ", tsne.kl_divergence_)
        red = labels == 1
        green = labels == 0
        plt.scatter(X_embedded[red, 0], X_embedded[red, 1], c="r")
        plt.scatter(X_embedded[green, 0], X_embedded[green, 1], c="g")
        plt.title("TSNE Visualization in 2D")
        # can choose to comment this and do: plt.savefig(filename)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def plot_clusters(self, data_list, cluster_list, fn_prefix="cph_"):
        """
        @brief method to plot the identified clusters
        @param data_list list of datasets to plot
        @param cluster_list list of cluster labels and centroids for each dataset
        @param fn_prefix prefix to the filename to save the cluster plot
        """
        colors = ["red", "green", "blue"]
        for i, (data, cluster_dict) in enumerate(zip(data_list, cluster_list)):
            fig = plt.figure(figsize=(8, 8))
            plt.scatter(
                data[:, 0],
                data[:, 1],
                c=cluster_dict["labels"],
                cmap=matplotlib.colors.ListedColormap(colors),
            )
            cb = plt.colorbar()
            loc = np.arange(
                0,
                max(cluster_dict["labels"]),
                max(cluster_dict["labels"]) / float(len(colors)),
            )
            cb.set_ticks(loc)
            cb.set_ticklabels(colors)
            plt.xlim(0, 0.1)
            plt.ylim(0, 0.1)
            plt.savefig(fn_prefix + "cluster_plot_" + str(i) + ".png")
            # plt.show()
            pass

    def plot_cluster_reprojection(self, cluster_list, type="each", fn_prefix="cph_"):
        """
        @brief method to plot the reprojection images of the clusters in reduced dimesions
        @param cluster_list list of cluster labels and centroids for each dataset
        @param fn_prefix prefix to the filename to save the cluster plot reprojection
        """
        for i, cluster_dict in enumerate(cluster_list):
            fig, axes = plt.subplots(2, 2)
            N = max(cluster_dict["labels"]) + 1
            coords = (
                np.array(self.files[i].coordinates)[:, 0:2] - 1
            )  # subtracted 1 because python indexes start
            n_rows = self.files[i].rows
            n_cols = self.files[i].cols
            im_frame = np.zeros((n_rows, n_cols))
            for temp in range(len(coords)):
                im_frame[coords[temp][1], coords[temp][0]] = cluster_dict["labels"][
                    temp
                ]

            cmap = plt.cm.jet
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # create the new map
            cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(0, N, N + 1)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

            im = axes[0, 0].imshow(im_frame, cmap=cmap)
            cb = fig.colorbar(im, ax=axes[0, 0], spacing="proportional", ticks=bounds)

            if type == "each":
                cluster_mean_spectra = cluster_dict["clusters"].dot(
                    self.files[i].nmf.components_
                )
            else:
                cluster_mean_spectra = cluster_dict["clusters"].dot(
                    self.nmf.components_
                )

            axes[0, 1].plot(cluster_mean_spectra[0], "b-")
            axes[1, 0].plot(cluster_mean_spectra[1], "g-")
            axes[1, 1].plot(cluster_mean_spectra[2], "r-")
            plt.savefig(fn_prefix + "cluster_reprojection_" + str(i) + ".png")


if __name__ == "__main__":
    cph_filenames = ["./dmsi0008.npy", "./dmsi0011.npy"]
    naive_filenames = ["./dmsi0009.npy", "./dmsi0012.npy"]
    cph_loader = DataLoader(cph_filenames, 1)
    naive_loader = DataLoader(naive_filenames, 1)

    cph_data_list = cph_loader.perform_nmf(n_components=2, type="all")
    cph_cluster_list = cph_loader.perform_kmeans(cph_data_list, 3)
    cph_cluster_list = cph_loader.align_clusters(cph_cluster_list)

    cph_loader.plot_clusters(cph_data_list, cph_cluster_list, "cph_2")
    cph_loader.plot_cluster_reprojection(cph_cluster_list, "cph_test_")
    cluster_mean_spectra = cph_cluster_list[0]["clusters"].dot(
        cph_loader.files[0].nmf.components_
    )
    plt.plot(cluster_mean_spectra[0], cluster_mean_spectra[1], "b.")
    plt.savefig("cph_mean_spectra_0.jpg")
    cluster_mean_spectra = cph_cluster_list[1]["clusters"].dot(
        cph_loader.files[1].nmf.components_
    )
    plt.plot(cluster_mean_spectra[0], cluster_mean_spectra[1], "b.")
    plt.savefig("cph_mean_spectra_1.jpg")

    naive_data_list = naive_loader.perform_nmf(n_components=2, type="each")
    naive_cluster_list = naive_loader.perform_kmeans(naive_data_list, 3)
    # # naive_loader.plot_clusters(naive_data_list, naive_labels_list, 'naive_20')
    # naive_loader.plot_cluster_reprojection(naive_labels_list, 'naive_')
    cluster_mean_spectra = naive_cluster_list[0]["clusters"].dot(
        naive_loader.files[0].nmf.components_
    )
    plt.plot(cluster_mean_spectra[0], cluster_mean_spectra[1], "b.")
    plt.savefig("naive_mean_spectra_0.jpg")
    cluster_mean_spectra = naive_cluster_list[1]["clusters"].dot(
        naive_loader.files[1].nmf.components_
    )
    plt.plot(cluster_mean_spectra[0], cluster_mean_spectra[1], "b.")
    plt.savefig("naive_mean_spectra_1.jpg")

    naive_loader.files[0].performTSNE(naive_data_list[0], filename="TSNE_naive_0.jpg")
    pass
