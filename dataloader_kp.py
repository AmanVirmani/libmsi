from sklearnex import patch_sklearn
patch_sklearn()

import sys
# sys.path.insert(1,'/home/kasun/aim_hi_project_kasun/kp_libmsi/libmsi')
# import dataloader as d

import libmsi
import numpy as np
import random as rd
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLParser import getionimage

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.preprocessing import minmax_scale
from mpl_toolkits import mplot3d


from itertools import combinations

from skimage import color
from skimage.transform import rescale, resize, downscale_local_mean

import pandas as pd
import pickle

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
import os

# import cuml.manifold.t_sne as cuml_TSNE

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from scipy.signal import find_peaks

from PIL import Image

import cv2

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

rd.seed(0)

class dataloader_kp:

    def __init__(self, global_path_name, dataset_order=[], msi_filename_array=[], he_filename_array=[], bin_size=1, tic_truncate_data=1, tic_add_dummy_data=0, tic_truncate_quantity=500):

        """

        @brief Initializes the dataloader_kp object
            Example usage:
                            from dataloader_kp import dataloader_kp
                            my_os = 'windows'
                            bin_size = 1
                            from msi_file_names import msi_filenames
                            from he_file_names import he_filenames
                            msi_filename_array, dataset_order, global_path_name = msi_filenames(my_os=my_os)
                            he_filename_array = he_filenames(my_os=my_os)
                            tic_truncate_data = 1
                            tic_truncate_quantity = 500
                            tic_add_dummy_data = 0

                            msi_data = dataloader_kp(global_path_name, dataset_order=dataset_order ,msi_filename_array=msi_filename_array, he_filename_array=he_filename_array, bin_size=bin_size, tic_truncate_data=tic_truncate_data, tic_add_dummy_data=tic_add_dummy_data, tic_truncate_quantity=tic_truncate_quantity)


        @param dataset_order This is the order of cph and naive datasets.
         I have this as an input so that if we ever get a lot more data, we can externally adjust the order in which
         our datasets order
        @param msi_filename_array This is the msi filepaths list. The names should be arranged in the order given by the
            the 'dataset_order' variable. The files should have been created by libmsi .
        @param he_filename_array This is the h&e stained image filepaths array. This should also be arranged in the order
            given by the 'dataset_order' variable.
        @param tic_truncate_data If set to 1, data will be truncated starting the lowest mz till lowest mz plus truncate quantity
        @param tic_truncate_quantity This is the range of mz data that need to be preserved. i.e, the data starting the lowest mz till the
            lowest mz plus tic truncate quantity will be preserved, while the rest will be removed
        @param tic_add_dummy_data Add dummy data if we want some proof data in our dataset to test a specific algorithm

        """

        self.msi_filename_array = msi_filename_array
        self.he_filename_array = he_filename_array
        self.dataset_order = dataset_order
        self.bin_size = bin_size
        self.global_path_name = global_path_name

        for i, dataset in enumerate(self.dataset_order):
            exec('self.' + dataset + '_binned' + '= libmsi.Imzml(self.msi_filename_array[i], bin_size=self.bin_size)')

        self.combined_data_object = []
        for i, dataset in enumerate(self.dataset_order):
            self.combined_data_object.append(eval('self.' + dataset + '_binned'))

        self.max_mz_of_original_data = self.combined_data_object[0].max_mz
        self.min_mz_of_original_data = self.combined_data_object[0].min_mz
        self.is_reduced_memory_version = 0

        self.normalize_tic(truncate_data=tic_truncate_data, truncate_quantity=tic_truncate_quantity, add_dummy_data=tic_add_dummy_data)

        # else:
        #     temp_self = np.load(use_reduced_memory_version,allow_pickle=True)[()]
        #     if temp_self.is_reduced_memory_version == 0:
        #         print("Error in saved compact msi data object file")
        #     else:
        #         self = temp_self

    def normalize_tic(self, truncate_data=1, truncate_quantity=500, add_dummy_data=0):

        """
        @brief TIC Normalization of each dataset separately. Also does truncating of the original data, and adding dummy data
            if necessary
            Example  Usage:     truncate_original_data = 1
                                truncate_quantity = 500
                                include_dummy_data = 0
                                tic_normalized_combined_data_object = msi_data.normalize_tic(truncate_data=truncate_original_data, truncate_quantity=truncate_quantity, add_dummy_data=include_dummy_data)


        @param truncate_data Set this to 1 to truncate data. If this is set to 0, data will not be truncated
        @param  truncate_quantity Set this to the mz value range that we want to keep. This value is ALWAYS set with a binsize of 1 in mind.
            If the binsize is different, the appropriate number of samples that need to be retained will be automatically calculated.
            The truncate_quantity is actually the number of mz values we want to keep. Truncation begins at the smallest available mz value, and
            goes up in integer values of 1. Therefore, for a bin size of 0.2, and a truncate_quantity of 500, the range of indices
            that will be preserved after truncation is [0  :  (500/0.2)] = [0 : 2500]
        @param add_dummy_data If this parameter is set to 1, a dummmy dataset is appended to the set of datasets.
            This was initially used to verify how TSNE was performing
        @return Returns a combined_data_object that is TIC normalized, (and if necessary truncated, with dummy data included)
        """
        self.truncate_data = truncate_data
        self.add_dummy_data = add_dummy_data
        self.truncate_quantity = truncate_quantity
        self.combined_tic_normalized_data = []
        temp = self.combined_data_object
        dummy_dataset = np.empty((0, temp[0].create2dArray().shape[1]))
        for count, binned_data_object in enumerate(self.combined_data_object):
            binned_dataset = binned_data_object.create2dArray()
            self.combined_tic_normalized_data.append(binned_dataset / np.sum(binned_dataset, axis=1, keepdims=True))
            dummy_dataset = np.append(dummy_dataset, np.squeeze(rd.choices(binned_dataset, k=4000)), axis=0)

        ## If necessary, add dummy data, and Do TIC as well.
        if add_dummy_data == 1:
            self.combined_tic_normalized_data.append(dummy_dataset / np.sum(dummy_dataset, axis=1, keepdims=True))

            ## Set a certain range of mz values to zero after TIC normalization. This is NOT the same as truncating.
            ## This is to make the dummy data have a spectrum that is manipulated according to our needs
            self.combined_tic_normalized_data[-1][:, 100:300] = 0

        if truncate_data == 1:
            for count, dataset in enumerate(self.combined_tic_normalized_data):
                self.combined_tic_normalized_data[count] = self.combined_tic_normalized_data[count][:, 0:int(truncate_quantity/self.bin_size)]

        self.min_mz_after_truncation = self.min_mz_of_original_data
        self.max_mz_after_truncation = self.min_mz_of_original_data + self.truncate_quantity

        return self.combined_tic_normalized_data

    def create_reduced_memory_footprint_file(self, path_and_filename):

        """
        @brief This function reduces the memory requirement when I load a libmsi object. Instead, when I run this,
            I will have a reduced footprint version of the libmsi objects that only contains the important stuff in the
            libmsi object like the dimensions of datasets etc, but not the data.

            Example usage:

                            compact_msi_data = msi_data.create_reduced_memory_footprint_file("D:/msi_project_data/binned_binsize_1/compact_data_object/compact_msi_data.npy")

        @param path_and_filename Enter the destination path where the compact data object should be saved.
        @return Returns a compact version of the msi_data object.
        """

        self.is_reduced_memory_version = 1
        for i, dataset in enumerate(self.dataset_order):
            exec('del self.' + dataset + '_binned')

        for i, dataset in enumerate(self.dataset_order):
            del self.combined_data_object[i].imzml_array

        np.save(path_and_filename, self, 'dtype=object')
        print("saved compact msi data object")

        return self

    def data_subset_creator(self, subset_pattern_array=[1, 1, 1, 1, 1, 1, 1, 1]):
        """
        @brief Depending on the positions of 1s in the subset_pattern_array, those corresponding data out of the tic_normalized_combined_data_array data is added to the output

            Example usage:  subset_pattern_array = [1, 0, 0, 0, 0, 0, 0, 0]
                            data_subset = pca_kp_object.data_subset_creator(subset_pattern_array=subset_pattern_array)


        @param subset_pattern_array: The datasets that we want included in the analysis are chosen here.
        @return: the subset of data that was chosen.
        """

        tic_normalized_combined_data_array_subset = []

        for count, dataset in enumerate(self.combined_tic_normalized_data):
            if subset_pattern_array[count] == 1:
                tic_normalized_combined_data_array_subset.append(dataset)

        data_subset_dict = {'tic_normalized_combined_data_array_subset': tic_normalized_combined_data_array_subset,
                                 'subset_pattern_array': subset_pattern_array}

        return data_subset_dict






