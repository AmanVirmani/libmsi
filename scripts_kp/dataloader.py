from sklearnex import patch_sklearn

patch_sklearn()

# sys.path.insert(1,'/home/kasun/aim_hi_project_kasun/kp_libmsi/libmsi')
# import dataloader as d

import libmsi_kp
import numpy as np
import random as rd
from pyimzml.ImzMLParser import ImzMLParser

from sklearn.preprocessing import minmax_scale

from itertools import combinations

from skimage import color

import pandas as pd
import pickle

from sklearn.cluster import KMeans

import os

# import cuml.manifold.t_sne as cuml_TSNE

from scipy.signal import find_peaks

from PIL import Image

import cv2

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

import scipy
import time

import multiprocessing

rd.seed(0)


class dataloader_kp:

    def __init__(self, global_path_name, perform_binning=0, binned_files_prefix='', dataset_order=[], imzml_filename_array=[], msi_filename_array=[], he_filename_array=[], bin_size=1, mode='negative', tic_truncate_data=1, tic_add_dummy_data=0, tic_truncate_quantity=500):

        """

        @brief Initializes the dataloader_kp object
            Example usage:
                            from dataloader_kp import dataloader_kp
                            my_os = 'windows'
                            bin_size = 1
                            mode = 'negative'
                            from msi_filenames import msi_filenames
                            from he_filenames import he_filenames
                            msi_filename_array, dataset_order, global_path_name, imzml_filename_array = msi_filenames(my_os=my_os, mode=mode, bin_size=bin_size)
                            perform_binning = 0
                            binned_files_prefix = ''
                            he_filename_array = he_filenames(my_os=my_os, mode=mode)
                            tic_truncate_data = 1
                            tic_truncate_quantity = 500
                            tic_add_dummy_data = 0

                            msi_data = dataloader_kp(global_path_name, perform_binning=perform_binning, binned_files_prefix=binned_files_prefix, dataset_order=dataset_order ,imzml_filename_array=imzml_filename_array, msi_filename_array=msi_filename_array, he_filename_array=he_filename_array, bin_size=bin_size, mode=mode, tic_truncate_data=tic_truncate_data, tic_add_dummy_data=tic_add_dummy_data, tic_truncate_quantity=tic_truncate_quantity)


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
        @param perform_binning If this is set to 1, re-run the binning algorithm, and save the binned result.
        @param binned_files_prefix If this is given, and perform_binning is set to 1, the binned files will be saved with this prefix.
        @param imzml_filename_array This is used if the perform_binning is set to 1.

        """

        self.imzml_filename_array = imzml_filename_array
        self.msi_filename_array = msi_filename_array
        self.he_filename_array = he_filename_array
        self.dataset_order = dataset_order
        self.bin_size = bin_size
        self.mode = mode
        self.global_path_name = global_path_name
        self.is_reduced_memory_version = 0
        self.is_double_memory_reduced_version = 0


        if perform_binning == 1:
            self.perform_binning_parallel(imzml_filename_array, binned_files_prefix=binned_files_prefix)

        self.create_combined_tic_normalized_data_list_and_combined_data_objects(truncate_data=tic_truncate_data, truncate_quantity=tic_truncate_quantity, add_dummy_data=tic_add_dummy_data, low_overhead=1)


    def determine_global_min_and_max_mz(self, imzml_filename_array):
        """
        @brief This method loads all the raw imzml datasets and checks what the minimum and maximum mz values of all the datasets are.
                This is necessary because each dataset has different minimum and maximum mz values for their spectra (see positive mode data)


        @param imzml_filename_array Enter the filenames of the raw imzml files in an array.
        @return minimum and maximum mz values of all the datasets
        """
        min_mz_all_datasets = np.Inf
        max_mz_all_datasets = 0
        for count, this_imzml_filename in enumerate(imzml_filename_array):
            print("Calculating min and max mz values from dataset ", count)
            this_imzml_dataset = ImzMLParser(this_imzml_filename)
            num_pixels_in_this_imzml = len(this_imzml_dataset.mzLengths)
            for i in range(num_pixels_in_this_imzml):
                this_spectrum = this_imzml_dataset.getspectrum(i)[0]
                this_spectrum_min_mz = this_spectrum.min()
                this_spectrum_max_mz = this_spectrum.max()
                min_mz_all_datasets = np.min([this_spectrum_min_mz, min_mz_all_datasets])
                max_mz_all_datasets = np.max([this_spectrum_max_mz, max_mz_all_datasets])

        return min_mz_all_datasets, max_mz_all_datasets

    def perform_binning_parallel(self, imzml_filename_array, binned_files_prefix=''):

        """
        @brief This method exploits multiprocessing to parallely perform the binning of datasets.
                Generates binned datasets and saves them to disk


        @param imzml_filename_array: An array containing the filepaths of each raw imzml file for all datasets
        @param binned_files_prefix: This can be used if I want to prepend a prefix to the standard name of imzml files when saving binned filenames.

        """

        start_time = time.time()

        print("Determining global max and min mz values")
        self.min_mz_all_datasets, self.max_mz_all_datasets = self.determine_global_min_and_max_mz(imzml_filename_array)

        print("Determining global max and min mz values complete")

        intermediate_stop_time = time.time()
        intermediate_duration = (intermediate_stop_time - start_time) / 60
        print("Global max and min mz determining process took ", intermediate_duration, " minutes")

        print("Starting the binning process")
        start_time = time.time()
        temp_array = []
        for dataset_number, this_imzml_filename in enumerate(imzml_filename_array):
            this_imzml_dataset = ImzMLParser(this_imzml_filename)
            temp_array.append(multiprocessing.Process(target=self.bin_individual_imzml_to_2d_array, args=(this_imzml_dataset, self.min_mz_all_datasets, self.max_mz_all_datasets, dataset_number, this_imzml_filename, binned_files_prefix)))

        for i in temp_array:
            i.start()

        for i in temp_array:
            i.join()

        stop_time = time.time()
        duration = (stop_time - start_time) / 60  # In minutes
        print("Binning process took ", duration, " minutes")
        print("Binning process complete")

    def perform_binning(self, imzml_filename_array, binned_files_prefix=''):

        """
        @brief THIS METHOD IS OBSOLETE and DEPRACATED.
                Generates binned datasets and saves them to disk

        @param imzml_filename_array: An array containing the filepaths of each raw imzml file for all datasets
        @param binned_files_prefix: This can be used if I want to prepend a prefix to the standard name of imzml files when saving binned filenames.

        """

        # start_time = time.time()
        #
        # print("Determining global max and min mz values")
        # self.min_mz_all_datasets, self.max_mz_all_datasets = self.determine_global_min_and_max_mz(imzml_filename_array)
        #
        # print("Determining global max and min mz values complete")
        #
        # intermediate_stop_time = time.time()
        # intermediate_duration = (intermediate_stop_time - start_time) / 60
        # print("Global max and min mz determining process took ", intermediate_duration, " minutes")
        #
        # print("Starting the binning process")
        # start_time = time.time()
        # for dataset_number, this_imzml_filename in enumerate(imzml_filename_array):
        #     this_imzml_dataset = ImzMLParser(this_imzml_filename)
        #     self.bin_individual_imzml_to_2d_array(this_imzml_dataset, self.min_mz_all_datasets, self.max_mz_all_datasets, dataset_number, this_imzml_filename, binned_files_prefix=binned_files_prefix)
        #
        # stop_time = time.time()
        # duration = (stop_time - start_time) / 60  # In minutes
        # print("Binning process took ", duration, " minutes")
        # print("Binning process complete")

    def bin_individual_imzml_to_2d_array(self, imzml_dataset, min_mz_all_datasets, max_mz_all_datasets, dataset_number, this_imzml_filename, binned_files_prefix=''):

        """
        @brief This code is used with porallel computing techniques as a worker function to do the binning of imzml data

        @param imzml_dataset: A libmsi_kp object containing imzml data of a given dataset
        @param min_mz_all_datasets: The global minimum mz value in ALL datasets
        @param max_mz_all_datasets: The global maximum mz value in ALL datasets
        @param dataset_number: Just a process identifier to avoid jumbled outputs during multiprocessing
        @param this_imzml_filename: Filename of the current imzml file
        @param binned_files_prefix: A prefix to append to the standard file names of binned files when saving them
        @return A 3d version of the binned datasets such that the z direction contains binned mz intensities for each msi image
        """

        binned_intensity_store = []
        bin_edges_store = []
        bin_to_which_each_mz_belonged_store = []
        self.num_bins = np.ceil((max_mz_all_datasets - min_mz_all_datasets) / self.bin_size)
        num_pixels_in_this_imzml = len(imzml_dataset.mzLengths)
        coordinates = imzml_dataset.coordinates
        for i in range(num_pixels_in_this_imzml):
            this_mz_array = imzml_dataset.getspectrum(i)[0]
            this_intensity_array = imzml_dataset.getspectrum(i)[1]
            binned_intensities, bin_edges, bin_to_which_each_mz_belonged = scipy.stats.binned_statistic(this_mz_array, this_intensity_array, statistic='max', bins=self.num_bins, range=(min_mz_all_datasets, max_mz_all_datasets))
            binned_intensities[np.argwhere(np.isnan(binned_intensities))] = 0
            binned_intensity_store.append(binned_intensities)
            bin_edges_store.append(bin_edges)
            bin_to_which_each_mz_belonged_store.append(bin_to_which_each_mz_belonged)

            if i % 1000 == 0:
                print(num_pixels_in_this_imzml - i, " iterations remaining to completion of binning for dataset ", dataset_number)

        imzml_2d_array = np.array(binned_intensity_store)
        bin_edges = np.array(np.unique(bin_edges_store))

        cols = imzml_dataset.imzmldict["max count of pixels x"]
        rows = imzml_dataset.imzmldict["max count of pixels y"]

        x_3d = np.zeros([rows, cols, imzml_2d_array.shape[-1]])
        for i, (x, y, z) in enumerate(coordinates):
            x_3d[y - 1, x - 1] = imzml_2d_array[i]

        data = {"imzml_array": x_3d, "min_mz": self.min_mz_all_datasets, "max_mz": self.max_mz_all_datasets, "coordinates": coordinates, "bin_edges": bin_edges, "rows": rows, "cols": cols, "bin_size": self.bin_size, "mode": self.mode}

        filename = "/" + os.path.join(*this_imzml_filename.split('/')[0:-1]) + "/" + binned_files_prefix + this_imzml_filename.split('/')[-1].split('.')[0] + "_" + str(self.bin_size).replace(".", "_") + ".pickle"

        with open(filename, "wb") as fh:
            pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
        print("saved: ", filename.split("/")[-1])

        return x_3d

    def normalize_tic(self, truncate_data=1, truncate_quantity=500, add_dummy_data=0):

        ## DEPRACATED
        """

        @brief This method is depracated. TIC Normalization of each dataset separately. Also does truncating of the original data, and adding dummy data
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
        # self.truncate_data = truncate_data
        # self.add_dummy_data = add_dummy_data
        # self.truncate_quantity = truncate_quantity
        # self.combined_tic_normalized_data = []
        # temp = self.combined_data_object
        # dummy_dataset = np.empty((0, temp[0].create2dArray().shape[1]))
        # for count, binned_data_object in enumerate(self.combined_data_object):
        #     binned_dataset = binned_data_object.create2dArray()
        #     self.combined_tic_normalized_data.append(binned_dataset / np.sum(binned_dataset, axis=1, keepdims=True))
        #     dummy_dataset = np.append(dummy_dataset, np.squeeze(rd.choices(binned_dataset, k=4000)), axis=0)
        #
        # ## If necessary, add dummy data, and Do TIC as well.
        # if add_dummy_data == 1:
        #     self.combined_tic_normalized_data.append(dummy_dataset / np.sum(dummy_dataset, axis=1, keepdims=True))
        #
        #     ## Set a certain range of mz values to zero after TIC normalization. This is NOT the same as truncating.
        #     ## This is to make the dummy data have a spectrum that is manipulated according to our needs
        #     self.combined_tic_normalized_data[-1][:, 100:300] = 0
        #
        # if truncate_data == 1:
        #     for count, dataset in enumerate(self.combined_tic_normalized_data):
        #         self.combined_tic_normalized_data[count] = self.combined_tic_normalized_data[count][:, 0:int(truncate_quantity / self.bin_size)]
        #
        # self.min_mz_after_truncation = self.min_mz_of_original_data
        # self.max_mz_after_truncation = self.min_mz_of_original_data + self.truncate_quantity
        #
        # return self.combined_tic_normalized_data

    def create_combined_tic_normalized_data_list_and_combined_data_objects(self, truncate_data=1, truncate_quantity=0, add_dummy_data=0, low_overhead=1):

        self.add_dummy_data = add_dummy_data

        start_time = time.time()
        print("Started TIC normalization and combined_data_object creation")
        output_dict_store = []
        for count, dataset in enumerate(self.dataset_order):
            this_output_dict = self.normalize_tic_individual_dataset(dataset, truncate_data=truncate_data, truncate_quantity=truncate_quantity, low_overhead=low_overhead)
            output_dict_store.append(this_output_dict)

        self.combined_tic_normalized_data = []
        self.combined_data_object = []
        dummy_dataset = np.empty([0, output_dict_store[0]['individual_dummy_data_contribution'].shape[1]])

        for i, dataset in enumerate(self.dataset_order):
            this_output_dict = output_dict_store[i]

            self.combined_tic_normalized_data.append(this_output_dict['individual_tic_normalized_data'])
            np.vstack((dummy_dataset, this_output_dict['individual_dummy_data_contribution']))

            self.combined_data_object.append(this_output_dict['individual_binned_data_object'])


        if self.add_dummy_data == 1:
            self.combined_tic_normalized_data.append(dummy_dataset / np.sum(dummy_dataset, axis=1, keepdims=True))
            self.combined_tic_normalized_data[-1][:, 100:300] = 0  # Set a certain range of mz values to zero after TIC normalization. This is NOT the same as truncating. This is to make the dummy data have a spectrum that is manipulated according to our needs


        self.truncate_data = truncate_data
        self.truncate_quantity = truncate_quantity

        self.max_mz_of_original_data = self.combined_data_object[0].max_mz
        self.min_mz_of_original_data = self.combined_data_object[0].min_mz

        self.min_mz_after_truncation = self.min_mz_of_original_data
        self.max_mz_after_truncation = self.min_mz_of_original_data + self.truncate_quantity

        duration = (time.time() - start_time)/60
        print("Parallel TIC normalization and combined_data_object creation complete. Time taken: ", duration)

    def normalize_tic_individual_dataset(self, binned_dataset_name, truncate_data=1, truncate_quantity=0, low_overhead=1):
        """
        @brief TIC Normalization of a single dataset. Also does truncating of the original data, and making an un-normalized contribution to a dummy dataset if necessary
            Example  Usage:     truncate_original_data = 1
                                truncate_quantity = 500
                                binned_dataset_name = ''
                                [tic_normalized_individual_dataset, dummy_dataset_contribution] = msi_data.normalize_tic_individual_dataset(binned_dataset_name=binned_dataset_name, truncate_data=truncate_original_data, truncate_quantity=truncate_quantity)

        @param binned_dataset_name This is the name of the binned dataset of interest. example; cph1.
        @param truncate_data Set this to 1 to truncate data. If this is set to 0, data will not be truncated
        @param  truncate_quantity Set this to the mz value range that we want to keep. This value is ALWAYS set with a binsize of 1 in mind.
            If the binsize is different, the appropriate number of samples that need to be retained will be automatically calculated.
            The truncate_quantity is actually the number of mz values we want to keep. Truncation begins at the smallest available mz value, and
            goes up in integer values of 1. Therefore, for a bin size of 0.2, and a truncate_quantity of 500, the range of indices
            that will be preserved after truncation is [0  :  (500/0.2)] = [0 : 2500]
        @param low_overhead If this is set to 1, the original binned data object will be erased from memory to save space.
        @return Returns a list containing a TIC normalized binned dataset, and a contribution to a dummy dataset (and if necessary truncated)
        """


        binned_data_object = libmsi_kp.Imzml(self.msi_filename_array[self.dataset_order.index(binned_dataset_name)], bin_size=self.bin_size)
        dataset_number = self.dataset_order.index(binned_dataset_name)

        binned_dataset = binned_data_object.create2dArray()

        tic_normalized_dataset = (binned_dataset / np.sum(binned_dataset, axis=1, keepdims=True))
        dummy_dataset_contribution = np.copy(np.squeeze(rd.choices(binned_dataset, k=4000)))

        if truncate_data == 1:
            tic_normalized_dataset = tic_normalized_dataset[:, 0:int(truncate_quantity / self.bin_size)]
            dummy_dataset_contribution = dummy_dataset_contribution[:, 0:int(truncate_quantity / self.bin_size)]

            tic_normalized_dataset = (tic_normalized_dataset / np.sum(tic_normalized_dataset, axis=1, keepdims=True))  # Re-normalize after truncation
            dummy_dataset_contribution = (dummy_dataset_contribution / np.sum(dummy_dataset_contribution, axis=1, keepdims=True))  # Re-normalize after truncation

        if np.any(np.isnan(tic_normalized_dataset)): # Remove NaNs if they occur
            print("Warning KP: NaNs have occured during TIC normalization. They have been replaced by zeros")
            tic_normalized_dataset[np.argwhere(np.isnan(tic_normalized_dataset))] = 0

        output_dict = {"individual_tic_normalized_data": tic_normalized_dataset,
                       "individual_dummy_data_contribution": dummy_dataset_contribution,
                       "individual_binned_data_object": binned_data_object,
                       "this_dataset_number": dataset_number,
                       "is_dataset_truncated": truncate_data}

        if low_overhead == 1:
            del output_dict['individual_binned_data_object'].imzml_array

        print("Normalization of ", self.msi_filename_array[self.dataset_order.index(binned_dataset_name)], "complete")

        return output_dict


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
        print("saving compact msi data object as pickle file ...")
        self.is_reduced_memory_version = 1

        with open(path_and_filename, "wb") as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

        print("saved compact msi data object as pickle file")

        return self

    def create_double_memory_reduced_footprint_file(self, path_and_filename_to_compact_msi_dataset):

        """
        @brief This function reduces the memory requirement further by removing even the TIC normalized data.

            Example usage:
                            #Need to have first loaded a compact_msi_data object
                            from dataloader_kp import dataloader_kp
                            import pickle

                            path_to_compact_msi_dataset = "/mnt/sda/kasun/compact_msi_data_negative_mode_0_05_bin_size.pickle"
                            compact_msi_data = pickle.load(open(path_to_compact_msi_dataset, 'rb'))  # Load the compact version of msi data

                            double_compact_msi_data = compact_msi_data.create_double_memory_reduced_footprint_file("D:/msi_project_data/binned_binsize_1/compact_data_object/compact_msi_data.npy")

        @param path_and_filename_to_compact_msi_dataset Enter the destination path where the compact data object should be saved.
        @return Returns a double compact version of the msi_data object.
        """

        print("saving double compact msi data object as pickle file ...")

        self.is_double_memory_reduced_version = 1

        for i in range(len(self.dataset_order) - 1, 0, -1):
            del self.combined_tic_normalized_data[i]

        with open(path_and_filename_to_compact_msi_dataset, "wb") as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

        print("saved double compact msi data object as pickle file")

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

        data_subset_dict = {'tic_normalized_combined_data_array_subset': tic_normalized_combined_data_array_subset, 'subset_pattern_array': subset_pattern_array}

        return data_subset_dict

