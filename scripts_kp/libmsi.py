"""
@package libmsi
This module is designed to read and process msi data.

Capabilities:

1. Preprocessing: Binning and Normalization
2. Dimensionality Reduction (PCA/ICA/NMF)
3. Clustering (KMeans)
"""
from pyimzml.ImzMLParser import ImzMLParser as read_msi
import numpy as np
import pickle


class Imzml:
    """
    @brief Imzml class is designed to load, process, and visualize MS spectra and Image data
    """

    def __init__(self, filename="", bin_size=1):
        """
        @brief The constructor for Imzml class
        @param filename input file to be read
        @param bin_size bin size to be used for binning
        """
        self.filename = filename
        self.binSize = bin_size

        data = self.load_binned_data()

        self.coordinates = data["coordinates"]
        self.min_mz = data["min_mz"]
        self.max_mz = data["max_mz"]
        self.imzml_array = data["imzml_array"]  # [:,:,:10000]
        self.rows, self.cols, _ = np.shape(self.imzml_array)

        print("loaded ", self.filename)


    def load_binned_data(self):
        """
        @brief Method to read data in .pickle format
        """
        with open(self.filename, "rb") as fh:
            data = pickle.load(fh)
        return data


    def create2dArray(self, store=None):
        """
        @brief Method to get a list of valid pixel spectra in 2d array format
        """
        imzml_2d_array = self.imzml_array[~np.all(self.imzml_array == 0, axis=-1)]

        return imzml_2d_array

    def reconstruct_3d_array(self, x_2d):
        """
        @brief Method to reconstruct the 3d image array from 2d valid pixel spectra
        @param x_2d 2D array with valid pixel spectra
        """
        x_3d = np.zeros([self.rows, self.cols, x_2d.shape[-1]])
        for i, (x, y, z) in enumerate(self.coordinates):
            x_3d[y - 1, x - 1] = x_2d[i]
        return x_3d

