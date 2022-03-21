import numpy as np
import random as rd
from sklearn.decomposition import FastICA, PCA, NMF
from sklearnex import patch_sklearn
import matplotlib.pyplot as plt
import pickle
patch_sklearn()
rd.seed(0)

class dim_reduction_common_kp:
    """
    @brief This class contains functions that are common to both nmf and pca.

    """
    def __init__(self, dim_reduced_object):

        """
        @brief Initializes the class. Different functions can then be performed on this class easily.
            Example usage:
                            dim_reduced_object = nmf_kp_object
                            my_arguments = ''
                            my_output_1 = dim_reduction_common_kp(nmf_kp_object).my_method_1(my_arguments)

        @param dim_reduced_object: Insert the dimensionality reduced object here. Example: nmf_kp_object, pca_kp_object, etc
        """

        print("Using a common method used for both nmf and pca")
        self.dim_reduced_object = dim_reduced_object

        self.used_dim_reduction_technique = self.dim_reduced_object.used_dim_reduction_technique
        self.recovered_dim_reduced_dict = self.dim_reduced_object.dim_reduced_dict
        self.dim_reduced_data = self.recovered_dim_reduced_dict['dim_reduced_outputs']
        self.saved_dim_reduced_filename = self.dim_reduced_object.saved_dim_reduced_filename


    def downsampler(self, step=12, save_data=0, filename_prefix=''):

        """
        @brief Downsample NMF or PCA data so that it becomes easier to do prototyping work

            Example Usage: For NMF for example:

                            save_data = 0
                            downsampler_step = 12
                            downsampler_prefix = 'downsampled_'
                            dim_reduced_object_kp = nmf_kp_object
                            downsampled_nmf_dict = dim_reduction_common_kp(dim_reduced_object_kp).downsampler(step=downsampler_step, save_data=save_data, filename_prefix=downsampler_prefix)

        @param step This gives how often an NMF or PCA output sample should be selected from the list of
            outputs. (That is, we are selecting only NMF/PCA output of every 12th pixel)
        @param save_data Save data if set to 1.
        @param filename_prefix Only required if save_data = 1.

        """

        w_data = self.dim_reduced_data[0][0]
        h_data = self.dim_reduced_data[1][0]
        downsampling_indices_w_data = range(0, w_data.shape[0], step)
        downsampling_indices_h_data = range(0, h_data.shape[1], step)
        downsampled_w_data = w_data[downsampling_indices_w_data, :]
        downsampled_h_data = h_data[:, downsampling_indices_h_data]

        downsampled_dim_reduced_data = [[downsampled_w_data], [downsampled_h_data]]
        downsampled_pixel_count_array = (self.recovered_dim_reduced_dict['pixel_count_array'] / 12).astype(int)

        downsampled_dim_reduced_dict = self.recovered_dim_reduced_dict
        downsampled_dim_reduced_dict[self.used_dim_reduction_technique+'_outputs'] = downsampled_dim_reduced_data
        downsampled_dim_reduced_dict['pixel_count_array'] = downsampled_pixel_count_array
        downsampled_dim_reduced_dict['global_path_name'] = self.dim_reduced_object.dataloader_kp_object.global_path_name

        if save_data == 1:
            folder_name = self.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/'+self.used_dim_reduction_technique+'_outputs/downsampled/'
            downsampled_file_name = filename_prefix + self.saved_dim_reduced_filename.split('/')[-1]
            np.save(folder_name + downsampled_file_name, downsampled_dim_reduced_dict, 'dtype=object')

        return downsampled_dim_reduced_dict

