import pickle
import random as rd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearnex import patch_sklearn
import copy

from dim_reduction_common_kp import dim_reduction_common_kp
import time
import os
patch_sklearn()
rd.seed(0)


class pca_kp:

    def __init__(self, dataloader_kp_object, saved_pca_filename=None, custom_pca_dict_from_memory=None):

        """

        @brief Initializes the pca_kp object. This class is designed to perform dimensionality reduction tasks like
            NMF, PCA, ICA etc. It will also include the ability to visualize various plots for these results.

            Example usage:  from pca_kp import pca_kp
                            saved_pca_filename = "D:/msi_project_data/saved_outputs/pca_outputs/truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_pca_dict_max_iter_10000_tic_normalized_pca_outputs_20.npy"
                            pca_kp_object = pca_kp(msi_data, saved_pca_filename=saved_pca_filename)

                            OR:

                            pca_kp_object = pca_kp(msi_data)

                            OR:

                            custom_pca_dict_from_memory = my_pca_dict
                            pca_kp_object = pca_kp(msi_data, custom_pca_dict_from_memory=custom_pca_dict_from_memory)

        @param dataloader_kp_object This is an output of the dataloader_kp class. This contains all the datasets
            that will be manipulated to perform pca.
        @param custom_pca_dict_from_memory: If this is given (i.e, an pca dictionary already stored in memory is included here), initialize the pca_kp class object using this dataset.
        """
        self.dataloader_kp_object = dataloader_kp_object
        self.combined_tic_normalized_data = dataloader_kp_object.combined_tic_normalized_data
        self.dataset_order = self.dataloader_kp_object.dataset_order
        self.bin_size = self.dataloader_kp_object.bin_size
        self.used_dim_reduction_technique = 'pca'

        if (((saved_pca_filename is None) or (saved_pca_filename == '')) and ((custom_pca_dict_from_memory == None) or (custom_pca_dict_from_memory == ''))):
            self.saved_dim_reduced_filename = ''
            print("Already calculated pca dictionary has NOT been loaded. Please invoke the 'data_subset_creator() function before calling calc_pca()")
            pass
        elif (saved_pca_filename is not None) and (saved_pca_filename is not ''):
            print("Already calculated pca dictionary has been loaded from file")
            self.dim_reduced_dict = np.load(saved_pca_filename, allow_pickle=True)[()]
            self.saved_dim_reduced_filename = saved_pca_filename
            self.num_dim_reduced_components = self.dim_reduced_dict['num_dim_reduced_components']
        elif (custom_pca_dict_from_memory is not None) and (custom_pca_dict_from_memory is not ''):
            print("Already calculated CUSTOM pca dictionary has been loaded from MEMORY")
            self.dim_reduced_dict = custom_pca_dict_from_memory
            self.saved_dim_reduced_filename = None
            self.num_dim_reduced_components = self.dim_reduced_dict['num_dim_reduced_components']



    def data_subset_creator(self, subset_pattern_array=[1, 1, 1, 1, 1, 1, 1, 1]):
        """
        @brief This is simply a link to the actual data_subset_creator() method which is a part of the dataloader_kp class.

            Example usage:  subset_pattern_array = [1, 0, 0, 0, 0, 0, 0, 0]
                            data_subset = pca_kp_object.data_subset_creator(subset_pattern_array=subset_pattern_array)


        @param subset_pattern_array: The datasets that we want included in the analysis are chosen here.
        @return: the subset of data that was chosen.
        """
        self.data_subset_dict = self.dataloader_kp_object.data_subset_creator(subset_pattern_array=subset_pattern_array)

        return self.data_subset_dict

    def calc_pca(self, num_components=20, pca_whiten=False, num_iter='auto', individual=0, save_data=0, filename_prefix=''):

        """
        @brief Perform pca on the given subset of data.

            Example usage:      pca_num_components = 20
                                pca_whiten = False
                                pca_individual = 0
                                pca_whiten = False
                                pca_filename_prefix = ''
                                num_iter = 'auto'  # Can take either 'auto' or an integer value
                                pca_dict = pca_kp_object.calc_pca(num_components=pca_num_components, pca_whiten=pca_whiten, num_iter=num_iter, individual=pca_individual, save_data=save_data,filename_prefix=pca_filename_prefix)



        @param num_components The number of pca components I want to have.
        @param pca_whiten Refer to sklearn documentation
        @param individual If this is set to 0, pca will be calculated on the entire sets of data put into a single large dataset.
            If this is set to 1, pca will be performed individually on each dataset in the data_subset
        @param save_data  If this is set to 1, data will be saved  with a filename with the prefix as given by the 'filename_prefix' parameter
        @param filename_prefix This is only used if the 'save_data' parameter is set to 1. The value entered here will be added as a prefix to
            the file that will be saved
        @return: The pca output dictionary
        """

        start_time = time.time()
        print("Starting PCA calculation")
        tic_normalized_combined_data_array_subset = self.data_subset_dict['tic_normalized_combined_data_array_subset']
        subset_pattern_array = self.data_subset_dict['subset_pattern_array']
        self.num_dim_reduced_components = num_components
        pca_model = PCA(n_components=num_components, whiten=pca_whiten, random_state=0)

        if individual == 0:
            num_datasets = len(tic_normalized_combined_data_array_subset)
            pixel_count_array = np.empty((0, 1))

            combined_dataset_for_pca = np.empty((0, tic_normalized_combined_data_array_subset[0].shape[1]))

            for count, dataset in enumerate(tic_normalized_combined_data_array_subset):
                combined_dataset_for_pca = np.append(combined_dataset_for_pca, dataset, axis=0)
                pixel_count_array = np.append(pixel_count_array, dataset.shape[0])

            w = pca_model.fit_transform(combined_dataset_for_pca)  # W matrix will hold the transformed values (n_samples x n_components)
            h = pca_model.components_  # H matrix will hold the PCA component eigen vector directions (n_components x n_features)
            pca_outputs = [[w], [h]]
            explained_variance = pca_model.explained_variance_ratio_
            total_explained_variance = np.sum(explained_variance)


        else:
            pca_outputs = []
            explained_variance = []
            total_explained_variance = []
            num_datasets = len(tic_normalized_combined_data_array_subset)
            pixel_count_array = np.empty((0, 1))

            for count, dataset in enumerate(tic_normalized_combined_data_array_subset):
                w = pca_model.fit_transform(dataset)
                h = pca_model.components_

                individual_pca_outputs = [[w], [h]]
                individual_dataset_explained_variance = pca_model.explained_variance_ratio_
                individual_dataset_total_explained_variance = np.sum(individual_dataset_explained_variance)

                pixel_count_array = np.append(pixel_count_array, dataset.shape[0])
                pca_outputs.append(individual_pca_outputs)
                explained_variance.append(individual_dataset_explained_variance)
                total_explained_variance.append(individual_dataset_total_explained_variance)

        duration = (time.time() - start_time)/60
        print("This PCA run completed in ", duration, " minutes")
        self.dim_reduced_dict = {'dim_reduced_outputs': pca_outputs,
                                 'explained_variance': explained_variance,
                                 'total_explained_variance':  total_explained_variance,
                                 'pixel_count_array': pixel_count_array,
                                 'subset_pattern_array': subset_pattern_array,
                                 'individual': individual,
                                 'dim_reduced_object': pca_model,
                                 'global_path_name': self.dataloader_kp_object.global_path_name,
                                 'num_dim_reduced_components': num_components,
                                 'used_dim_reduction_technique': self.used_dim_reduction_technique,
                                 'num_iter': num_iter,
                                 'duration_for_completion_in_minutes': duration}

        ### Saving the pca outputs as an npy file.
        if save_data == 1:
            if individual == 1:
                np.save(self.dataloader_kp_object.global_path_name + 'saved_outputs/pca_outputs/' + filename_prefix + 'pca_on_binsize_' + str(self.dataloader_kp_object.bin_size).replace('.', '_') + '_' + self.dataloader_kp_object.mode + '_mode_individual_datasubset_' + str(subset_pattern_array).replace(' ', '_').replace(',', '') + '_max_iter_' + str(num_iter) + '_num_components_' + str(num_components) + '.npy', self.dim_reduced_dict, 'dtype=object')
            else:
                np.save(self.dataloader_kp_object.global_path_name + 'saved_outputs/pca_outputs/' + filename_prefix + 'pca_on_binsize_' + str(self.dataloader_kp_object.bin_size).replace('.', '_') + '_' + self.dataloader_kp_object.mode + '_mode_combined_datasubset_' + str(subset_pattern_array).replace(' ', '_').replace(',', '') + '_max_iter_' + str(num_iter) + '_num_components_' + str(num_components) + '.npy', self.dim_reduced_dict, 'dtype=object')

        return self.dim_reduced_dict

    def pca_downsampler(self, step=12, save_data=0, filename_prefix=''):

        """
        @brief Downsample pca data so that it becomes easier to do prototyping work. This is only a link to the actual implementation of this function
            The actual implementation is in the dim_reduction_common_kp' class

            Example Usage:  save_data = 0
                            downsampler_step = 12
                            downsampler_prefix = 'downsampled_'
                            downsampled_pca_dict = pca_kp_object.pca_downsampler(step=downsampler_step, save_data=save_data, filename_prefix=downsampler_prefix)


        @param step This gives how often an pca output sample should be selected from the list of
            outputs. (That is, we are selecting only pca output of every 12th pixel)
        @param save_data Save data if set to 1.
        @param filename_prefix Only required if save_data = 1.

        """
        self.downsampled_pca_dict = dim_reduction_common_kp(self).downsampler(step=step, save_data=save_data, filename_prefix=filename_prefix)
        return self.downsampled_pca_dict

    def order_pca_according_to_reconstruction_error(self, save_data=0, filename_prefix=''):

        """
        THIS METHOD IS DEPRACATED. THIS HAS BEEN REPLACED BY: self.order_pca_according_to_RESIDUAL_reconstruction_error()
        @brief Order the internal representation of PCA components according to the reconstruction error.
                Example usage:

                                save_data = 0
                                filename_prefix = ''
                                ordered_pca_dict = pca_kp_object.order_pca_according_to_reconstruction_error(save_data=save_data, filename_prefix=filename_prefix)

        @param save_data Set to 1 to save the data.
        @param filename_prefix Use this prefix in the filename used to save this result.
        @return pca_dict with pca components ordered correctly according to their reconstruction error.
        """


        # if not (hasattr(self, 'data_subset_dict')):
        #     subset_pattern_array = self.dim_reduced_dict['subset_pattern_array']
        #     data_subset_dict = self.data_subset_creator(subset_pattern_array=subset_pattern_array)
        # tic_normalized_combined_data_array_subset = self.data_subset_dict['tic_normalized_combined_data_array_subset']
        # num_datasets = len(tic_normalized_combined_data_array_subset)
        # print("Combining datasets...")
        # combined_dataset_for_pca = np.empty((0, tic_normalized_combined_data_array_subset[0].shape[1]))
        # for count, dataset in enumerate(tic_normalized_combined_data_array_subset):
        #     combined_dataset_for_pca = np.append(combined_dataset_for_pca, dataset, axis=0)
        #
        # print("Combining datasets done")
        #
        # dim_reduced_outputs = self.dim_reduced_dict['dim_reduced_outputs']
        # subset_pattern_array = self.dim_reduced_dict['subset_pattern_array']
        # num_components = self.dim_reduced_dict['num_dim_reduced_components']
        # w_array = dim_reduced_outputs[0][0]
        # h_array = dim_reduced_outputs[1][0]
        #
        # print("Calculating reconstruction errors...")
        # reconstruction_error_store = np.empty([0, 2])
        # for component_number in range(num_components):
        #     print("Now checking component: ", component_number)
        #     reconstruction_difference = combined_dataset_for_pca - np.matmul(w_array[:, [component_number]], h_array[[component_number], :])
        #     reconstruction_error = np.linalg.norm(reconstruction_difference, ord='fro')
        #     reconstruction_error_store = np.vstack((reconstruction_error_store, [component_number, reconstruction_error]))
        #
        # ordered_reconstruction_error = reconstruction_error_store[:, 1].argsort()
        # ordered_reconstruction_error_store = reconstruction_error_store[ordered_reconstruction_error, :]
        #
        # ordered_w_array = w_array[:, np.int16(ordered_reconstruction_error_store[:, 0])]
        # ordered_h_array = h_array[np.int16(ordered_reconstruction_error_store[:, 0]), :]
        #
        # ordered_dim_reduced_outputs = [[ordered_w_array], [ordered_h_array]]
        #
        # self.dim_reduced_dict['dim_reduced_outputs'] = ordered_dim_reduced_outputs
        # self.dim_reduced_dict['ordered_reconstruction_error_store'] = ordered_reconstruction_error_store
        #
        # print("Internal representation of the PCA order got changed")
        #
        # if save_data == 1:
        #     np.save(self.dataloader_kp_object.global_path_name + '/saved_outputs/pca_outputs/' + filename_prefix + 'combined_dataset_' + str(subset_pattern_array).replace(' ', '_').replace(',', '') + '_tic_normalized_pca_outputs_' + str(num_components) + '.npy', self.dim_reduced_dict, 'dtype=object')
        #
        # return self.dim_reduced_dict

    def calculate_reconstruction_error_for_given_set_of_components(self, original_combined_data_for_pca, modified_dim_reduced_outputs):

        """
        @brief Calculate the reconstruction error for a given set of components.
        @param modified_dim_reduced_outputs: A dimensionality reduced array with a subset of the components, or all the components.
        @param original_combined_data_for_pca: This is the ground truth dataset
        @return reconstruction error: The reconstruction error with the selected set of components
        """

        w_array = modified_dim_reduced_outputs[0][0]
        h_array = modified_dim_reduced_outputs[1][0]

        reconstruction_difference = original_combined_data_for_pca - np.matmul(w_array, h_array)
        reconstruction_error = np.linalg.norm(reconstruction_difference, ord='fro')

        return reconstruction_error

    def explained_variance_vs_num_pca_components_calculator(self, max_num_components=20, pca_whiten=False, pca_individual=0, subset_pattern_array=[1, 1, 1, 1, 1, 1, 1, 1], plot_result=0, save_plot=0, plot_file_format='svg', plot_dpi=600, save_data=0, data_filename_prefix=''):

        """
        @brief Calculate pca for explained variance vs number of pca components used.

            Usage example:
                            *Note: pca_kp object must be initialized before hand.

                            max_num_components = 20
                            pca_whiten = False
                            pca_individual = 0
                            subset_pattern_array = [1, 1, 1, 1, 1, 1, 1, 1]
                            plot_result = 1
                            save_plot = 1
                            plot_file_format = 'svg'
                            plot_dpi = 600
                            save_data = 1
                            data_filename_prefix = 'test_num_comp_vs_accuracy_'

                            pca_kp_object.explained_variance_vs_num_pca_components_calculator(max_num_components=max_num_components, pca_whiten=pca_whiten, pca_individual=pca_individual, subset_pattern_array=subset_pattern_array, plot_result=plot_result, save_plot=save_plot, plot_file_format=plot_file_format, plot_dpi=plot_dpi, save_data=save_data, data_filename_prefix=data_filename_prefix)


        @param max_num_components: Maximum number of pca components that will be tried
        @param pca_whiten: Please see sklearn documentation
        @param pca_individual: If set to 1, perform pca on each individual dataset (depracated). If set to 0, performs
            pca on the entire dataset (as selected by the subset_pattern_array)   taken as a whole.
        @param subset_pattern_array:  Determines the subset of datasets to be used in the pca calculation.
        @param plot_result: Set to 1 if the result needs to be plotted.
        @param save_plot: Set to 1 if the plot needs to be saved
        @param plot_file_format: Set to 'svg', 'png', 'pdf' etc.
        @param plot_dpi: Set the dpi when saving the plot
        @param save_data: Set to 1 if all pca trials need to be saved. Warning: This will generate a lot of data; one result file for each pca trial.
            Number of saved files is equal to the max_num_components variable.
        @param data_filename_prefix: Filename prefix for saving the data for each pca trial if save_data is enabled
        @return: total_explained_variance_array that contains the pca component count used and explained variance (2 columns) for
            that pca component count in each row of the array.

        """

        data_subset_dict = self.data_subset_creator(subset_pattern_array)  # This is necessary because the calc_pca() function expects a datasubsetarray dictionary
        total_explained_variance_array = np.empty((0, 2))
        for num_comps in range(1, (max_num_components+1)):
            pca_num_components = num_comps
            pca_dict = self.calc_pca(num_components=pca_num_components, pca_whiten=pca_whiten, individual=pca_individual, save_data=save_data, filename_prefix=data_filename_prefix)
            total_explained_variance_array = np.vstack((total_explained_variance_array, np.array([[num_comps, pca_dict['total_explained_variance']]])))
            print("Current trial:  num_components: " + str(num_comps) + ", total_explained_variance: " + str(pca_dict['total_explained_variance']))

        if plot_result == 1:
            fig, ax = plt.subplots(1,1)
            ax.plot(total_explained_variance_array[:, 0], total_explained_variance_array[:, 1])
            ax.set_xlabel('Number of pca components', fontsize=20)
            ax.set_ylabel('Total variance explained', fontsize=20)
            ax.set_title('Total explained variance vs number of pca components', fontsize=30)
            ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)

            plot_title = 'Variance explained vs number of pca  components used'
            plt.suptitle(plot_title, x=6, y=1.5, fontsize=80)
            # plt.subplots_adjust(bottom=0, left=0, right=12, top=1)
            plt.show()

        if save_plot == 1:
            folder_name = self.dataloader_kp_object.global_path_name + '/saved_outputs/figures/'
            fig.savefig(folder_name + plot_title.replace(' ', '_') + '.' + plot_file_format, dpi=plot_dpi, bbox_inches='tight')
        print('\n')

        return total_explained_variance_array

    def convert_aman_pca_output_to_match_kp_pca_class(self, aman_pca_filename='', save_data=0, filename_prefix=''):
        """
        @brief Convert an existing pca object generated via libmsi into a pca_dict_kp form compatible with my
            pca_kp class.

            Example usage:
                            * Note: pca_kp class must be initialized.
                            aman_pca_filename = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/from_aman/pca_20_all_0_05.pickle"
                            save_data = 1
                            filename_prefix = 'aman_0.05_'
                            pca_dict = pca_kp_object.convert_aman_pca_output_to_match_kp_pca_class(aman_pca_filename, save_data=save_data, filename_prefix=filename_prefix)
        @param save_data Will save the converted pca dict if set to 1.
        @param filename_prefix Required only if save_data is set to 1.
        @param aman_pca_filename: Path to the file containing the pca results for combined datasets, generated via libmsi
        @return pca_dict
        """

        converted_dim_reduced_dict = {}
        converted_dim_reduced_dict['subset_pattern_array'] = [1, 1, 1, 1, 1, 1, 1, 1]
        converted_dim_reduced_dict['individual'] = 'individual'
        converted_dim_reduced_dict['used_dim_reduction_technique'] = 'pca'

        aman_pca_dict = pickle.load(open(aman_pca_filename, 'rb'))

        w_aman = aman_pca_dict["pca_data_list"]
        h_aman = aman_pca_dict["pca_components_spectra"]

        w_aman_rearranged = np.empty([0, h_aman.shape[0]])
        pixel_count_array = []
        for j in w_aman:
            w_aman_rearranged = np.append(w_aman_rearranged, j, axis=0)
            pixel_count_array.append(j.shape[0])

        converted_dim_reduced_dict['pixel_count_array'] = pixel_count_array
        converted_dim_reduced_dict['dim_reduced_outputs'] = [[w_aman_rearranged], [h_aman.T]]
        converted_dim_reduced_dict['num_dim_reduced_components'] = w_aman_rearranged.shape[1]
        converted_dim_reduced_dict['dim_reduced_object'] = ''  # To Do: Add the nmf/PCA object from aman's nmf/pca result
        converted_dim_reduced_dict['explained_variance'] = ''  # To Do: Add the reconstruction error from aman's nmf/pca result


        pca_dict = converted_dim_reduced_dict

        if save_data == 1:
            folder_name = self.dataloader_kp_object.global_path_name + '/saved_outputs/pca_outputs/'
            np.save(folder_name + filename_prefix + '_' + aman_pca_filename.split('.')[0].split('/')[-1] + '.npy', pca_dict, 'dtype=object')

        self.dim_reduced_dict = pca_dict
        print('Updated the internal pca_dict attribute in the pca_kp_object (self.dim_reduced_dict) \n')

        return self.dim_reduced_dict

    def modify_existing_pca_dict(self, old_saved_pca_filename, save_data=1, modified_pca_filename_prefix=''):
        """
        @brief This function is designed to load an existing saved pca dictionary, and modify it to suit newer versions of the pca_kp class
            Example use:
                        old_saved_pca_filename  = "D:/msi_project_data/saved_outputs/pca_outputs/no_whitening_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_20_components_pca_dict_tic_normalized.npy"
                        save_data =  1
                        modified_pca_filename_prefix = 'modified_'
                        modified_pca_dict = pca_kp_object.modify_existing_pca_dict(old_saved_pca_filename, save_data=save_data, modified_pca_filename_prefix=modified_pca_filename_prefix)

        @param old_saved_pca_filename: This is an essential argument, This points to the existing pca file that needs to be modified
        @param save_data:  Save the modified pca file.
        @param modified_pca_filename_prefix: Enter a prefix that I want the modified pca dictionary to have if it gets saved (i.e, if save_data ==1)
            This prefix will be prepended to the name of the old_saved_pca_filename's filename part.
        @return  modified_pca_dict:  Returns the modified pca dictionary
        """
        print("This code must be adjusted everytime you use it to include the modifications you intend to do")

        old_saved_pca_dict = np.load(old_saved_pca_filename, allow_pickle=True)[()]
        modified_pca_dict = copy.deepcopy(old_saved_pca_dict)
        #############################################
        ###  Do the modifications:
        ### Ex:
        ## modified_pca_dict['explained_variance'] = old_saved_pca_dict['variance_explained']
        ## modified_pca_dict['total_explained_variance'] = np.sum(old_saved_pca_dict['variance_explained'])
        # del modified_pca_dict['pca_outputs']
        # modified_pca_dict['dim_reduced_outputs'] = old_saved_pca_dict['pca_outputs']
        #
        # del modified_pca_dict['pca_object']
        # modified_pca_dict['dim_reduced_object'] = old_saved_pca_dict['pca_object']
        #
        # modified_pca_dict['used_dim_reduction_technique'] = 'pca'
        # modified_pca_dict['num_dim_reduced_components'] = 20

        self.dim_reduced_dict = modified_pca_dict
        print("Internal representation of the pca_dict variable has been updated (Added total explained variance, etc)")

        if save_data == 1:
            old_filename = old_saved_pca_filename.split('/')[-1]
            folder_name = self.dataloader_kp_object.global_path_name + '/saved_outputs/pca_outputs/'
            np.save(folder_name + modified_pca_filename_prefix + old_filename, modified_pca_dict, 'dtype=object')

        return modified_pca_dict

    def order_pca_according_to_RESIDUAL_reconstruction_error(self, save_data=1, filename_prefix='', saved_order_statistics_filename=''):

        """
        @brief Order pca components according to the residual error of reconstruction after removing the next strongest component with each iteration

                Usage Example: save_data = 1
                                filename_prefix = ''
                                pca_kp_object.order_pca_according_to_RESIDUAL_reconstruction_error(save_data=save_data, filename_prefix=filename_prefix)

                                ## OR: Load an already ordered statistics file and use it to do the actual ordering of the pca dict
                                # saved_order_statistics_filename = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/pca_outputs/statistics_of_ordering_according_to_residual_reconstruction_error_redo_batch_mode_test_negative_mode_1_binsize_partial_complete_current_batch_11_individual_dataset_nmf_dict_max_iter_2000_tic_normalized_pca_outputs_20.npy"
                                # pca_kp_object.order_pca_according_to_RESIDUAL_reconstruction_error(save_data=save_data, filename_prefix=filename_prefix, saved_order_statistics_filename=saved_order_statistics_filename)


        @param save_data: Whether to save data
        @param filename_prefix: Prefix to use when saving the file
        @param saved_order_statistics_filename: If this is available, the finding of the optimal order can be skipped. However, the pca must still be aligned using this order
        @return ordered_dim_reduced_dict: Returns an ordered nmf dict
        """

        if not(saved_order_statistics_filename):
            if not (hasattr(self, 'data_subset_dict')):
                subset_pattern_array = self.dim_reduced_dict['subset_pattern_array']
                data_subset_dict = self.data_subset_creator(subset_pattern_array=subset_pattern_array)

            tic_normalized_combined_data_array_subset = self.data_subset_dict['tic_normalized_combined_data_array_subset']
            num_datasets = len(tic_normalized_combined_data_array_subset)
            print("Combining datasets...")
            combined_dataset_for_pca = np.empty((0, tic_normalized_combined_data_array_subset[0].shape[1]))
            for count, dataset in enumerate(tic_normalized_combined_data_array_subset):
                combined_dataset_for_pca = np.append(combined_dataset_for_pca, dataset, axis=0)

            print("Combining datasets done")

            dim_reduced_dict_recovered = self.dim_reduced_dict
            dim_reduced_outputs_recovered = dim_reduced_dict_recovered['dim_reduced_outputs']
            num_dim_reduced_components = self.num_dim_reduced_components
            main_loop_dim_reduced_outputs_recovered = copy.deepcopy(dim_reduced_outputs_recovered)

            main_loop_comparison_table = {'count_array': [], 'component_removed': [], 'residual_error': [], 'sub_loop_comparison_table': []}

            current_component_order = np.arange(0, num_dim_reduced_components)

            recon_error_with_all_components = self.calculate_reconstruction_error_for_given_set_of_components(combined_dataset_for_pca, main_loop_dim_reduced_outputs_recovered)
            print("Reconstruction error with NO component removed ", recon_error_with_all_components)
            main_loop_comparison_table['count_array'].append(0)
            main_loop_comparison_table['component_removed'].append("None")
            main_loop_comparison_table['residual_error'].append(np.round(recon_error_with_all_components, 3))

            for count in range(num_dim_reduced_components):

                sub_loop_comparison_table = {'component_removed': [], 'residual_error': []}

                remaining_num_components = main_loop_dim_reduced_outputs_recovered[1][0].shape[0]

                for sub_count in range(remaining_num_components):
                    sub_component_removed = current_component_order[sub_count]
                    new_w_array = np.delete(main_loop_dim_reduced_outputs_recovered[0][0], sub_count, axis=1)  # Remove component from w array
                    new_h_array = np.delete(main_loop_dim_reduced_outputs_recovered[1][0], sub_count, axis=0)  # Remove component from h array
                    sub_loop_dim_reduced_outputs_recovered = [[new_w_array], [new_h_array]]
                    this_recon_error = self.calculate_reconstruction_error_for_given_set_of_components(combined_dataset_for_pca, sub_loop_dim_reduced_outputs_recovered)
                    print("Count: ", count, ". Residual reconstruction error with component  ", sub_component_removed, " removed: ", this_recon_error)
                    sub_loop_comparison_table['component_removed'].append(sub_component_removed)
                    sub_loop_comparison_table['residual_error'].append(np.round(this_recon_error, 3))

                best_component_index = np.argmax(sub_loop_comparison_table['residual_error'])
                best_component_removed = current_component_order[best_component_index]
                residual_error_after_best_component_removal = sub_loop_comparison_table['residual_error'][best_component_index]
                main_loop_comparison_table['count_array'].append(count)
                main_loop_comparison_table['component_removed'].append(best_component_removed)
                main_loop_comparison_table['residual_error'].append(np.round(residual_error_after_best_component_removal, 3))
                main_loop_comparison_table['sub_loop_comparison_table'].append(sub_loop_comparison_table)
                current_component_order = np.delete(current_component_order, best_component_index)

                new_w_array = np.delete(main_loop_dim_reduced_outputs_recovered[0][0], best_component_index, axis=1)  # Remove component from w array
                new_h_array = np.delete(main_loop_dim_reduced_outputs_recovered[1][0], best_component_index, axis=0)  # Remove component from h array
                main_loop_dim_reduced_outputs_recovered = [[new_w_array], [new_h_array]]

                print(count)
                print("Main loop comparison table", main_loop_comparison_table)
                print('___________\n')


            if save_data == 1:
                split_name = self.saved_dim_reduced_filename.split('/')
                file_path = split_name[: -1]
                file_name = split_name[-1]
                pathname_to_save = '/' + os.path.join(*file_path) + '/' + filename_prefix + 'statistics_of_ordering_according_to_residual_reconstruction_error_' + file_name
                np.save(pathname_to_save, main_loop_comparison_table, 'dtype=object')

        else:
            main_loop_comparison_table = np.load(saved_order_statistics_filename, allow_pickle=True)[()]

        optimal_component_order = main_loop_comparison_table['component_removed'][1:]
        ordered_dim_reduced_dict = self.dim_reduced_dict
        w_array_rearranged = ordered_dim_reduced_dict['dim_reduced_outputs'][0][0][:, optimal_component_order]
        h_array_rearranged = ordered_dim_reduced_dict['dim_reduced_outputs'][1][0][optimal_component_order, :]

        rearranged_dim_reduced_outputs = [[w_array_rearranged], [h_array_rearranged]]
        ordered_dim_reduced_dict['dim_reduced_outputs'] = rearranged_dim_reduced_outputs
        ordered_dim_reduced_dict['ordered_reconstruction_error_store'] = main_loop_comparison_table['residual_error'][1:]


        self.dim_reduced_dict = ordered_dim_reduced_dict
        print("Internal representation of the dim_reduced_object got changed")

        if save_data == 1:
            split_name = self.saved_dim_reduced_filename.split('/')
            file_path = split_name[: -1]
            file_name = split_name[-1]
            pathname_to_save = '/' + os.path.join(*file_path) + '/' + filename_prefix + 'ordered_according_to_residual_reconstruction_error_' + file_name
            np.save(pathname_to_save, ordered_dim_reduced_dict, 'dtype=object')

        return ordered_dim_reduced_dict