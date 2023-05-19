import os
import pickle
import random as rd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearnex import patch_sklearn
import copy

import multiprocessing
from dim_reduction_common_kp import dim_reduction_common_kp
import time
patch_sklearn()
rd.seed(0)


class nmf_kp:

    def __init__(self, dataloader_kp_object, saved_nmf_filename=None, custom_nmf_dict_from_memory=None):

        """

        @brief Initializes the nmf_kp object. This class is designed to perform dimensionality reduction tasks like
            NMF, PCA, ICA etc. It will also include the ability to visualize various plots for these results.

            Example usage:  from nmf_kp import nmf_kp
                            saved_nmf_filename = "D:/msi_project_data/saved_outputs/nmf_outputs/truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
                            nmf_kp_object = nmf_kp(msi_data, saved_nmf_filename=saved_nmf_filename)

                            OR:

                            nmf_kp_object = nmf_kp(msi_data)

                            OR:

                            custom_nmf_dict_from_memory = my_nmf_dict
                            nmf_kp_object = nmf_kp(msi_data, custom_nmf_dict_from_memory=custom_nmf_dict_from_memory)

        @param dataloader_kp_object This is an output of the dataloader_kp class. This contains all the datasets
            that will be manipulated to perform NMF.
        @param custom_nmf_dict_from_memory: If this is given (i.e, an nmf dictionary already stored in memory is included here), initialize the nmf_kp class object using this dataset.
        """
        self.dataloader_kp_object = dataloader_kp_object
        self.combined_tic_normalized_data = dataloader_kp_object.combined_tic_normalized_data
        self.dataset_order = self.dataloader_kp_object.dataset_order
        self.bin_size = self.dataloader_kp_object.bin_size
        self.used_dim_reduction_technique = 'nmf'

        if (((saved_nmf_filename is None) or (saved_nmf_filename == '')) and ((custom_nmf_dict_from_memory == None) or (custom_nmf_dict_from_memory == ''))):
            self.saved_dim_reduced_filename = ''
            print("Already calculated nmf dictionary has NOT been loaded. Please invoke the 'data_subset_creator() function before calling calc_nmf()")
            pass
        elif (saved_nmf_filename is not None) and (saved_nmf_filename is not ''):
            print("Already calculated nmf dictionary has been loaded from file")
            self.dim_reduced_dict = np.load(saved_nmf_filename, allow_pickle=True)[()]
            self.saved_dim_reduced_filename = saved_nmf_filename
            self.num_dim_reduced_components = self.dim_reduced_dict['num_dim_reduced_components']

        elif (custom_nmf_dict_from_memory is not None) and (custom_nmf_dict_from_memory is not ''):
            print("Already calculated CUSTOM nmf dictionary has been loaded from MEMORY")
            self.dim_reduced_dict = custom_nmf_dict_from_memory
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

    def calc_nmf(self, num_components=20, num_iter=5000, individual=0, save_data=0, filename_prefix=''):

        """
        @brief Perform nmf on the given subset of data.

            Example usage:      nmf_num_components = 20
                                nmf_num_iter = 50
                                nmf_individual = 0
                                nmf_filename_prefix = ''
                                nmf_dict = nmf_kp_object.calc_nmf(num_components=nmf_num_components, num_iter=nmf_num_iter, individual=nmf_individual, save_data=save_data,filename_prefix=nmf_filename_prefix)



        @param num_components The number of NMF components I want to have.
        @param num_iter The number of iterations that I want the NMF to run before stopping the algorithm
        @param individual If this is set to 0, NMF will be calculated on the entire sets of data put into a single large dataset.
            If this is set to 1, NMF will be performed individually on each dataset in the data_subset
        @param save_data  If this is set to 1, data will be saved  with a filename with the prefix as given by the 'filename_prefix' parameter
        @param filename_prefix This is only used if the 'save_data' parameter is set to 1. The value entered here will be added as a prefix to
            the file that will be saved
        @return: The NMF output dictionary
        """

        tic_normalized_combined_data_array_subset = self.data_subset_dict['tic_normalized_combined_data_array_subset']
        subset_pattern_array = self.data_subset_dict['subset_pattern_array']
        self.num_dim_reduced_components = num_components
        nmf_model = NMF(n_components=num_components, max_iter=num_iter, init='random', random_state=0)

        if individual == 0:
            num_datasets = len(tic_normalized_combined_data_array_subset)
            pixel_count_array = np.empty((0, 1))

            combined_dataset_for_nmf = np.empty((0, tic_normalized_combined_data_array_subset[0].shape[1]))

            for count, dataset in enumerate(tic_normalized_combined_data_array_subset):
                combined_dataset_for_nmf = np.append(combined_dataset_for_nmf, dataset, axis=0)
                pixel_count_array = np.append(pixel_count_array, dataset.shape[0])

            w = nmf_model.fit_transform(combined_dataset_for_nmf)
            h = nmf_model.components_
            nmf_outputs = [[w], [h]]
            error = nmf_model.reconstruction_err_


        else:
            nmf_outputs = []
            error = []
            num_datasets = len(tic_normalized_combined_data_array_subset)
            pixel_count_array = np.empty((0, 1))

            for count, dataset in enumerate(tic_normalized_combined_data_array_subset):
                w = nmf_model.fit_transform(dataset)
                h = nmf_model.components_

                individual_nmf_outputs = [[w], [h]]
                individual_error = nmf_model.reconstruction_err_

                pixel_count_array = np.append(pixel_count_array, dataset.shape[0])
                nmf_outputs.append(individual_nmf_outputs)
                error.append(individual_error)

        self.dim_reduced_dict = {'dim_reduced_outputs': nmf_outputs,
                                 'error': error,
                                 'pixel_count_array': pixel_count_array,
                                 'subset_pattern_array': subset_pattern_array,
                                 'individual': individual,
                                 'dim_reduced_object': nmf_model,
                                 'global_path_name': self.dataloader_kp_object.global_path_name,
                                 'num_dim_reduced_components': num_components,
                                 'used_dim_reduction_technique': self.used_dim_reduction_technique}


        ### Saving the nmf outputs as an npy file.
        if save_data == 1:
            if individual == 1:
                np.save(self.dataloader_kp_object.global_path_name+'saved_outputs/nmf_outputs/' + filename_prefix + 'nmf_on_binsize_' + str(self.dataloader_kp_object.bin_size).replace('.', '_') + '_' + self.dataloader_kp_object.mode + '_mode_individual_datasubset_' + str(subset_pattern_array).replace(' ', '_').replace(',', '') + '_max_iter_' + str(num_iter) + '_num_components_' + str(num_components) + '.npy', self.dim_reduced_dict, 'dtype=object')
            else:
                np.save(self.dataloader_kp_object.global_path_name + 'saved_outputs/nmf_outputs/' + filename_prefix + 'nmf_on_binsize_' + str(self.dataloader_kp_object.bin_size).replace('.', '_') + '_' + self.dataloader_kp_object.mode + '_mode_combined_datasubset_' + str(subset_pattern_array).replace(' ', '_').replace(',', '') + '_max_iter_' + str(num_iter) + '_num_components_' + str(num_components) + '.npy', self.dim_reduced_dict, 'dtype=object')

        return self.dim_reduced_dict

    def calc_batch_nmf(self, batch_size=50, saved_intermediate_nmf_dict_filename='', num_components=20, num_iter=5000, save_data=1, filename_prefix='', folder_name=''):

        """
        @brief Perform nmf on the given subset of data by breaking the number of iterations into smaller batches, and saving these intermediate results in the harddisk. Then, these saved results are fed as the initializing matrix again to the NMF algorithm for the next batch of iterations. This way, we would'nt loose all progress if an NMF run with a large number of iterations crash midway through.

            Example usage:      
                                subset_pattern_array = [1, 1, 1, 1, 1, 1, 1, 1]  
                                data_subset = nmf_kp_object.data_subset_creator(subset_pattern_array=subset_pattern_array)  
                                nmf_num_components = 20
                                saved_intermediate_nmf_dict_filename = ''  # If the NMF stops or hangs before completing its iterations, simply load the most recently saved partial NMF dictionary here so that the batch nmf can start from there.
                                batch_size = 20
                                nmf_num_iter = 60
                                nmf_filename_prefix = 'batch_mode_test_negative_mode_1_binsize_'
                                save_data = 1
                                folder_name = ''
                                nmf_dict = nmf_kp_object.calc_batch_nmf(saved_intermediate_nmf_dict_filename=saved_intermediate_nmf_dict_filename, batch_size=batch_size, num_components=nmf_num_components, num_iter=nmf_num_iter, save_data=save_data,filename_prefix=nmf_filename_prefix, folder_name=folder_name)




        @param num_components The number of NMF components I want to have.
        @param num_iter The number of iterations that I want the NMF to run before stopping the algorithm
        @param save_data  If this is set to 1, data will be saved  with a filename with the prefix as given by the 'filename_prefix' parameter
        @param filename_prefix This is only used if the 'save_data' parameter is set to 1. The value entered here will be added as a prefix to
            the file that will be saved
        @param batch_size This gives the block of iterations that will be run before saving the results, and restarting NMF by using this saved data as the initialization matrices.
        @param saved_intermediate_nmf_dict_filename This can be used to re-enter a partially complete intermediate nmf dictionary that was saved, so that the nmf process can be restarted from there.

        @return: The NMF output dictionary
        """

        nmf_partial_result_save_path = self.dataloader_kp_object.global_path_name + 'saved_outputs/nmf_outputs/'
        if (folder_name not in os.listdir(nmf_partial_result_save_path)):
            os.mkdir(self.dataloader_kp_object.global_path_name + 'saved_outputs/nmf_outputs/' + folder_name)

        internal_batch_restart = 0
        tic_normalized_combined_data_array_subset = self.data_subset_dict['tic_normalized_combined_data_array_subset']
        subset_pattern_array = self.data_subset_dict['subset_pattern_array']
        self.num_dim_reduced_components = num_components

        num_datasets = len(tic_normalized_combined_data_array_subset)
        pixel_count_array = np.empty((0, 1))

        combined_dataset_for_nmf = np.empty((0, tic_normalized_combined_data_array_subset[0].shape[1]))
        for count, dataset in enumerate(tic_normalized_combined_data_array_subset):
            combined_dataset_for_nmf = np.append(combined_dataset_for_nmf, dataset, axis=0)
            pixel_count_array = np.append(pixel_count_array, dataset.shape[0])

        # del tic_normalized_combined_data_array_subset
        # del self.data_subset_dict['tic_normalized_combined_data_array_subset']
        # del self.dataloader_kp_object.combined_tic_normalized_data
        # del self.combined_tic_normalized_data

        ## The first block of iterations. This will be initialized randomly if a partially completed nmf dict is NOT given using the saved_intermediate_nmf_dict_filename.
        if (saved_intermediate_nmf_dict_filename is None) or (saved_intermediate_nmf_dict_filename == ''):
            print("Starting with random initialization")
            nmf_model = NMF(n_components=num_components, max_iter=batch_size, init='random', random_state=0)
            custom_init_used = 0
            current_batch_number = 0
        else:
            print("Starting with saved partially completed nmf result. Overriding num_iter, batch_size with saved values")
            partial_dim_reduced_dict = np.load(saved_intermediate_nmf_dict_filename, allow_pickle=True)[()]
            dim_reduced_outputs = partial_dim_reduced_dict['dim_reduced_outputs']
            partial_w_init = dim_reduced_outputs[0][0]
            partial_h_init = dim_reduced_outputs[1][0]
            nmf_model = NMF(n_components=num_components, max_iter=batch_size, init='custom', random_state=0)
            current_batch_number = partial_dim_reduced_dict['current_batch_number']
            num_iter = partial_dim_reduced_dict['num_iter']
            batch_size = partial_dim_reduced_dict['batch_size']
            custom_init_used = 1

        duration = 0

        while (current_batch_number < int(num_iter/batch_size)):
            start_time = time.time()
            print("Duration taken for the previous batch: ", duration, " minutes")
            print("Current batch number is: ", current_batch_number)
            print("Number of batches to go: ", int(num_iter / batch_size) - current_batch_number)
            if internal_batch_restart == 1: # This flag gets set when one NMF batch has run, and its result has to be fed back into the next batch.
                nmf_model = NMF(n_components=num_components, max_iter=batch_size, init='custom', random_state=0)
                partial_w_init = w
                partial_h_init = h
                custom_init_used = 1
                print("Current reconstruction error: ", error)

            if custom_init_used == 0:
                w = nmf_model.fit_transform(combined_dataset_for_nmf)
                h = nmf_model.components_
            else:
                w = nmf_model.fit_transform(combined_dataset_for_nmf, W=partial_w_init, H=partial_h_init)
                h = nmf_model.components_

            nmf_outputs = [[w], [h]]
            error = nmf_model.reconstruction_err_

            current_batch_number = current_batch_number + 1
            internal_batch_restart = 1

            duration = (time.time() - start_time)/60
            partial_dim_reduced_dict = {'dim_reduced_outputs': nmf_outputs,
                                        'used_dim_reduction_technique': self.used_dim_reduction_technique,
                                        'error': error,
                                        'pixel_count_array': pixel_count_array,
                                        'subset_pattern_array': subset_pattern_array,
                                        'individual': 0,
                                        'dim_reduced_object': nmf_model,
                                        'global_path_name': self.dataloader_kp_object.global_path_name,
                                        'num_dim_reduced_components': num_components,
                                        'current_batch_number': current_batch_number,
                                        'num_iter': num_iter,
                                        'batch_size': batch_size,
                                        'is_partially_complete': 1,
                                        'duration_for_this_batch_in_minutes': duration}

            ### Saving the partially completed nmf outputs as an npy file.
            if save_data == 1:
                np.save(self.dataloader_kp_object.global_path_name + 'saved_outputs/nmf_outputs/' + folder_name + '/' + filename_prefix + 'partial_complete_current_batch_' + str(current_batch_number) + '_nmf_on_binsize_' + str(self.dataloader_kp_object.bin_size).replace('.', '_') + '_' + self.dataloader_kp_object.mode + '_mode_combined_datasubset_' + str(subset_pattern_array).replace(' ', '_').replace(',', '') + '_max_iter_' + str(num_iter) + '_num_components_' + str(num_components) + '.npy', partial_dim_reduced_dict, 'dtype=object')
                # if current_batch_number > 1:
                #     os.remove(self.dataloader_kp_object.global_path_name + 'saved_outputs/nmf_outputs/' + folder_name + '/' + filename_prefix + 'partial_complete_current_batch_' + str(current_batch_number-1) + '_nmf_on_binsize_' + str(self.dataloader_kp_object.bin_size).replace('.', '_') + '_' + self.dataloader_kp_object.mode + '_mode_combined_datasubset_' + str(subset_pattern_array).replace(' ', '_').replace(',', '') + '_max_iter_' + str(num_iter) + '_num_components_' + str(num_components) + '.npy')

        total_duration = self.calculate_total_duration_of_a_batchwise_nmf_run(folder_name)

        self.dim_reduced_dict = {'dim_reduced_outputs': nmf_outputs,
                                 'used_dim_reduction_technique': self.used_dim_reduction_technique,
                                 'error': error,
                                 'pixel_count_array': pixel_count_array,
                                 'subset_pattern_array': subset_pattern_array,
                                 'individual': 0,
                                 'dim_reduced_object': nmf_model,
                                 'global_path_name': self.dataloader_kp_object.global_path_name,
                                 'num_dim_reduced_components': num_components,
                                 'num_iter': num_iter,
                                 'batch_size': batch_size,
                                 'is_partially_complete': 0,
                                 'total_duration': total_duration}

        ### Saving the completed nmf outputs as an npy file.
        if save_data == 1:
            np.save(self.dataloader_kp_object.global_path_name + 'saved_outputs/nmf_outputs/' + folder_name + '/' + filename_prefix + 'nmf_on_binsize_' + str(self.dataloader_kp_object.bin_size).replace('.', '_') + '_' + self.dataloader_kp_object.mode + '_mode_combined_datasubset_' + str(subset_pattern_array).replace(' ', '_').replace(',', '') + '_max_iter_' + str(num_iter) + '_num_components_' + str(num_components) + '.npy', self.dim_reduced_dict, 'dtype=object')

        print("All blocks complete. Final error: ", error)
        
        return self.dim_reduced_dict

    def nmf_downsampler(self, step=12, save_data=0, filename_prefix=''):

        """
        @brief Downsample NMF data so that it becomes easier to do prototyping work. This is only a link to the actual implementation of this function
            The actual implementation is in the dim_reduction_common_kp' class

            Example Usage:  save_data = 0
                            downsampler_step = 12
                            downsampler_prefix = 'downsampled_'
                            downsampled_nmf_dict = nmf_kp_object.nmf_downsampler(step=downsampler_step, save_data=save_data, filename_prefix=downsampler_prefix)


        @param step This gives how often an NMF output sample should be selected from the list of
            outputs. (That is, we are selecting only NMF output of every 12th pixel)
        @param save_data Save data if set to 1.
        @param filename_prefix Only required if save_data = 1.

        """
        self.downsampled_nmf_dict = dim_reduction_common_kp(self).downsampler(step=step, save_data=save_data, filename_prefix=filename_prefix)
        return self.downsampled_nmf_dict

    def order_nmf_according_to_reconstruction_error(self, save_data=0, filename_prefix=''):

        """
        THIS METHOD IS DEPRACATED. THIS HAS BEEN REPLACED BY: self.order_nmf_according_to_RESIDUAL_reconstruction_error()
        @brief Order the internal representation of NMF components according to the reconstruction error.
                Example usage:

                                save_data = 0
                                filename_prefix = ''
                                ordered_nmf_dict = nmf_kp_object.order_nmf_according_to_reconstruction_error(save_data=save_data, filename_prefix=filename_prefix)

        @param save_data Set to 1 to save the data.
        @param filename_prefix Use this prefix in the filename used to save this result.
        @return nmf_dict with nmf components ordered correctly according to their reconstruction error.
        """

        # if not(hasattr(self, 'data_subset_dict')):
        #     subset_pattern_array = self.dim_reduced_dict['subset_pattern_array']
        #     data_subset_dict = self.data_subset_creator(subset_pattern_array=subset_pattern_array)
        #
        # tic_normalized_combined_data_array_subset = self.data_subset_dict['tic_normalized_combined_data_array_subset']
        # num_datasets = len(tic_normalized_combined_data_array_subset)
        # print("Combining datasets...")
        # combined_dataset_for_nmf = np.empty((0, tic_normalized_combined_data_array_subset[0].shape[1]))
        # for count, dataset in enumerate(tic_normalized_combined_data_array_subset):
        #     combined_dataset_for_nmf = np.append(combined_dataset_for_nmf, dataset, axis=0)
        #
        # print("Combining datasets done")
        #
        # dim_reduced_outputs = self.dim_reduced_dict['dim_reduced_outputs']
        # subset_pattern_array = self.dim_reduced_dict['subset_pattern_array']
        # num_components = self.dim_reduced_dict['num_dim_reduced_components']
        # num_iter = self.dim_reduced_dict['num_iter']
        # w_array = dim_reduced_outputs[0][0]
        # h_array = dim_reduced_outputs[1][0]
        #
        # print("Calculating reconstruction errors...")
        # reconstruction_error_store = np.empty([0, 2])
        # for component_number in range(num_components):
        #     print("Now checking component: ", component_number)
        #     reconstruction_difference = combined_dataset_for_nmf - np.matmul(w_array[:, [component_number]], h_array[[component_number], :])
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
        # print("Internal representation of the NMF order got changed")
        #
        # if save_data == 1:
        #     np.save(self.dataloader_kp_object.global_path_name + '/saved_outputs/nmf_outputs/' + filename_prefix + 'combined_dataset_' + str(subset_pattern_array).replace(' ', '_').replace(',', '') + '_nmf_dict_max_iter_' + str(num_iter) + '_tic_normalized_outputs_' + str(num_components) + '.npy', self.dim_reduced_dict, 'dtype=object')
        # return self.dim_reduced_dict

    def calculate_reconstruction_error_for_given_set_of_components(self, original_combined_data_for_nmf, modified_dim_reduced_outputs):

        """
        @brief Calculate the reconstruction error for a given set of components.
        @param modified_dim_reduced_outputs: A dimensionality reduced array with a subset of the components, or all the components.
        @param original_combined_data_for_nmf: This is the ground truth dataset
        @return reconstruction error: The reconstruction error with the selected set of components
        """

        w_array = modified_dim_reduced_outputs[0][0]
        h_array = modified_dim_reduced_outputs[1][0]

        reconstruction_difference = original_combined_data_for_nmf - np.matmul(w_array, h_array)
        reconstruction_error = np.linalg.norm(reconstruction_difference, ord='fro')

        return reconstruction_error

    def reconstruction_error_vs_num_nmf_components_calculator(self, max_num_components=20, nmf_num_iter=5000, nmf_individual=0, subset_pattern_array=[1, 1, 1, 1, 1, 1, 1, 1], plot_result=0, save_plot=0, plot_file_format='svg', plot_dpi=600, save_data=0, data_filename_prefix=''):

        """
        @brief Calculate NMF for Reconstruction error vs number of NMF components used.

            Usage example:
                            *Note: nmf_kp object must be initialized before hand.

                            max_num_components = 20
                            nmf_num_iter = 50
                            nmf_individual = 0
                            subset_pattern_array = [1, 1, 1, 1, 1, 1, 1, 1]
                            plot_result = 1
                            save_plot = 1
                            plot_file_format = 'svg'
                            plot_dpi = 600
                            save_data = 1
                            data_filename_prefix = 'test_num_comp_vs_accuracy_'

                            nmf_kp_object.reconstruction_error_vs_num_nmf_components_calculator(max_num_components=max_num_components, nmf_num_iter=nmf_num_iter, nmf_individual=nmf_individual, subset_pattern_array=subset_pattern_array, plot_result=plot_result, save_plot=save_plot, plot_file_format=plot_file_format, plot_dpi=plot_dpi, save_data=save_data, data_filename_prefix=data_filename_prefix)


        @param max_num_components: Maximum number of NMF components that will be tried
        @param nmf_num_iter: Maximum number of iterations to run for each trial
        @param nmf_individual: If set to 1, perform nmf on each individual dataset (depracated). If set to 0, performs
            nmf on the entire dataset (as selected by the subset_pattern_array)   taken as a whole.
        @param subset_pattern_array:  Determines the subset of datasets to be used in the NMF calculation.
        @param plot_result: Set to 1 if the result needs to be plotted.
        @param save_plot: Set to 1 if the plot needs to be saved
        @param plot_file_format: Set to 'svg', 'png', 'pdf' etc.
        @param plot_dpi: Set the dpi when saving the plot
        @param save_data: Set to 1 if all nmf trials need to be saved. Warning: This will generate a lot of data; one result file for each NMF trial.
            Number of saved files is equal to the max_num_components variable.
        @param data_filename_prefix: Filename prefix for saving the data for each NMF trial if save_data is enabled
        @return: error_array that contains the nmf component count used and reconstruction error (2 columns) for
            that nmf component count in each row of the array.

        """

        data_subset_dict = self.data_subset_creator(subset_pattern_array)  # This is necessary because the calc_nmf() function expects a datasubsetarray dictionary
        error_array = np.empty((0,2))
        for num_comps in range(1, (max_num_components+1)):
            nmf_num_components = num_comps
            nmf_dict = self.calc_nmf(num_components=nmf_num_components, num_iter=nmf_num_iter, individual=nmf_individual, save_data=save_data, filename_prefix=data_filename_prefix)
            error_array = np.vstack((error_array, np.array([[num_comps, nmf_dict['error']]])))
            print("Current trial:  num_components: " + str(num_comps) + ", error: " + str(nmf_dict['error']))

        if plot_result == 1:
            fig, ax = plt.subplots(1,1)
            ax.plot(error_array[:, 0], error_array[:, 1])
            ax.set_xlabel('Number of NMF components', fontsize=20)
            ax.set_ylabel('Frobenius norm of the difference between original and reconstructed data matrices', fontsize=20)
            ax.set_title('Reconstruction error vs number of NMF components', fontsize=30)
            ax.tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)

            plot_title = 'Reconstruction error vs number of nmf components used'
            plt.suptitle(plot_title, x=6, y=1.5, fontsize=80)
            # plt.subplots_adjust(bottom=0, left=0, right=12, top=1)
            plt.show()

        if save_plot == 1:
            folder_name = self.dataloader_kp_object.global_path_name + '/saved_outputs/figures/'
            fig.savefig(folder_name + plot_title.replace(' ', '_') + '.' + plot_file_format, dpi=plot_dpi, bbox_inches='tight')
        print('\n')

        return error_array

    def convert_aman_nmf_output_to_match_kp_nmf_class(self, aman_nmf_filename='', save_data=0, filename_prefix=''):
        """
        @brief Convert an existing nmf object generated via libmsi into a nmf_dict_kp form compatible with my
            nmf_kp class.

            Example usage:
                            * Note: nmf_kp class must be initialized.
                            aman_nmf_filename = "D:/msi_project_data/other/from_aman/nmf_20_all_0_05.pickle"
                            save_data = 1
                            filename_prefix = 'aman_0.05_'
                            nmf_dict = nmf_kp_object.convert_aman_nmf_output_to_match_kp_nmf_class(aman_nmf_filename, save_data=save_data, filename_prefix=filename_prefix)

        @param save_data Will save the converted nmf dict if set to 1.
        @param filename_prefix Required only if save_data is set to 1.
        @param aman_nmf_filename: Path to the file containing the nmf results for combined datasets, generated via libmsi
        @return nmf_dict
        """

        
        converted_dim_reduced_dict = {}

        converted_dim_reduced_dict['subset_pattern_array'] = [1, 1, 1, 1, 1, 1, 1, 1]
        converted_dim_reduced_dict['individual'] = 'individual'
        converted_dim_reduced_dict['used_dim_reduction_technique'] = 'nmf'

        aman_nmf_dict = pickle.load(open(aman_nmf_filename, 'rb'))

        w_aman = aman_nmf_dict["nmf_data_list"]
        h_aman = aman_nmf_dict["nmf_component_spectra"]

        w_aman_rearranged = np.empty([0, h_aman.shape[0]])
        pixel_count_array = []
        for j in w_aman:
            w_aman_rearranged = np.append(w_aman_rearranged, j, axis=0)
            pixel_count_array.append(j.shape[0])

        converted_dim_reduced_dict['pixel_count_array'] = pixel_count_array
        converted_dim_reduced_dict['dim_reduced_outputs']=[[w_aman_rearranged],[h_aman.T]]
        converted_dim_reduced_dict['num_dim_reduced_components'] = w_aman_rearranged.shape[1]
        converted_dim_reduced_dict['dim_reduced_object'] = ''  # To Do: Add the nmf/PCA object from aman's nmf/pca result
        converted_dim_reduced_dict['error'] = ''  # To Do: Add the reconstruction error from aman's nmf/pca result
        converted_dim_reduced_dict['num_iter'] = ''  # To Do: Add the number of iterations from aman's nmf/pca result

        nmf_dict = converted_dim_reduced_dict

        if save_data == 1:
            folder_name = self.dataloader_kp_object.global_path_name + '/saved_outputs/nmf_outputs/'
            np.save(folder_name + filename_prefix + '_'+aman_nmf_filename.split('.')[0].split('/')[-1] + '.npy', nmf_dict, 'dtype=object')

        self.dim_reduced_dict = nmf_dict
        print('Updated the internal nmf_dict attribute in the nmf_kp_object (self.dim_reduced_dict) \n')

        return self.dim_reduced_dict

    def modify_existing_nmf_dict(self, old_saved_nmf_filename, save_data=1, modified_nmf_filename_prefix=''):

        """
        @brief This function is designed to load an existing saved nmf dictionary, and modify it to suit newer versions of the nmf_kp class
            Example use:
                        old_saved_nmf_filename  = "D:/msi_project_data/saved_outputs/nmf_outputs/no_whitening_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_20_components_nmf_dict_tic_normalized.npy"
                        save_data =  1
                        modified_nmf_filename_prefix = 'modified_'
                        modified_nmf_dict = nmf_kp_object.modify_existing_nmf_dict(old_saved_nmf_filename, save_data=save_data, modified_nmf_filename_prefix=modified_nmf_filename_prefix)

        @param old_saved_nmf_filename: This is an essential argument, This points to the existing nmf file that needs to be modified
        @param save_data:  Save the modified nmf file.
        @param modified_nmf_filename_prefix: Enter a prefix that I want the modified nmf dictionary to have if it gets saved (i.e, if save_data ==1)
            This prefix will be prepended to the name of the old_saved_nmf_filename's filename part.
        @return  modified_nmf_dict:  Returns the modified nmf dictionary
        """
        print("This code must be adjusted everytime you use it to include the modifications you intend to do")

        old_saved_nmf_dict = np.load(old_saved_nmf_filename, allow_pickle=True)[()]
        modified_nmf_dict = copy.deepcopy(old_saved_nmf_dict)

        #############################################
        ###  Do the modifications:
        ### Ex:
        ## modified_nmf_dict['error_new'] = old_saved_nmf_dict['error']
        ## modified_nmf_dict['new_element'] = 5
        # del modified_nmf_dict['nmf_outputs']
        # modified_nmf_dict['dim_reduced_outputs'] = old_saved_nmf_dict['nmf_outputs']
        #
        # del modified_nmf_dict['nmf_object']
        # modified_nmf_dict['dim_reduced_object'] = old_saved_nmf_dict['nmf_object']
        #
        # modified_nmf_dict['used_dim_reduction_technique'] = 'nmf'
        modified_nmf_dict['num_dim_reduced_components'] = 20

        #############################################

        self.dim_reduced_dict = modified_nmf_dict
        print("Internal representation of the nmf_dict variable has been updated (Added total explained variance, etc)")

        if save_data == 1:
            old_filename = old_saved_nmf_filename.split('/')[-1]
            folder_name = self.dataloader_kp_object.global_path_name + '/saved_outputs/nmf_outputs/'
            np.save(folder_name + modified_nmf_filename_prefix + old_filename, modified_nmf_dict, 'dtype=object')

        return modified_nmf_dict

    def calculate_total_duration_of_a_batchwise_nmf_run(self, partial_nmf_folder_name):
        """
        @brief Load all the saved partial nmf datasets into memory one by one, cue their duration values, and sum them together. Then print and return this value.

        @param partial_nmf_folder_name This is the folder into which all the partial nmf results are saved
        @return: returns the total duration
        """

        global_path_name = self.dataloader_kp_object.global_path_name

        folder_path = global_path_name + 'saved_outputs/nmf_outputs/' + partial_nmf_folder_name + '/'

        partial_result_filename_array = os.listdir(folder_path)

        total_duration = 0
        for count, this_partial_result_filename in enumerate(partial_result_filename_array):
            if this_partial_result_filename == '.' or this_partial_result_filename == '..':
                pass
            elif ('partial' not in this_partial_result_filename):
                pass
            else:
                print("Now processing partial dimensionality reduced dataset ", count, " out of ", len(partial_result_filename_array), " datasets")

                this_dim_reduced_dict_pathname = folder_path + this_partial_result_filename
                dim_reduced_dict = np.load(this_dim_reduced_dict_pathname, allow_pickle=True)[()]
                this_partial_dataset_duration = dim_reduced_dict['duration_for_this_batch_in_minutes']

                total_duration = total_duration + this_partial_dataset_duration

        print("Total duration for the completion of this run: ", total_duration)

        return total_duration

    def order_nmf_according_to_RESIDUAL_reconstruction_error(self, save_data=1, filename_prefix='', saved_order_statistics_filename=''):

        """
        @brief: Order NMF components according to the residual error of reconstruction after removing the next strongest component with each iteration

                Usage Example: save_data = 1
                                filename_prefix = ''
                                nmf_kp_object.order_nmf_according_to_RESIDUAL_reconstruction_error(save_data=save_data, filename_prefix=filename_prefix)

                                ## OR: Load an already ordered statistics file and use it to do the actual ordering of the nmf dict
                                # saved_order_statistics_filename = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/nmf_outputs/statistics_of_ordering_according_to_residual_reconstruction_error_redo_batch_mode_test_negative_mode_1_binsize_partial_complete_current_batch_11_individual_dataset_nmf_dict_max_iter_2000_tic_normalized_nmf_outputs_20.npy"
                                # nmf_kp_object.order_nmf_according_to_RESIDUAL_reconstruction_error(save_data=save_data, filename_prefix=filename_prefix, saved_order_statistics_filename=saved_order_statistics_filename)


        @param save_data: Whether to save data
        @param filename_prefix: Prefix to use when saving the file
        @param saved_order_statistics_filename: If this is available, the finding of the optimal order can be skipped. However, the nmf must still be aligned using this order
        @return ordered_dim_reduced_dict: Returns an ordered nmf dict
        """

        if not(saved_order_statistics_filename):
            if not (hasattr(self, 'data_subset_dict')):
                subset_pattern_array = self.dim_reduced_dict['subset_pattern_array']
                data_subset_dict = self.data_subset_creator(subset_pattern_array=subset_pattern_array)

            tic_normalized_combined_data_array_subset = self.data_subset_dict['tic_normalized_combined_data_array_subset']
            num_datasets = len(tic_normalized_combined_data_array_subset)
            print("Combining datasets...")
            combined_dataset_for_nmf = np.empty((0, tic_normalized_combined_data_array_subset[0].shape[1]))
            for count, dataset in enumerate(tic_normalized_combined_data_array_subset):
                combined_dataset_for_nmf = np.append(combined_dataset_for_nmf, dataset, axis=0)

            print("Combining datasets done")

            dim_reduced_dict_recovered = self.dim_reduced_dict
            dim_reduced_outputs_recovered = dim_reduced_dict_recovered['dim_reduced_outputs']
            num_dim_reduced_components = self.num_dim_reduced_components
            main_loop_dim_reduced_outputs_recovered = copy.deepcopy(dim_reduced_outputs_recovered)

            main_loop_comparison_table = {'count_array': [], 'component_removed': [], 'residual_error': [], 'sub_loop_comparison_table': []}

            current_component_order = np.arange(0, num_dim_reduced_components)

            recon_error_with_all_components = self.calculate_reconstruction_error_for_given_set_of_components(combined_dataset_for_nmf, main_loop_dim_reduced_outputs_recovered)
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
                    this_recon_error = self.calculate_reconstruction_error_for_given_set_of_components(combined_dataset_for_nmf, sub_loop_dim_reduced_outputs_recovered)
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

