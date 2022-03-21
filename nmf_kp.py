import numpy as np
import random as rd
from sklearn.decomposition import FastICA, PCA, NMF
from sklearnex import patch_sklearn
import matplotlib.pyplot as plt
import pickle
from dim_reduction_common_kp import dim_reduction_common_kp
patch_sklearn()
rd.seed(0)


class nmf_kp:

    def __init__(self, dataloader_kp_object, saved_nmf_filename=None):

        """

        @brief Initializes the nmf_kp object. This class is designed to perform dimensionality reduction tasks like
            NMD, PCA, ICA etc. It will also include the ability to visualize various plots for these results.

            Example usage:  from nmf_kp import nmf_kp
                            saved_nmf_filename = "D:/msi_project_data/saved_outputs/nmf_outputs/truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
                            nmf_kp_object = nmf_kp(msi_data, saved_nmf_filename=saved_nmf_filename)

                            OR:

                            nmf_kp_object = nmf_kp(msi_data)

        @param dataloader_kp_object This is an output of the dataloader_kp class. This contains all the datasets
            that will be manipulated to perform NMF.
        """
        self.dataloader_kp_object = dataloader_kp_object
        self.combined_tic_normalized_data = dataloader_kp_object.combined_tic_normalized_data
        self.dataset_order = self.dataloader_kp_object.dataset_order
        self.bin_size = self.dataloader_kp_object.bin_size
        self.used_dim_reduction_technique = 'nmf'

        if (saved_nmf_filename is None) or (saved_nmf_filename == ''):
            self.saved_dim_reduced_filename = ''
            print("Please invoke the 'data_subset_creator() function before calling calc_nmf()")
            pass
        else:
            self.dim_reduced_dict = np.load(saved_nmf_filename, allow_pickle=True)[()]
            self.saved_dim_reduced_filename = saved_nmf_filename
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

        self.dim_reduced_dict = {'dim_reduced_outputs': nmf_outputs, 'error': error, 'pixel_count_array': pixel_count_array,
                         'subset_pattern_array': subset_pattern_array, 'individual': individual,
                         'dim_reduced_object': nmf_model, 'global_path_name': self.dataloader_kp_object.global_path_name, 'num_dim_reduced_components': num_components}


        ### Saving the nmf outputs as an npy file.
        if save_data == 1:
            if individual == 1:
                np.save(self.dataloader_kp_object.global_path_name+'/saved_outputs/nmf_outputs/' + filename_prefix + 'individual_dataset_' + str(subset_pattern_array) + '_nmf_dict_max_iter_' + str(num_iter) + '_tic_normalized_nmf_outputs_' + str(num_components) + '.npy', self.dim_reduced_dict, 'dtype=object')
            else:
                np.save(self.dataloader_kp_object.global_path_name+'/saved_outputs/nmf_outputs/' + filename_prefix + 'combined_dataset_' + str(subset_pattern_array) + '_nmf_dict_max_iter_' + str(num_iter) + '_tic_normalized_nmf_outputs_' + str(num_components) + '.npy', self.dim_reduced_dict, 'dtype=object')

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

        nmf_dict = {}
        temp_nmf_dict = self.dim_reduced_dict
        nmf_dict['subset_pattern_array'] = temp_nmf_dict['subset_pattern_array']
        nmf_dict['individual'] = temp_nmf_dict['individual']
        nmf_dict['pixel_count_array'] = temp_nmf_dict['pixel_count_array']


        aman_nmf_dict = pickle.load(open(aman_nmf_filename, 'rb'))
        w_aman = aman_nmf_dict["nmf_data_list"]
        h_aman = aman_nmf_dict["components_spectra"]
        w_aman_rearranged = np.empty([0, h_aman.shape[1]])
        for i in nmf_dict['pixel_count_array']:
            for j in w_aman:
                if j.shape[0] == i:
                    w_aman_rearranged = np.append(w_aman_rearranged, j, axis=0)

        nmf_dict['dim_reduced_outputs'] = [[w_aman_rearranged], [h_aman.T]]
        nmf_dict['dim_reduced_object'] = ''  # To Do: Add the nmf object from aman's nmf result
        nmf_dict['error'] = ''  # To Do: Add the reconstruction error from aman's nmf result
        nmf_dict['num_iter'] = ''  # To Do: Add the number of iterations from aman's nmf result


        if save_data == 1:
            folder_name = self.dataloader_kp_object.global_path_name + '/saved_outputs/nmf_outputs/'
            # np.save(folder_name + filename_prefix + 'individual_dataset_' + str(nmf_dict['subset_pattern_array']) + '_nmf_dict_max_iter_' + str(nmf_dict['num_iter']) + '_tic_normalized_nmf_outputs_' + str(nmf_dict['num_components']) + '.npy', nmf_dict, 'dtype=object')
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
        modified_nmf_dict = old_saved_nmf_dict.copy()

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
        # modified_nmf_dict['num_dim_reduced_components'] = 20

        #############################################

        self.dim_reduced_dict = modified_nmf_dict
        print("Internal representation of the nmf_dict variable has been updated (Added total explained variance, etc)")

        if save_data == 1:
            old_filename = old_saved_nmf_filename.split('/')[-1]
            folder_name = self.dataloader_kp_object.global_path_name + '/saved_outputs/nmf_outputs/'
            np.save(folder_name + modified_nmf_filename_prefix + old_filename, modified_nmf_dict, 'dtype=object')

        return modified_nmf_dict
