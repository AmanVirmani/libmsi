import numpy as np
import random as rd
from sklearn.decomposition import FastICA, PCA, NMF
from sklearnex import patch_sklearn
import matplotlib.pyplot as plt
import pickle
from dim_reduction_common_kp import dim_reduction_common_kp
patch_sklearn()
rd.seed(0)


class pca_kp:

    def __init__(self, dataloader_kp_object, saved_pca_filename=None):

        """

        @brief Initializes the pca_kp object. This class is designed to perform dimensionality reduction tasks like
            NMD, PCA, ICA etc. It will also include the ability to visualize various plots for these results.

            Example usage:  from pca_kp import pca_kp
                            saved_pca_filename = "D:/msi_project_data/saved_outputs/pca_outputs/truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_pca_dict_max_iter_10000_tic_normalized_pca_outputs_20.npy"
                            pca_kp_object = pca_kp(msi_data, saved_pca_filename=saved_pca_filename)

                            OR:

                            pca_kp_object = pca_kp(msi_data)

        @param dataloader_kp_object This is an output of the dataloader_kp class. This contains all the datasets
            that will be manipulated to perform pca.
        """
        self.dataloader_kp_object = dataloader_kp_object
        self.combined_tic_normalized_data = dataloader_kp_object.combined_tic_normalized_data
        self.dataset_order = self.dataloader_kp_object.dataset_order
        self.bin_size = self.dataloader_kp_object.bin_size
        self.used_dim_reduction_technique = 'pca'

        if (saved_pca_filename is None) or (saved_pca_filename == ''):
            self.saved_dim_reduced_filename = ''
            print("Please invoke the 'data_subset_creator() function before calling calc_pca()")
            pass
        else:
            self.dim_reduced_dict = np.load(saved_pca_filename, allow_pickle=True)[()]
            self.saved_dim_reduced_filename = saved_pca_filename
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

    def calc_pca(self, num_components=20, pca_whiten=False, individual=0, save_data=0, filename_prefix=''):

        """
        @brief Perform pca on the given subset of data.

            Example usage:      pca_num_components = 20
                                pca_whiten = False
                                pca_individual = 0
                                pca_whiten = False
                                pca_filename_prefix = ''
                                pca_dict = pca_kp_object.calc_pca(num_components=pca_num_components, pca_whiten=pca_whiten, individual=pca_individual, save_data=save_data,filename_prefix=pca_filename_prefix)



        @param num_components The number of pca components I want to have.
        @param pca_whiten Refer to sklearn documentation
        @param individual If this is set to 0, pca will be calculated on the entire sets of data put into a single large dataset.
            If this is set to 1, pca will be performed individually on each dataset in the data_subset
        @param save_data  If this is set to 1, data will be saved  with a filename with the prefix as given by the 'filename_prefix' parameter
        @param filename_prefix This is only used if the 'save_data' parameter is set to 1. The value entered here will be added as a prefix to
            the file that will be saved
        @return: The pca output dictionary
        """

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

        self.dim_reduced_dict = {'dim_reduced_outputs': pca_outputs, 'explained_variance': explained_variance, 'total_explained_variance':  total_explained_variance, 'pixel_count_array': pixel_count_array,
                         'subset_pattern_array': subset_pattern_array, 'individual': individual,
                         'dim_reduced_object': pca_model, 'global_path_name': self.dataloader_kp_object.global_path_name, 'num_dim_reduced_components': num_components}

        ### Saving the pca outputs as an npy file.
        if save_data == 1:
            if individual == 1:
                np.save(self.dataloader_kp_object.global_path_name+'/saved_outputs/pca_outputs/' + filename_prefix + 'individual_dataset_' + str(subset_pattern_array) + '_' + str(num_components) + '_components_pca_dict_tic_normalized' + '.npy', self.dim_reduced_dict, 'dtype=object')

            else:
                np.save(self.dataloader_kp_object.global_path_name+'/saved_outputs/pca_outputs/' + filename_prefix + 'combined_dataset_' + str(subset_pattern_array) + '_' + str(num_components) + '_components_pca_dict_tic_normalized' + '.npy', self.dim_reduced_dict, 'dtype=object')

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
                            pca_dict = pca_kp_object.convert_aman_pca_output_to_match_kp_pca_class(aman_pca_filename)
        @param save_data Will save the converted pca dict if set to 1.
        @param filename_prefix Required only if save_data is set to 1.
        @param aman_pca_filename: Path to the file containing the pca results for combined datasets, generated via libmsi
        @return pca_dict
        """

        pca_dict = {}
        temp_pca_dict = self.dim_reduced_dict
        pca_dict['subset_pattern_array'] = temp_pca_dict['subset_pattern_array']
        pca_dict['individual'] = temp_pca_dict['individual']
        pca_dict['pixel_count_array'] = temp_pca_dict['pixel_count_array']


        aman_pca_dict = pickle.load(open(aman_pca_filename, 'rb'))
        w_aman = aman_pca_dict["pca_data_list"]
        h_aman = aman_pca_dict["components_spectra"]
        w_aman_rearranged = np.empty([0, h_aman.shape[1]])
        for i in pca_dict['pixel_count_array']:
            for j in w_aman:
                if j.shape[0] == i:
                    w_aman_rearranged = np.append(w_aman_rearranged, j, axis=0)

        pca_dict['pca_outputs'] = [[w_aman_rearranged], [h_aman.T]]
        pca_dict['pca_object'] = ''  # To Do: Add the pca object from aman's pca result
        pca_dict['explained_variance'] = '' # To Do: Add the explained_variance from aman's pca result

        if save_data == 1:
            folder_name = self.dataloader_kp_object.global_path_name + '/saved_outputs/pca_outputs/'
            np.save(folder_name + filename_prefix + 'individual_dataset_' + str(pca_dict['subset_pattern_array']) + '_pca_dict_max_iter_' + str(pca_dict['num_iter']) + '_tic_normalized_pca_outputs_' + str(pca_dict['num_components']) + '.npy', pca_dict, 'dtype=object')

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
        modified_pca_dict = old_saved_pca_dict.copy()
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

