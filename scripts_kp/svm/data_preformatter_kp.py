import os

import numpy as np
import copy

class data_preformatter_kp:
    """
    @brief This class does preformatting of an NMF, PCA, ICA object etc to eventually enable SVM computation.
    """

    def __init__(self, dim_reduced_object, cph_label=0, naive_label=1):

        """
              @brief Initializes the data_preformatter_kp object. This class is designed to perform data preformatting of
               NMF, PCA, ICA objects to enable SVM computation.

            Example usage:
                from data_preformatter_kp import data_preformatter_kp
                data_preformatter_object = data_preformatter_kp(nmf_kp_object)

        @param dim_reduced_object This is an output of the dimensionality reduction class such as nmf_kp.
        @param cph_label Set this to the value that should be assigned as a cph label. Default is 0
        @param naive_label Set this to the value that should be assigned as a naive label. Default is 1


        """
        self.cph_label = cph_label
        self.naive_label = naive_label
        self.dim_reduced_object = dim_reduced_object
        self.used_dim_reduction_technique = self.dim_reduced_object.used_dim_reduction_technique
        self.num_dim_reduced_components = self.dim_reduced_object.num_dim_reduced_components
        self.num_datasets = len(self.dim_reduced_object.dataloader_kp_object.combined_data_object)
        self.dataset_order = self.dim_reduced_object.dataloader_kp_object.dataset_order
        self.class_labels = []
        self.cph_dataset_names = []
        self.naive_dataset_names = []
        for i in self.dataset_order:
            if 'cph' in i:
                self.class_labels.append(cph_label)
                self.cph_dataset_names.append(i)

            elif 'naive' in i:
                self.class_labels.append(naive_label)
                self.naive_dataset_names.append(i)

    def create_3d_array_from_dim_reduced_data(self, saved_dim_reduced_filename=None, save_data=0, filename_prefix=''):

        """
        @brief This creates a 3D array out of dimensionality reduced data. That is, this converts the
            2D format of dim reduced data back into the spatial domain so that we get num_dim_reduced_components x (ion_image)
            number of ion images.

            ### Calling example:

            saved_dim_reduced_filename='/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/nmf_outputs/truncated_with_dummy_data_3_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 1]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy'
            output_3d_array_filename_prefix=''
            save_data=1
            datagrid_store_dict=data_preformatter_kp_object.create_3d_array_from_dim_reduced_data(saved_dim_reduced_filename=saved_dim_reduced_filename , save_data=save_data, filename_prefix=output_3d_array_filename_prefix )

            OR:

            datagrid_store_dict=data_preformatter_kp_object.create_3d_array_from_dim_reduced_data(save_data=save_data, filename_prefix=output_3d_array_filename_prefix )


        @param saved_dim_reduced_filename THis is an optional parameter, which, if provided as a path, will cause this piece of
            code to use that saved dim_reduced object instead of the dim_reduced object that came in through the initialization
            of this class.
        @param save_data Will save data if this is enabled
        @param filename_prefix Will be only necessary if save data is enabled.
        @return Returns a 3D version of a dim reduced dataset, in the form of a dictionary.


        """


        if (saved_dim_reduced_filename is not None) and (saved_dim_reduced_filename != ''):
            dim_reduced_dict_recovered = np.load(saved_dim_reduced_filename, allow_pickle=True)[()]
        else:
            dim_reduced_dict_recovered = self.dim_reduced_object.dim_reduced_dict

        dim_reduced_outputs_recovered = dim_reduced_dict_recovered['dim_reduced_outputs']

        dim_reduced_dataset = dim_reduced_outputs_recovered[0][0]

        pix_count_previous = 0
        datagrid_store = []

        for i in range(self.num_datasets):
            num_rows = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].rows
            num_cols = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].cols
            coordinate_array = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].coordinates
            pix_count = len(self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].coordinates)
            individual_dim_reduced_data = dim_reduced_dataset[pix_count_previous: pix_count_previous + pix_count, :]
            pix_count_previous = pix_count_previous + pix_count


            if self.used_dim_reduction_technique == 'nmf':
                datagrid = np.ones([self.num_dim_reduced_components, num_rows, num_cols]) * -1
            elif self.used_dim_reduction_technique == 'pca':
                datagrid = np.ones([self.num_dim_reduced_components, num_rows, num_cols]) * 1000

            for temp1 in range(self.num_dim_reduced_components):
                for temp2 in range(pix_count):
                    this_col = coordinate_array[temp2][0]
                    this_row = coordinate_array[temp2][1]
                    datagrid[temp1, this_row - 1, this_col - 1] = individual_dim_reduced_data[temp2, temp1]

            datagrid_store.append(datagrid)

        datagrid_store_dict = {'datagrid_store': datagrid_store,
                               'used_dim_reduced_file': saved_dim_reduced_filename,
                               'used_dim_reduction_technique': self.used_dim_reduction_technique,
                               'global_path_name': self.dim_reduced_object.dataloader_kp_object.global_path_name,
                               'num_dim_reduced_components': self.num_dim_reduced_components,
                               'num_datasets': self.num_datasets}

        folder_name = self.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/output_3d_datagrids/'
        file_name = filename_prefix + 'output_3d_datagrid_store_' + self.used_dim_reduction_technique + '_' + str(self.num_dim_reduced_components) + '_dim_reduced_components' + '_ver_1.npy'

        if save_data == 1:
            for count in range(2, 20):
                dir_list = os.listdir(folder_name)
                if file_name in dir_list:
                    file_name = file_name[0:-5] + str(count) + '.npy'

                else:
                    # print("File exists. New run number: "+ str(count-1))
                    break

            np.save(folder_name+file_name, datagrid_store_dict)

        self.datagrid_store_dict = datagrid_store_dict

        return datagrid_store_dict

    def image_patches_unrolled(self, saved_3d_datagrid_filename=None, reject_percentage=25, window_size=30, overlap=0, save_data=0, filename_prefix=''):

        """
        @brief This generates patches of images out of 3D data, and generate both a copy of these patches in 3D, as well
            as patches that have been unrolled into a vector.

            ### Calling example:
            # saved_3d_datagrid_filename='/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/output_3d_datagrids/version_2_output_3d_datagrid_store_20_nmf_components.npy'
            # image_patch_reject_percentage=25
            # image_patch_window_size=30
            # image_patch_overlap=0
            # image_patch_filename_prefix=''
            # save_data=0

            # image_patch_data_store_dict = data_preformatter_kp_object.image_patches_unrolled(saved_3d_datagrid_filename=saved_3d_datagrid_filename, reject_percentage=image_patch_reject_percentage, window_size=image_patch_window_size, overlap=image_patch_overlap, save_data=save_data, filename_prefix=image_patch_filename_prefix)

            # OR:

            # image_patch_data_store_dict = data_preformatter_kp_object.image_patches_unrolled(reject_percentage=image_patch_reject_percentage, window_size=image_patch_window_size, overlap=image_patch_overlap, save_data=save_data, filename_prefix=image_patch_filename_prefix)

        @param saved_3d_datagrid_filename: THis is an optional parameter, which, if provided as a path, will cause this piece of
            code to use that saved dim_3d_datagrid dict instead of the self.datagrid_store_dict object that came in through the class.
        @param reject_percentage: Defines the percentage of the background area of an nmf ion image which above which,
            if included in an image patch, the image patch gets rejected.
        @param window_size: Defines the window size, i.e, the size of a patch. Patches are square in shape in the spatial axes.
        @param overlap: This defines if image patches can overlap with each other.
        @param save_data: Will save data if this is enabled
        @param filename_prefix: Will be only necessary if save data is enabled.
        @return: Returns a dictionary containing an nmf dataset divided into patches defined by the above parameters.
        """

        if (saved_3d_datagrid_filename is not None) and (saved_3d_datagrid_filename != ''):
            datagrid_store_dict_recovered = np.load(saved_3d_datagrid_filename, allow_pickle=True)[()]
        else:
            datagrid_store_dict_recovered = copy.deepcopy(self.datagrid_store_dict)

        datagrid_store_recovered = copy.deepcopy(datagrid_store_dict_recovered['datagrid_store'])
        num_dim_reduced_components = self.num_dim_reduced_components
        used_dim_reduction_technique = self.used_dim_reduction_technique

        data_store_flattened = []
        data_store = []

        print("Any patch containing more than " + str(reject_percentage) + "% of background will be discarded")

        if overlap == 0:
            print("Overlapping is NOT allowed. Patches will be selected from a uniform grid")
            for i in range(self.num_datasets):
                row_start = 0
                col_start = 0
                num_rows = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].rows
                num_cols = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].cols
                this_image = copy.deepcopy(datagrid_store_recovered[i])

                individual_dataset_patches_flattened = np.empty([0, num_dim_reduced_components * window_size * window_size])
                individual_dataset_patches = np.empty([0, num_dim_reduced_components, window_size, window_size])

                while (row_start <= num_rows - window_size - 1):
                    while (col_start <= num_cols - window_size - 1):

                        temp1 = copy.deepcopy(this_image[:, row_start:row_start + window_size, col_start:col_start + window_size])
                        data_patch_flattened = np.array([temp1.flatten()])
                        data_patch = temp1
                        col_start = col_start + window_size

                        ### Remove image patches that have too much background
                        if (used_dim_reduction_technique == 'nmf'):
                            if (100 * np.count_nonzero(data_patch_flattened == -1) / data_patch_flattened.shape[1]) > reject_percentage:
                                continue
                            else:
                                ### For image patches that are acceptable, replace the background -1 values with 0.
                                data_patch_flattened[data_patch_flattened == -1] = 0
                                data_patch[data_patch == -1] = 0
                        elif (used_dim_reduction_technique == 'pca'):
                            if (100 * np.count_nonzero(data_patch_flattened == 1000) / data_patch_flattened.shape[1]) > reject_percentage:
                                continue
                            else:
                                ### For image patches that are acceptable, replace the background 1000 values with 0.
                                data_patch_flattened[data_patch_flattened == 1000] = 0
                                data_patch[data_patch == 1000] = 0

                        individual_dataset_patches_flattened = np.append(individual_dataset_patches_flattened, data_patch_flattened, axis=0)
                        individual_dataset_patches = np.append(individual_dataset_patches, [data_patch], axis=0)
                    col_start = 0
                    row_start = row_start + window_size

                data_store_flattened.append(individual_dataset_patches_flattened)
                data_store.append(individual_dataset_patches)


        elif overlap == 1:
            print("Overlapping is allowed")

        ### Adding labels
        labels_array = self.class_labels
        labelled_data_store_flattened = []
        combined_labelled_flattened_data = np.empty([0, data_store_flattened[0].shape[1] + 1])
        for count, i in enumerate(data_store_flattened):
            temp5 = np.append(i, np.ones([i.shape[0], 1]) * labels_array[count], axis=1)
            combined_labelled_flattened_data = np.append(combined_labelled_flattened_data, temp5, axis=0)
            labelled_data_store_flattened.append(temp5)

        image_patch_data_store_dict = {'combined_labelled_flattened_data': combined_labelled_flattened_data,
                                       'labelled_data_store_flattened': labelled_data_store_flattened,
                                       'data_store': data_store,
                                       'reject_percentage': reject_percentage,
                                       'window_size': window_size,
                                       'used_3d_datagrid_filename': saved_3d_datagrid_filename,
                                       'used_dim_reduction_technique': used_dim_reduction_technique,
                                       'global_path_name': self.dim_reduced_object.dataloader_kp_object.global_path_name,
                                       'num_dim_reduced_components': self.num_dim_reduced_components,
                                       'num_datasets': self.num_datasets,
                                       'cph_label': self.cph_label,
                                       'naive_label': self.naive_label,
                                       'label_order_array': self.class_labels
                                       }


        folder_name = self.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/image_patch/'
        file_name = filename_prefix + 'image_patch_data_' + used_dim_reduction_technique + '_' + str(num_dim_reduced_components) + '_dim_reduced_components_window_' + str(window_size) + '_rejection_' + str(reject_percentage) + '_ver_1.npy'

        if save_data == 1:
            for count in range(2, 20):
                dir_list = os.listdir(folder_name)
                if file_name in dir_list:
                    file_name = file_name[0:-5] + str(count) + '.npy'

                else:
                    # print("File exists. New run number: "+ str(count-1))
                    break

            np.save(folder_name+file_name, image_patch_data_store_dict)

        self.image_patch_data_store_dict = image_patch_data_store_dict

        return image_patch_data_store_dict

    def separate_training_testing_data(self, saved_image_patch_data_filename=None, training_percentage=80, random_select=1, save_data=0, filename_prefix='', repeatable_random_values=0, totally_unseen_test_data=0, totally_unseen_test_data_override_percentage=0, fix_patch_count_per_dataset=1, use_externally_provided_unseen_test_dataset=0, externally_provided_datasets_as_unseen_test_data={'test_cph_dataset_index':3, 'test_naive_dataset_index':3}, shuffle_patches_across_datasets=1):

        """
        @brief This seperates an image patch dataset into training and testing sets.

            # saved_image_patch_data_filename='/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/image_patch/version_2_image_patch_data_20_nmf_components_window_30_rejection_25.npy'
            # data_segration_training_percentage=80
            # data_segregation_random_select=1
            # data_segregation_filename_prefix=''
            # save_data=0
            # totally_unseen_test_data = 1
            # totally_unseen_test_data_override_percentage = 0  ## This can accept a percentage. If a non zero value is given for this, it will be interpretted as a percentage, a percentage of the totally unseen test data which will actually be forcibly included in the train data, hence making part of the totally unseen test data to be actually seen.
            # use_externally_provided_unseen_test_dataset = 0
            # externally_provided_datasets_as_unseen_test_data = {'test_cph_dataset_index':3, 'test_naive_dataset_index':3}  ## Only used if "use_externally_provided_unseen_test_dataset" is set to 1
            # repeatable_random_values = 1  # Set this to 1 if you want the data to be split randomly, but need those random values to be reproducible between multiple function calls.
            # fix_patch_count_per_dataset = 1 #Set this to 1 to ensure that there will be the same number of patches from each dataset.
            # shuffle_patches_across_datasets = 1 # THis addresses the problem of gridsearchcv algorithm not shuffling the patches randomly.

            # segregated_data_dict=data_preformatter_kp_object.separate_training_testing_data(saved_image_patch_data_filename=saved_image_patch_data_filename, training_percentage=data_segration_training_percentage, random_select=data_segregation_random_select, save_data = save_data, filename_prefix=data_segregation_filename_prefix, repeatable_random_values=repeatable_random_values, totally_unseen_test_data=totally_unseen_test_data, fix_patch_count_per_dataset=fix_patch_count_per_dataset)

            # OR

            # segregated_data_dict=data_preformatter_kp_object.separate_training_testing_data(training_percentage=data_segration_training_percentage, random_select=data_segregation_random_select, save_data = save_data, filename_prefix=data_segregation_filename_prefix, repeatable_random_values=repeatable_random_values, totally_unseen_test_data=totally_unseen_test_data, fix_patch_count_per_dataset=fix_patch_count_per_dataset)

        @param saved_image_patch_data_filename: THis is an optional parameter, which, if provided as a path, will cause this piece of
            code to use that saved dim_3d_datagrid dict instead of the self.datagrid_store_dict object that came in through the class.
        @param training_percentage Defines the percentage of data that needs to be set as training data. Accordingly, the testing data will
            be the remaining data.
        @param random_select This parameter, if set to 1, will shuffle the dataset each time when generating the training vs testing data.
            If this is set to 0, then, everytime, we will get the same sets of training and testing data.
        @param save_data: Will save data if this is enabled
        @param filename_prefix: Will be only necessary if save data is enabled.
        @param repeatable_random_values: Set this to 1 if you want the data to be split randomly, but need those random values to be reproducible between multiple function calls.
        @param totally_unseen_test_data: If this is set to 1, keep aside a CPH and NAIVE dataset completely when training. Do not use any patch at all from these two entire datasets during training.
        @param totally_unseen_test_data_override_percentage: This can accept a percentage. If a non zero value is given for this, it will be interpretted as a percentage, a percentage of the totally unseen test data which will actually be forcibly included in the train data, hence making part of the totally unseen test data to be actually seen.
        @param use_externally_provided_unseen_test_dataset: If this is set to 1, we are able to externally provide the index of the datasets that we need to totally set aside for testing purpose.
        @param externally_provided_datasets_as_unseen_test_data: A dictionary of the following format that will only be used if the variable  "use_externally_provided_unseen_test_dataset" is set to 1. {'test_cph_dataset_index':3, 'test_naive_dataset_index':3}
        @param shuffle_patches_across_datasets: THis addresses the problem of gridsearchcv algorithm not shuffling the patches randomly.

        @param fix_patch_count_per_dataset: Set this to 1 to ensure that there will be the same number of patches from each dataset.
        @return: Returns a dictionary containing an nmf dataset divided into patches defined by the above parameters.


        """

        if (saved_image_patch_data_filename is not None) and (saved_image_patch_data_filename != ''):
            image_patch_data_store_dict_recovered = np.load(saved_image_patch_data_filename, allow_pickle=True)[()]
        else:
            image_patch_data_store_dict_recovered = self.image_patch_data_store_dict

        ####################################
        ## Separate the list of cph and naive datasets into two lists, each containing sublists for each animal
        cph_labelled_datasets = []
        naive_labelled_datasets = []
        for temp_count, labelled_dataset in enumerate(image_patch_data_store_dict_recovered['labelled_data_store_flattened']):
            if self.class_labels[temp_count] == self.cph_label:
                cph_labelled_datasets.append(labelled_dataset)
            elif self.class_labels[temp_count] == self.naive_label:
                naive_labelled_datasets.append(labelled_dataset)

        if fix_patch_count_per_dataset == 1:  # Determine the lowest number of patches possible from each dataset such that the number of patches from each dataset is the same
            limiting_patch_count = np.inf
            for i in cph_labelled_datasets:
                if i.shape[0] < limiting_patch_count:
                    limiting_patch_count = i.shape[0]
            for i in naive_labelled_datasets:
                if i.shape[0] < limiting_patch_count:
                    limiting_patch_count = i.shape[0]

            print("Equal number of patches selected from each dataset : " + str(limiting_patch_count))
        elif fix_patch_count_per_dataset == 0:
            limiting_patch_count = None

        ####################################
        ## Shuffle the datasets individually
        individually_shuffled_cph_labelled_datasets = []
        individually_shuffled_naive_labelled_datasets = []

        for i in cph_labelled_datasets:
            this_shuffled_dataset = i[np.random.permutation(np.arange(0, i.shape[0]))]
            individually_shuffled_cph_labelled_datasets.append(this_shuffled_dataset)
        for i in naive_labelled_datasets:
            this_shuffled_dataset = i[np.random.permutation(np.arange(0, i.shape[0]))]
            individually_shuffled_naive_labelled_datasets.append(this_shuffled_dataset)

        ######################################
        ## Start segregating data

        if totally_unseen_test_data == 0:  # Test data will also be patches from animals which the svm classifier has already seen during training
            if random_select == 1:
                if repeatable_random_values == 1:
                    print("KP warning: Repeatable random numbers will be used because a seed has been set.")
                    np.random.seed(1)
                ####################################
                ## Divide training and testing data from each dataset individually
                cph_labelled_training_data = []
                naive_labelled_training_data = []
                cph_labelled_testing_data = []
                naive_labelled_testing_data = []

                for i in individually_shuffled_cph_labelled_datasets:
                    if fix_patch_count_per_dataset == 1:
                        temp_limiting_patch_count = limiting_patch_count
                    else:
                        temp_limiting_patch_count = i.shape[0]
                    limit_enforced_this_dataset = i[0:temp_limiting_patch_count, :]
                    this_dataset_divide_index = int(training_percentage / 100 * limit_enforced_this_dataset.shape[0])
                    limit_enforced_this_dataset_training_portion = limit_enforced_this_dataset[0:this_dataset_divide_index, :]
                    limit_enforced_this_dataset_testing_portion = limit_enforced_this_dataset[this_dataset_divide_index:-1, :]
                    cph_labelled_training_data.append(limit_enforced_this_dataset_training_portion)
                    cph_labelled_testing_data.append(limit_enforced_this_dataset_testing_portion)

                for i in individually_shuffled_naive_labelled_datasets:
                    if fix_patch_count_per_dataset == 1:
                        temp_limiting_patch_count = limiting_patch_count
                    else:
                        temp_limiting_patch_count = i.shape[0]
                    limit_enforced_this_dataset = i[0:temp_limiting_patch_count, :]
                    this_dataset_divide_index = int(training_percentage / 100 * limit_enforced_this_dataset.shape[0])
                    limit_enforced_this_dataset_training_portion = limit_enforced_this_dataset[0:this_dataset_divide_index, :]
                    limit_enforced_this_dataset_testing_portion = limit_enforced_this_dataset[this_dataset_divide_index:-1, :]
                    naive_labelled_training_data.append(limit_enforced_this_dataset_training_portion)
                    naive_labelled_testing_data.append(limit_enforced_this_dataset_testing_portion)

                ####################################
                ## Combine all cph data into a single numpy array, and combine all naive data into a separate numpy array

                combined_cph_train_patch_dataset_names = np.empty([0, 1])
                combined_cph_test_patch_dataset_names = np.empty([0, 1])
                combined_cph_train_data = np.empty([0, cph_labelled_training_data[0].shape[1]])
                combined_cph_test_data = np.empty([0, cph_labelled_testing_data[0].shape[1]])
                for i in range(len(cph_labelled_datasets)):
                    combined_cph_train_data = np.append(combined_cph_train_data, cph_labelled_training_data[i], axis=0)
                    combined_cph_test_data = np.append(combined_cph_test_data, cph_labelled_testing_data[i], axis=0)
                    combined_cph_train_patch_dataset_names = np.append(combined_cph_train_patch_dataset_names, np.repeat(np.array([[self.cph_dataset_names[i]]]), cph_labelled_training_data[i].shape[0], axis=0))
                    combined_cph_test_patch_dataset_names = np.append(combined_cph_test_patch_dataset_names, np.repeat(np.array([[self.cph_dataset_names[i]]]), cph_labelled_testing_data[i].shape[0], axis=0))

                combined_naive_train_patch_dataset_names = np.empty([0, 1])
                combined_naive_test_patch_dataset_names = np.empty([0, 1])
                combined_naive_train_data = np.empty([0, naive_labelled_training_data[0].shape[1]])
                combined_naive_test_data = np.empty([0, naive_labelled_testing_data[0].shape[1]])
                for i in range(len(naive_labelled_datasets)):
                    combined_naive_train_data = np.append(combined_naive_train_data, naive_labelled_training_data[i], axis=0)
                    combined_naive_test_data = np.append(combined_naive_test_data, naive_labelled_testing_data[i], axis=0)
                    combined_naive_train_patch_dataset_names = np.append(combined_naive_train_patch_dataset_names, np.repeat(np.array([[self.naive_dataset_names[i]]]), naive_labelled_training_data[i].shape[0], axis=0))
                    combined_naive_test_patch_dataset_names = np.append(combined_naive_test_patch_dataset_names, np.repeat(np.array([[self.naive_dataset_names[i]]]), naive_labelled_testing_data[i].shape[0], axis=0))

                ####################################
                ## Combine the cph and naive datasets into a single training array and a single testing array
                x_train = np.append(combined_cph_train_data[:, 0:-1], combined_naive_train_data[:, 0:-1], axis=0)
                y_train = np.append(combined_cph_train_data[:, [-1]], combined_naive_train_data[:, [-1]], axis=0)

                x_test = np.append(combined_cph_test_data[:, 0:-1], combined_naive_test_data[:, 0:-1], axis=0)
                y_test = np.append(combined_cph_test_data[:, [-1]], combined_naive_test_data[:, [-1]], axis=0)

                ####################################
                ## Combine the cph and naive patch dataset names for training and testing data into a single array
                train_patch_dataset_names = np.append(combined_cph_train_patch_dataset_names, combined_naive_train_patch_dataset_names, axis=0)
                test_patch_dataset_names = np.append(combined_cph_test_patch_dataset_names, combined_naive_test_patch_dataset_names, axis=0)

                if shuffle_patches_across_datasets == 1:
                    print("Final shuffle across datasets enabled")
                    x_train_shuffled_indices = np.random.permutation(np.arange(0, x_train.shape[0]))
                    y_train_shuffled_indices = x_train_shuffled_indices
                    x_test_shuffled_indices = np.random.permutation(np.arange(0, x_test.shape[0]))
                    y_test_shuffled_indices = x_test_shuffled_indices

                    x_train = x_train[x_train_shuffled_indices]
                    y_train = y_train[y_train_shuffled_indices]
                    train_patch_dataset_names = train_patch_dataset_names[x_train_shuffled_indices]

                    x_test = x_test[x_test_shuffled_indices]
                    y_test = y_test[y_test_shuffled_indices]
                    test_patch_dataset_names = test_patch_dataset_names[x_test_shuffled_indices]

            elif random_select == 0:
                print("Selecting determinately")

            print("Using patches from all datasets for both training and testing")

            file_name = filename_prefix + 'segregated_data_training_percentage_' + str(training_percentage) + '_random_select_' + str(random_select) + '_' + self.used_dim_reduction_technique + '_ver_1.npy'

        elif totally_unseen_test_data == 1:  # Set aside a CPH dataset and a naive dataset on its entirety for testing. # Test data will be patches from animals which the svm classifier has NEVER seen during training

            if random_select == 1 :
                ## In this case, select which cph and naive datasets should be left out for testing randomly
                if repeatable_random_values == 1:
                    print("KP warning: Repeatable random numbers will be used because a seed has been set.")
                    np.random.seed(1)

            if use_externally_provided_unseen_test_dataset == 0:
                test_data_cph_index = np.random.randint(len(self.cph_dataset_names))
                test_data_naive_index = np.random.randint(len(self.naive_dataset_names))

            elif use_externally_provided_unseen_test_dataset == 1:
                ## In this case, select which cph and naive datasets should be left out for testing determinately (I use the externally provided indices to get this result)
                print("Using an externally provided pair of cph and naive datasets as totally unseen test data")
                test_data_cph_index = externally_provided_datasets_as_unseen_test_data['test_cph_dataset_index']
                test_data_naive_index = externally_provided_datasets_as_unseen_test_data['test_naive_dataset_index']


            ########################################
            ## Separate cph and naive data into training and testing arrays

            combined_cph_train_patch_dataset_names = np.empty([0, 1])
            combined_cph_test_patch_dataset_names = np.empty([0, 1])
            combined_cph_train_data = np.empty([0, cph_labelled_datasets[0].shape[1]])
            combined_cph_test_data = np.empty([0, cph_labelled_datasets[0].shape[1]])
            for count, this_dataset in enumerate(individually_shuffled_cph_labelled_datasets):
                if fix_patch_count_per_dataset == 1:
                    this_dataset = this_dataset[0:limiting_patch_count, :]
                else:
                    this_dataset = this_dataset
                if count == test_data_cph_index:
                    if totally_unseen_test_data_override_percentage != 0:
                        forced_patch_count = int(totally_unseen_test_data_override_percentage / 100 * this_dataset.shape[0])
                        combined_cph_train_data = np.append(combined_cph_train_data, this_dataset[0:forced_patch_count, :], axis=0)
                        combined_cph_train_patch_dataset_names = np.append(combined_cph_train_patch_dataset_names, np.repeat(np.array([[self.cph_dataset_names[count]]]), this_dataset[0:forced_patch_count, :].shape[0], axis=0))
                        combined_cph_test_data = np.append(combined_cph_test_data, this_dataset[forced_patch_count:, :], axis=0)
                        combined_cph_test_patch_dataset_names = np.append(combined_cph_test_patch_dataset_names, np.repeat(np.array([[self.cph_dataset_names[count]]]), this_dataset[forced_patch_count:, :].shape[0], axis=0))
                        print(str(forced_patch_count) + " patches of supposed to be totally unseen cph test animal is included in the training set")
                    elif totally_unseen_test_data_override_percentage == 0:
                        combined_cph_test_data = np.append(combined_cph_test_data, this_dataset, axis=0)
                        combined_cph_test_patch_dataset_names = np.append(combined_cph_test_patch_dataset_names, np.repeat(np.array([[self.cph_dataset_names[count]]]), this_dataset.shape[0], axis=0))

                else:
                    combined_cph_train_data = np.append(combined_cph_train_data, this_dataset, axis=0)
                    combined_cph_train_patch_dataset_names = np.append(combined_cph_train_patch_dataset_names, np.repeat(np.array([[self.cph_dataset_names[count]]]), this_dataset.shape[0], axis=0))

            combined_naive_train_patch_dataset_names = np.empty([0, 1])
            combined_naive_test_patch_dataset_names = np.empty([0, 1])
            combined_naive_train_data = np.empty([0, naive_labelled_datasets[0].shape[1]])
            combined_naive_test_data = np.empty([0, naive_labelled_datasets[0].shape[1]])
            for count, this_dataset in enumerate(individually_shuffled_naive_labelled_datasets):
                if fix_patch_count_per_dataset == 1:
                    this_dataset = this_dataset[0:limiting_patch_count, :]
                else:
                    this_dataset = this_dataset
                if count == test_data_naive_index:
                    if totally_unseen_test_data_override_percentage != 0:
                        forced_patch_count = int(totally_unseen_test_data_override_percentage / 100 * this_dataset.shape[0])
                        combined_naive_train_data = np.append(combined_naive_train_data, this_dataset[0:forced_patch_count, :], axis=0)
                        combined_naive_train_patch_dataset_names = np.append(combined_naive_train_patch_dataset_names, np.repeat(np.array([[self.naive_dataset_names[count]]]), this_dataset[0:forced_patch_count, :].shape[0], axis=0))
                        combined_naive_test_data = np.append(combined_naive_test_data, this_dataset[forced_patch_count:, :], axis=0)
                        combined_naive_test_patch_dataset_names = np.append(combined_naive_test_patch_dataset_names, np.repeat(np.array([[self.naive_dataset_names[count]]]), this_dataset[forced_patch_count:, :].shape[0], axis=0))
                        print(str(forced_patch_count) + " patches of supposed to be totally unseen naive test animal is included in the training set")
                    elif totally_unseen_test_data_override_percentage == 0:
                        combined_naive_test_data = np.append(combined_naive_test_data, this_dataset, axis=0)
                        combined_naive_test_patch_dataset_names = np.append(combined_naive_test_patch_dataset_names, np.repeat(np.array([[self.naive_dataset_names[count]]]), this_dataset.shape[0], axis=0))

                else:
                    combined_naive_train_data = np.append(combined_naive_train_data, this_dataset, axis=0)
                    combined_naive_train_patch_dataset_names = np.append(combined_naive_train_patch_dataset_names, np.repeat(np.array([[self.naive_dataset_names[count]]]), this_dataset.shape[0], axis=0))


            ## Combine the cph and naive datasets into a single training array and a single testing array
            x_train = np.append(combined_cph_train_data[:, 0:-1], combined_naive_train_data[:, 0:-1], axis=0)
            y_train = np.append(combined_cph_train_data[:, [-1]], combined_naive_train_data[:, [-1]], axis=0)

            x_test = np.append(combined_cph_test_data[:, 0:-1], combined_naive_test_data[:, 0:-1], axis=0)
            y_test = np.append(combined_cph_test_data[:, [-1]], combined_naive_test_data[:, [-1]], axis=0)

            ####################################
            ## Combine the cph and naive patch dataset names for training and testing data into a single array
            train_patch_dataset_names = np.append(combined_cph_train_patch_dataset_names, combined_naive_train_patch_dataset_names, axis=0)
            test_patch_dataset_names = np.append(combined_cph_test_patch_dataset_names, combined_naive_test_patch_dataset_names, axis=0)

            if shuffle_patches_across_datasets == 1:
                print("Final shuffle across datasets enabled")
                x_train_shuffled_indices = np.random.permutation(np.arange(0, x_train.shape[0]))
                y_train_shuffled_indices = x_train_shuffled_indices
                x_test_shuffled_indices = np.random.permutation(np.arange(0, x_test.shape[0]))
                y_test_shuffled_indices = x_test_shuffled_indices

                x_train = x_train[x_train_shuffled_indices]
                y_train = y_train[y_train_shuffled_indices]
                train_patch_dataset_names = train_patch_dataset_names[x_train_shuffled_indices]

                x_test = x_test[x_test_shuffled_indices]
                y_test = y_test[y_test_shuffled_indices]
                test_patch_dataset_names = test_patch_dataset_names[x_test_shuffled_indices]


            print("Setting aside " + self.cph_dataset_names[test_data_cph_index] + " and " + self.naive_dataset_names[test_data_naive_index] + " as totally unseen test data")

            file_name = filename_prefix + 'segregated_data_random_select_' + str(random_select) + '_totally_unseen_test_data_' + self.cph_dataset_names[test_data_cph_index] + "_and_" + self.naive_dataset_names[test_data_naive_index] + '_' + self.used_dim_reduction_technique + '_ver_1.npy'


        segregated_data_dict = {'x_train': x_train,
                                'y_train': y_train,
                                'x_test': x_test,
                                'y_test': y_test,
                                'train_patch_dataset_names': train_patch_dataset_names,
                                'test_patch_dataset_names': test_patch_dataset_names,
                                'random_select_status': random_select,
                                'training_percentage': training_percentage,
                                'used_image_patch_filename': saved_image_patch_data_filename,
                                'reject_percentage_used_in_image_patch': image_patch_data_store_dict_recovered['reject_percentage'],
                                'window_size_used_in_image_patch': image_patch_data_store_dict_recovered['window_size'],
                                'used_3d_datagrid_filename_for_image_patch': image_patch_data_store_dict_recovered['used_3d_datagrid_filename'],
                                'used_dim_reduction_technique': self.used_dim_reduction_technique,
                                'global_path_name': self.dim_reduced_object.dataloader_kp_object.global_path_name,
                                'used_repeatable_random_numbers': repeatable_random_values,
                                'num_dim_reduced_components': self.num_dim_reduced_components,
                                'num_datasets': self.num_datasets,
                                'totally_unseen_test_data_enabled': totally_unseen_test_data,
                                'totally_unseen_test_data_override_percentage': totally_unseen_test_data_override_percentage,
                                'cph_label': self.cph_label,
                                'naive_label': self.naive_label,
                                'label_order_array': self.class_labels,
                                'fix_patch_count_per_dataset_enabled': fix_patch_count_per_dataset,
                                'limiting_patch_count': limiting_patch_count,
                                'patch_shuffling_across_datasets_enabled': shuffle_patches_across_datasets
                                }

        if totally_unseen_test_data==1:
            segregated_data_dict['totally_unseen_test_data_datasets'] = [self.cph_dataset_names[test_data_cph_index], self.naive_dataset_names[test_data_naive_index]]

        folder_name = self.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/segregated_data/'
        if save_data == 1:
            for count in range(2, 20):
                dir_list = os.listdir(folder_name)
                if file_name in dir_list:
                    file_name = file_name[0:-5] + str(count) + '.npy'

                else:
                    # print("File exists. New run number: "+ str(count-1))
                    break


            np.save(folder_name+file_name, segregated_data_dict)

        self.segregated_data_dict = segregated_data_dict

        return segregated_data_dict

    def data_preformatting_pipeline(self, saved_dim_reduced_filename=None, data_preformatting_pipeline_parameter_dict = {'preformatting_pipeline_filename_prefix': '', 'save_data': 0, 'image_patch_reject_percentage': 25, 'image_patch_window_size': 15, 'image_patch_overlap': 0, 'segregate_data_training_percentage': 80, 'segregate_data_random_select': 1, 'segregate_data_repeatable_random_values': 1, 'segregate_data_totally_unseen_test_data': 1, 'segregate_data_shuffle_patches_across_datasets':1}):

        """
        @brief This will run all the steps required to generate the final preformatted data, ready to be sent into
        an svm algorithm. This will be similar to the result that would have been generated if one had sequentially run the
        following functions, feeding the output of one to the next.

            Example Usage:

                        data_preformatting_pipeline_parameter_dict = {'preformatting_pipeline_filename_prefix': 'test_',
                                              'save_data': 1,
                                              'image_patch_reject_percentage': 25,
                                              'image_patch_window_size': 15,
                                              'image_patch_overlap': 0,
                                              'segregate_data_training_percentage': 80,
                                              'segregate_data_random_select': 1,
                                              'segregate_data_repeatable_random_values': 1,
                                              'segregate_data_totally_unseen_test_data': 1,
                                              'segregate_data_totally_unseen_test_data_override_percentage': 0,
                                              'segregate_data_use_externally_provided_unseen_test_dataset': 0,
                                              'segregate_data_externally_provided_datasets_as_unseen_test_data': {'test_cph_dataset_index':3, 'test_naive_dataset_index':3},
                                              'segregate_data_fix_patch_count_per_dataset': 1,
                                              'segregate_data_shuffle_patches_across_datasets': 1}



                        data_preformatter_kp_object.data_preformatting_pipeline(saved_dim_reduced_filename=saved_dim_reduced_filename, data_preformatting_pipeline_parameter_dict=data_preformatting_pipeline_parameter_dict)

                        Or

                        data_preformatter_kp_object.data_preformatting_pipeline(data_preformatting_pipeline_parameter_dict=data_preformatting_pipeline_parameter_dict)



        @param saved_dim_reduced_filename THis is an optional parameter, which, if provided as a path, will cause this piece of
            code to use that saved dim_reduced object instead of the dim_reduced object that came in through the initialization
            of this class.

        @param data_preformatting_pipeline_parameter_dict:
                image_patch_reject_percentage: Defines the percentage of the background area of an nmf ion image which above which,
                if included in an image patch, the image patch gets rejected.

                image_patch_window_size: Defines the window size, i.e, the size of a patch. Patches are square in shape in the spatial axes.

                image_patch_overlap: This defines if image patches can overlap with each other.

                segregate_data_training_percentage Defines the percentage of data that needs to be set as training data. Accordingly, the testing data will
                be the remaining data.

                segregate_data_random_select This parameter, if set to 1, will shuffle the dataset each time when generating the training vs testing data.
                If this is set to 0, then, everytime, we will get the same sets of training and testing data.

                save_data: Will save data if this is enabled

                preformatting_pipeline_filename_prefix: Will be only necessary if save data is enabled.

                segregate_data_repeatable_random_values: Set this to 1 if you want the data to be split randomly, but need those random values to be reproducible between multiple function calls.

                segregate_data_totally_unseen_test_data: Set this to 1 if I want to keep a cph and naive dataset entirely aside to do testing

                segregate_data_totally_unseen_test_data_override_percentage: This can accept a percentage. If a non zero value is given for this, it will be interpretted as a percentage, a percentage of the totally unseen test data which will actually be forcibly included in the train data, hence making part of the totally unseen test data to be actually seen.

                segregate_data_use_externally_provided_unseen_test_dataset: If this is set to 1, we are able to externally provide the index of the datasets that we need to totally set aside for testing purpose.

                segregate_data_externally_provided_datasets_as_unseen_test_data: A dictionary of the following format that will only be used if the variable  "use_externally_provided_unseen_test_dataset" is set to 1. {'test_cph_dataset_index':3, 'test_naive_dataset_index':3}

                segregate_data_fix_patch_count_per_dataset: Set this to 1 to ensure that there will be the same number of patches from each dataset.

                segregate_data_shuffle_patches_across_datasets: THis addresses the problem of gridsearchcv algorithm not shuffling the patches randomly.

        @return: Returns a segregated_data_dict variable.
        """

        image_patch_reject_percentage = data_preformatting_pipeline_parameter_dict['image_patch_reject_percentage']
        image_patch_window_size = data_preformatting_pipeline_parameter_dict['image_patch_window_size']
        image_patch_overlap = data_preformatting_pipeline_parameter_dict['image_patch_overlap']
        segregate_data_training_percentage = data_preformatting_pipeline_parameter_dict['segregate_data_training_percentage']
        segregate_data_random_select = data_preformatting_pipeline_parameter_dict['segregate_data_random_select']
        save_data = data_preformatting_pipeline_parameter_dict['save_data']
        preformatting_pipeline_filename_prefix = data_preformatting_pipeline_parameter_dict['preformatting_pipeline_filename_prefix']
        segregate_data_repeatable_random_values = data_preformatting_pipeline_parameter_dict['segregate_data_repeatable_random_values']
        segregate_data_totally_unseen_test_data = data_preformatting_pipeline_parameter_dict['segregate_data_totally_unseen_test_data']
        segregate_data_fix_patch_count_per_dataset = data_preformatting_pipeline_parameter_dict['segregate_data_fix_patch_count_per_dataset']
        segregate_data_totally_unseen_test_data_override_percentage = data_preformatting_pipeline_parameter_dict['segregate_data_totally_unseen_test_data_override_percentage']
        segregate_data_use_externally_provided_unseen_test_dataset = data_preformatting_pipeline_parameter_dict['segregate_data_use_externally_provided_unseen_test_dataset']
        segregate_data_externally_provided_datasets_as_unseen_test_data = data_preformatting_pipeline_parameter_dict['segregate_data_externally_provided_datasets_as_unseen_test_data']
        segregate_data_shuffle_patches_across_datasets = data_preformatting_pipeline_parameter_dict['segregate_data_shuffle_patches_across_datasets']

        datagrid_store_dict = self.create_3d_array_from_dim_reduced_data(saved_dim_reduced_filename=saved_dim_reduced_filename, save_data=save_data, filename_prefix=preformatting_pipeline_filename_prefix)
        image_patches_unrolled_dict = self.image_patches_unrolled(reject_percentage=image_patch_reject_percentage, window_size=image_patch_window_size, overlap=image_patch_overlap, save_data=save_data, filename_prefix=preformatting_pipeline_filename_prefix)
        segregated_training_testing_dict = self.separate_training_testing_data(training_percentage=segregate_data_training_percentage, random_select=segregate_data_random_select, save_data=save_data, filename_prefix=preformatting_pipeline_filename_prefix, repeatable_random_values=segregate_data_repeatable_random_values, totally_unseen_test_data=segregate_data_totally_unseen_test_data, fix_patch_count_per_dataset=segregate_data_fix_patch_count_per_dataset, totally_unseen_test_data_override_percentage=segregate_data_totally_unseen_test_data_override_percentage, use_externally_provided_unseen_test_dataset=segregate_data_use_externally_provided_unseen_test_dataset, externally_provided_datasets_as_unseen_test_data=segregate_data_externally_provided_datasets_as_unseen_test_data, shuffle_patches_across_datasets=segregate_data_shuffle_patches_across_datasets)

        return segregated_training_testing_dict, image_patches_unrolled_dict, datagrid_store_dict

