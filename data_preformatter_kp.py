import numpy as np
import os

class data_preformatter_kp:
    """
    @brief This class does preformatting of an NMF, PCA, ICA object etc to eventually enable SVM computation.
    """

    def __init__(self, dim_reduced_object):

        """
              @brief Initializes the data_preformatter_kp object. This class is designed to perform data preformatting of
               NMF, PCA, ICA objects to enable SVM computation.

            Example usage:
                from data_preformatter_kp import data_preformatter_kp
                data_preformatter_object = data_preformatter_kp(nmf_kp_object)

        @param dim_reduced_object This is an output of the dimensionality reduction class such as nmf_kp.

        """

        self.dim_reduced_object = dim_reduced_object
        self.used_dim_reduction_technique = self.dim_reduced_object.used_dim_reduction_technique

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

        for i in range(len(self.dim_reduced_object.dataloader_kp_object.combined_data_object)):
            num_rows = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].rows
            num_cols = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].cols
            coordinate_array = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].coordinates
            pix_count = len(self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].coordinates)
            individual_dim_reduced_data = dim_reduced_dataset[pix_count_previous: pix_count_previous + pix_count, :]
            pix_count_previous = pix_count_previous + pix_count
            num_dim_reduced_components = individual_dim_reduced_data.shape[1]

            if self.used_dim_reduction_technique == 'nmf':
                datagrid = np.ones([num_dim_reduced_components, num_rows, num_cols]) * -1
            elif self.used_dim_reduction_technique == 'pca':
                datagrid = np.ones([num_dim_reduced_components, num_rows, num_cols]) * 1000

            for temp1 in range(num_dim_reduced_components):
                for temp2 in range(pix_count):
                    this_col = coordinate_array[temp2][0]
                    this_row = coordinate_array[temp2][1]
                    datagrid[temp1, this_row - 1, this_col - 1] = individual_dim_reduced_data[temp2, temp1]

            datagrid_store.append(datagrid)

        datagrid_store_dict = {'datagrid_store': datagrid_store, 'used_dim_reduced_file': saved_dim_reduced_filename,
                               'used_dim_reduction_technique': self.used_dim_reduction_technique, 'global_path_name': self.dim_reduced_object.dataloader_kp_object.global_path_name}

        folder_name = self.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/output_3d_datagrids/'
        file_name = filename_prefix + 'output_3d_datagrid_store_' + self.used_dim_reduction_technique + '_' + str(
            num_dim_reduced_components) + '_dim_reduced_components' + '_ver_1.npy'

        if save_data == 1:
            for count in range(2, 20):
                dir_list = os.listdir(folder_name)
                if file_name in dir_list:
                    file_name = file_name[0:-5] + str(count) + '.npy'

                else:
                    #             print("File exists. New run number: "+ str(count-1))
                    break

            np.save(folder_name+file_name, datagrid_store_dict, 'dtype=object')

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
            datagrid_store_dict_recovered = self.datagrid_store_dict.copy()

        datagrid_store_recovered = datagrid_store_dict_recovered['datagrid_store'].copy()
        num_dim_reduced_components = datagrid_store_recovered[0].shape[0]
        used_dim_reduction_technique = self.used_dim_reduction_technique

        labelled_data_store_flattened = []
        labelled_data_store = []

        if overlap == 0:
            #         print("Overlapping is NOT allowed")
            for i in range(len(self.dim_reduced_object.dataloader_kp_object.combined_data_object)):
                row_start = 0
                col_start = 0
                num_rows = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].rows
                num_cols = self.dim_reduced_object.dataloader_kp_object.combined_data_object[i].cols
                this_image = datagrid_store_recovered[i]

                labelled_individual_dataset_patches_flattened = np.empty(
                    [0, num_dim_reduced_components * window_size * window_size])
                labelled_individual_dataset_patches = np.empty(
                    [0, num_dim_reduced_components, window_size, window_size])

                while (row_start <= num_rows - window_size - 1):
                    while (col_start <= num_cols - window_size - 1):

                        temp1 = this_image[:, row_start:row_start + window_size, col_start:col_start + window_size].copy()
                        labelled_data_patch_flattened = np.array([temp1.flatten()])
                        labelled_data_patch = temp1
                        col_start = col_start + window_size

                        ### Remove image patches that have too much background
                        if (used_dim_reduction_technique == 'nmf'):
                            if (100 * np.count_nonzero(labelled_data_patch_flattened == -1) / labelled_data_patch_flattened.shape[1]) > reject_percentage:
                                continue
                            else:
                                ### For image patches that are acceptable, replace the background -1 values with 0.
                                labelled_data_patch_flattened[labelled_data_patch_flattened == -1] = 0
                                labelled_data_patch[labelled_data_patch == -1] = 0
                        elif (used_dim_reduction_technique == 'pca'):
                            if (100 * np.count_nonzero(labelled_data_patch_flattened == 1000) / labelled_data_patch_flattened.shape[1]) > reject_percentage:
                                continue
                            else:
                                ### For image patches that are acceptable, replace the background 1000 values with 0.
                                labelled_data_patch_flattened[labelled_data_patch_flattened == 1000] = 0
                                labelled_data_patch[labelled_data_patch == 1000] = 0

                        labelled_individual_dataset_patches_flattened = np.append(labelled_individual_dataset_patches_flattened, labelled_data_patch_flattened, axis=0)
                        labelled_individual_dataset_patches = np.append(labelled_individual_dataset_patches, [labelled_data_patch], axis=0)
                    col_start = 0
                    row_start = row_start + window_size

                labelled_data_store_flattened.append(labelled_individual_dataset_patches_flattened)
                labelled_data_store.append(labelled_individual_dataset_patches)


        else:

            print("Overlapping is allowed")

        ### Adding labels
        labels_array = [0, 0, 0, 0, 1, 1, 1, 1]
        combined_labelled_flattened_data = np.empty([0, labelled_data_store_flattened[0].shape[1] + 1])
        for count, i in enumerate(labelled_data_store_flattened):
            temp5 = np.append(i, np.ones([i.shape[0], 1]) * labels_array[count], axis=1)
            combined_labelled_flattened_data = np.append(combined_labelled_flattened_data, temp5, axis=0)

        image_patch_data_store_dict = {'combined_labelled_flattened_data': combined_labelled_flattened_data,
                                       'labelled_data_store_flattened': labelled_data_store_flattened,
                                       'labelled_data_store': labelled_data_store,
                                       'reject_percentage': reject_percentage, 'window_size': window_size,
                                       'used_3d_datagrid_filename': saved_3d_datagrid_filename,
                                       'used_dim_reduction_technique': used_dim_reduction_technique,
                                       'global_path_name': self.dim_reduced_object.dataloader_kp_object.global_path_name}


        folder_name = self.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/image_patch/'
        file_name = filename_prefix + 'image_patch_data_' + used_dim_reduction_technique + '_' + str(num_dim_reduced_components) + '_dim_reduced_components_window_' + str(window_size) + '_rejection_' + str(reject_percentage) + '_ver_1.npy'

        if save_data == 1:
            for count in range(2, 20):
                dir_list = os.listdir(folder_name)
                if file_name in dir_list:
                    file_name = file_name[0:-5] + str(count) + '.npy'

                else:
                    #             print("File exists. New run number: "+ str(count-1))
                    break

            np.save(folder_name+file_name, image_patch_data_store_dict, 'dtype=object')

        self.image_patch_data_store_dict = image_patch_data_store_dict

        return image_patch_data_store_dict

    def separate_training_testing_data(self, saved_image_patch_data_filename=None, training_percentage=80, random_select=1, save_data=0, filename_prefix=''):

        """
        @brief This seperates an image patch dataset into training and testing sets.

            # saved_image_patch_data_filename='/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/image_patch/version_2_image_patch_data_20_nmf_components_window_30_rejection_25.npy'
            # data_segration_training_percentage=80
            # data_segregation_random_select=1
            # data_segregation_filename_prefix=''
            # save_data=0

            # segregated_data_dict=data_preformatter_kp_object.separate_training_testing_data(saved_image_patch_data_filename=saved_image_patch_data_filename, training_percentage=data_segration_training_percentage, random_select=data_segregation_random_select, save_data = save_data, filename_prefix=data_segregation_filename_prefix)

            # OR

            # segregated_data_dict=data_preformatter_kp_object.separate_training_testing_data(training_percentage=data_segration_training_percentage, random_select=data_segregation_random_select, save_data = save_data, filename_prefix=data_segregation_filename_prefix)

        @param saved_image_patch_data_filename: THis is an optional parameter, which, if provided as a path, will cause this piece of
            code to use that saved dim_3d_datagrid dict instead of the self.datagrid_store_dict object that came in through the class.
        @param training_percentage Defines the percentage of data that needs to be set as training data. Accordingly, the testing data will
            be the remaining data.
        @param random_select This parameter, if set to 1, will shuffle the dataset each time when generating the training vs testing data.
            If this is set to 0, then, everytime, we will get the same sets of training and testing data.
        @param save_data: Will save data if this is enabled
        @param filename_prefix: Will be only necessary if save data is enabled.
        @return: Returns a dictionary containing an nmf dataset divided into patches defined by the above parameters.

        """

        if (saved_image_patch_data_filename is not None) and (saved_image_patch_data_filename != ''):
            image_patch_data_store_dict_recovered = np.load(saved_image_patch_data_filename, allow_pickle=True)[()]
        else:
            image_patch_data_store_dict_recovered = self.image_patch_data_store_dict

        x = image_patch_data_store_dict_recovered['combined_labelled_flattened_data'][:, 0:-1]
        y = image_patch_data_store_dict_recovered['combined_labelled_flattened_data'][:, [-1]]

        window_size = image_patch_data_store_dict_recovered['window_size']
        reject_percentage = image_patch_data_store_dict_recovered['reject_percentage']
        saved_3d_datagrid_filename = image_patch_data_store_dict_recovered['used_3d_datagrid_filename']
        used_dim_reduction_technique = self.used_dim_reduction_technique

        if random_select == 1:
            class_0_x = x[np.squeeze(y == 0), :]
            class_0_y = y[np.squeeze(y == 0), :]

            class_1_x = x[np.squeeze(y == 1), :]
            class_1_y = y[np.squeeze(y == 1), :]

            rand_pattern_class_0 = np.random.permutation(np.arange(0, class_0_x.shape[0]))
            class_0_x_rand = class_0_x[rand_pattern_class_0]
            class_0_y_rand = class_0_y[rand_pattern_class_0]

            rand_pattern_class_1 = np.random.permutation(np.arange(0, class_1_x.shape[0]))
            class_1_x_rand = class_1_x[rand_pattern_class_1]
            class_1_y_rand = class_1_y[rand_pattern_class_1]

            class_0_divide_index = int(training_percentage / 100 * class_0_x_rand.shape[0])
            class_1_divide_index = int(training_percentage / 100 * class_1_x_rand.shape[0])

            x_train = np.append(class_0_x_rand[0:class_0_divide_index, :], class_1_x_rand[0:class_1_divide_index, :], axis=0)
            y_train = np.append(class_0_y_rand[0:class_0_divide_index, :], class_1_y_rand[0:class_1_divide_index, :], axis=0)

            x_test = np.append(class_0_x_rand[class_0_divide_index:-1, :], class_1_x_rand[class_1_divide_index:-1, :], axis=0)
            y_test = np.append(class_0_y_rand[class_0_divide_index:-1, :], class_1_y_rand[class_1_divide_index:-1, :], axis=0)

        else:
            print("Selecting determinately")

        segregated_data_dict = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
                                'random_select_status': random_select, 'training_percentage': training_percentage,
                                'used_image_patch_filename': saved_image_patch_data_filename,
                                'reject_percentage_used_in_image_patch': reject_percentage,
                                'window_size_used_in_image_patch': window_size,
                                'used_3d_datagrid_filename_for_image_patch': saved_3d_datagrid_filename,
                                'used_dim_reduction_technique': used_dim_reduction_technique,
                                'global_path_name': self.dim_reduced_object.dataloader_kp_object.global_path_name}
        if (random_select == 1):
            segregated_data_dict['rand_pattern_class_0'] = rand_pattern_class_0
            segregated_data_dict['rand_pattern_class_1'] = rand_pattern_class_1


        folder_name = self.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/segregated_data/'
        file_name = filename_prefix + 'segregated_data_trianing_percentage_' + str(training_percentage) + '_random_select_' + str(random_select) + '_' + used_dim_reduction_technique + '_' + '_ver_1.npy'

        if save_data == 1:
            for count in range(2, 20):
                dir_list = os.listdir(folder_name)
                if file_name in dir_list:
                    file_name = file_name[0:-5] + str(count) + '.npy'

                else:
                    #             print("File exists. New run number: "+ str(count-1))
                    break


            np.save(folder_name+file_name, segregated_data_dict, 'dtype=object')

        self.segregated_data_dict = segregated_data_dict
        return segregated_data_dict

    def data_preformatting_pipeline(self, saved_dim_reduced_filename=None, image_patch_reject_percentage=25, image_patch_window_size = 30, image_patch_overlap=0, segregate_data_training_percentage = 80, segregate_data_random_select = 1, save_data=0, preformatting_pipeline_filename_prefix=''):

        """
        @brief This will run all the steps required to generate the final preformatted data, ready to be sent into
        an svm algorithm. This will be similar to the result that would have been generated if one had sequentially run the
        following functions, feeding the output of one to the next.

            Example Usage:

                        saved_dim_reduced_filename = None
                        image_patch_reject_percentage=25
                        image_patch_window_size = 30
                        image_patch_overlap=0
                        segregate_data_training_percentage = 80
                        segregate_data_random_select = 1
                        save_data=0
                        preformatting_pipeline_filename_prefix='test_data_preformatting_pipeline'

                        data_preformatter_kp_object.data_preformatting_pipeline(saved_dim_reduced_filename=saved_dim_reduced_filename, image_patch_reject_percentage=image_patch_reject_percentage, image_patch_window_size = image_patch_window_size, image_patch_overlap=image_patch_overlap, segregate_data_training_percentage = segregate_data_training_percentage, segregate_data_random_select = segregate_data_random_select, save_data=save_data, preformatting_pipeline_filename_prefix=preformatting_pipeline_filename_prefix)

                        Or

                        data_preformatter_kp_object.data_preformatting_pipeline(image_patch_reject_percentage=image_patch_reject_percentage, image_patch_window_size = image_patch_window_size, image_patch_overlap=image_patch_overlap, segregate_data_training_percentage = segregate_data_training_percentage, segregate_data_random_select = segregate_data_random_select, save_data=save_data, preformatting_pipeline_filename_prefix=preformatting_pipeline_filename_prefix)



        @param saved_dim_reduced_filename THis is an optional parameter, which, if provided as a path, will cause this piece of
            code to use that saved dim_reduced object instead of the dim_reduced object that came in through the initialization
            of this class.

        @param image_patch_reject_percentage: Defines the percentage of the background area of an nmf ion image which above which,
            if included in an image patch, the image patch gets rejected.
        @param image_patch_window_size: Defines the window size, i.e, the size of a patch. Patches are square in shape in the spatial axes.
        @param image_patch_overlap: This defines if image patches can overlap with each other.

        @param segregate_data_training_percentage Defines the percentage of data that needs to be set as training data. Accordingly, the testing data will
            be the remaining data.
        @param segregate_data_random_select This parameter, if set to 1, will shuffle the dataset each time when generating the training vs testing data.
            If this is set to 0, then, everytime, we will get the same sets of training and testing data.

        @param save_data: Will save data if this is enabled
        @param preformatting_pipeline_filename_prefix: Will be only necessary if save data is enabled.


        @return: Returns a segregated_data_dict variable.
        """

        datagrid_store_dict = self.create_3d_array_from_dim_reduced_data(saved_dim_reduced_filename=saved_dim_reduced_filename, save_data=save_data, filename_prefix=preformatting_pipeline_filename_prefix)
        image_patches_unrolled_dict = self.image_patches_unrolled(reject_percentage=image_patch_reject_percentage, window_size=image_patch_window_size, overlap=image_patch_overlap, save_data=save_data, filename_prefix=preformatting_pipeline_filename_prefix)
        segregated_training_testing_dict = self.separate_training_testing_data(training_percentage=segregate_data_training_percentage, random_select=segregate_data_random_select, save_data=save_data, filename_prefix=preformatting_pipeline_filename_prefix)

        return segregated_training_testing_dict, image_patches_unrolled_dict, datagrid_store_dict

