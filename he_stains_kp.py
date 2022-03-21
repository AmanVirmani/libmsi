import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.metrics as sk_metrics
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.preprocessing import minmax_scale
from skimage import color
from skimage.transform import rescale, resize, downscale_local_mean
import pickle
from scipy.signal import find_peaks

from sklearnex import patch_sklearn
patch_sklearn()


class he_stains_kp:

    def __init__(self, data_preformatter_kp_object):

        """
        @brief Initializes the he_stains_kp class. If we do not specify a saved aligned H&E stained image dictionary,a data preformatter class object must be presented.
            With this, the image alignment will happen automatically. If a presaved aligned he dict pathname is given, it will be loaded, and no new alignment will be done.

            usage example:
                            he_stains_kp_object = he_stains_kp()
                            he_stains_kp_object = he_stains_kp(data_preformatter_kp_object)

        @param data_preformatter_kp_object: Necessary if we are doing the alignment from the start as opposed to loading a presaved file
        """

        self.data_preformatter_kp_object = data_preformatter_kp_object
        self.num_dim_reduced_components = self.data_preformatter_kp_object.dim_reduced_object.num_dim_reduced_components
        self.num_datasets = len(self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.combined_data_object)
        self.datagrid_store_recovered = self.data_preformatter_kp_object.datagrid_store_dict['datagrid_store']
        self.he_filename_array = self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.he_filename_array
        self.dataset_order = self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.dataset_order
        self.global_path_name =  self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.global_path_name
        self.used_dim_reduction_technique = self.data_preformatter_kp_object.used_dim_reduction_technique

        self.channel_extracted_he_store, self.channel_extracted_he_store_flattened, self.he_store_3d = self.load_he_stained_images()
        self.normalized_datagrid_store = self.normalize_3d_datagrid_store_to_given_range(0, 255)
        self.false_colored_normalized_datagrid_store = self.create_false_colored_normalized_dim_reduced_datagrid_store()
        self.optimal_warp_matrix_store = None
        self.temp_now_aligning_segment = ''  # A global flag used to trace if a specific h&e segment is being aligned

    def normalize_3d_datagrid_store_to_given_range(self, range_min_value, range_max_value):

        """
        @brief Normalize the input array (containing nmf or pca images) to be between range_min_value (ex: 0) and range_min_value (Ex: 255) so that they can be compared against the H&E stained images.
        Also removes -1s in NMF backgrounds and 1000s in PCA backgrounds that I artificially introduced durinng the creation of datagrid_store_dicts
            Usage example:
                            array_to_be_normalized = data_preformatter_kp_object.datagrid_store_dict['datagrid_store']
                            range_min_value = 0
                            range_max_value = 255
                            normalized_datagrid_store = he_stains_kp_object.normalize_3d_datagrid_store_to_given_range(array_to_be_normalized, range_min_value, range_max_value)


        @param range_max_value: Max value that we need the array data to have after normalization (ex: 255)
        @param range_min_value: Min value that we need the array data to have after normalization (ex: 0)
        """

        normalized_datagrid_store = []
        for i in range(self.num_datasets):
            this_dataset = self.datagrid_store_recovered[i].copy()
            for_nmf = this_dataset == -1
            for_pca = this_dataset == 1000
            if self.used_dim_reduction_technique == 'nmf':
                this_dataset[for_nmf | for_pca] = 0
            elif self.used_dim_reduction_technique == 'pca':
                for component_k in range(this_dataset.shape[0]):
                    this_component = this_dataset[component_k].copy()
                    this_component[this_component == 1000] = np.min(this_dataset[component_k])
                    this_dataset[component_k] = this_component
            # The above 4 lines removed the -1s or 1000s I introduced to the background pixels of nmf and pca respectively

            each_dim_reduced_component_unrolled = np.reshape(this_dataset, (self.num_dim_reduced_components, -1))
            max_per_dim_reduced_component = np.tile(np.reshape(np.max(each_dim_reduced_component_unrolled, axis=1), (-1, 1, 1)), (1, this_dataset.shape[1], this_dataset.shape[2]))
            min_per_dim_reduced_component = np.tile(np.reshape(np.min(each_dim_reduced_component_unrolled, axis=1), (-1, 1, 1)), (1, this_dataset.shape[1], this_dataset.shape[2]))
            this_dataset_normalized = 0 + np.multiply((np.divide((this_dataset - (min_per_dim_reduced_component)), ((max_per_dim_reduced_component) - (min_per_dim_reduced_component)))), (range_max_value - range_min_value))
            this_dataset_normalized = np.uint8(this_dataset_normalized)

            normalized_datagrid_store.append(this_dataset_normalized)

        return normalized_datagrid_store

    def load_he_stained_images(self):

        """
        @brief loads the h&e stained images into numpy arrays.
            Returns an array containing the loaded BGR color images in 'he_store_3d' variable
            Returns an array containing the the extracted Blue, Green, and Red channels (In that order) in 'channel_extracted_he_store' variable
            Returns an array containing the flattened version of the above (each channel is flattened) 'channel_extracted_he_store_flattened' variable
            Example usage:

                        channel_extracted_he_store, channel_extracted_he_store_flattened, he_store_3d = he_stains_kp_object.load_he_stained_images()
        @return channel_extracted_he_store, channel_extracted_he_store_flattened, he_store_3d:
        """

        he_file_names = self.he_filename_array
        combined_data_object = self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.combined_data_object

        he_store_3d = []
        channel_extracted_he_store_flattened = []
        channel_extracted_he_store = []

        for i in range(self.num_datasets):
            num_rows = combined_data_object[i].rows
            num_cols = combined_data_object[i].cols

            this_he_image = cv2.imread(he_file_names[0][i])
            this_he_image = cv2.resize(this_he_image, (num_cols, num_rows), interpolation=cv2.INTER_LINEAR)

            extracted_BGR_he_channels = np.array(cv2.split(this_he_image))
            extracted_BGR_he_channels_flattened = extracted_BGR_he_channels.reshape(3,-1)
            ## In the above line, I reshape each of the images corresponding to the 3 channels: Blue, Green, Red, such that I have a 3 x num_pixels array

            channel_extracted_he_store.append(extracted_BGR_he_channels)
            channel_extracted_he_store_flattened.append(extracted_BGR_he_channels_flattened)
            he_store_3d.append(this_he_image)

        self.channel_extracted_he_store = channel_extracted_he_store
        self.channel_extracted_he_store_flattened = channel_extracted_he_store_flattened
        self.he_store_3d = he_store_3d

        return self.channel_extracted_he_store, self.channel_extracted_he_store_flattened, self.he_store_3d

    def get_image_gradient(self, image_input, plot_gradient_of_image=0):

        """
        @brief Calculate gradient of an image (Edge detection). Used to get the edges of an image so that a homography transform algorithm will work better.

            Usage example:
                        image_input=grayscale_he_image
                        plot_gradient_of_image=0
                        image_gradient = he_stains_kp_object.get_image_gradient(image_input, plot_gradient_of_image=plot_gradient_of_image)


        @param plot_gradient_of_image: If set to 1, a the gradient image will be plotted.
        @param image_input: The image whose gradient must be calculated. The image must be grayscale.
        @return grad: A 2D variable containing the gradient at each pixel of the input image.
        """

        # Calculate the x and y gradients using Sobel operator
        grad_x = cv2.Sobel(image_input, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_input, cv2.CV_32F, 0, 1, ksize=3)

        # Combine the two gradients
        grad_of_input_image = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

        if plot_gradient_of_image == 1:

            plt.imshow(grad_of_input_image)
            plt.pause(5)

        return grad_of_input_image

    def calculate_ecc(self, reference_image, warped_image):
        """
        @brief Takes in two images and spits out the ECC between them
        @param reference_image: The first image
        @param warped_image: The second image
        @return ecc_value: The ECC between the two images
        """

        i_r = np.float32(reference_image.flatten())
        i_w = np.float32(warped_image.flatten())
        i_r_mean_centered = i_r - np.mean(i_r)
        i_w_mean_centered = i_w - np.mean(i_w)
        norm_i_r_mean_centered = np.linalg.norm(i_r_mean_centered)
        norm_i_w_mean_centered = np.linalg.norm(i_w_mean_centered)

        ecc_value = np.dot((i_r_mean_centered/norm_i_r_mean_centered), (i_w_mean_centered/norm_i_w_mean_centered))

        return ecc_value

    def align_all_combinations_of_dim_reduced_images_with_he_stained_images(self, gradient_alignment=1, saved_aligned_he_dict_filename=None, custom_he_store=None, custom_warp_matrix_store=None, save_data=0, filename_prefix=''):

        """
        @brief This function does a homography transformation on the h&e stained images to spatially match them with the nmf/pca images.
            It calculates the correlation coefficient after aligning an H&E stained image of a dataset  with each and every nmf/pca component. It does NOT  determine which component gives the best homography
            This function calculates warp transformations to transform the h&e image  into the nmf image based on maximizing enhanced corr. coeff
            Maximum enhanced correlation coefficient is calculated for a given h&e stained image against all of the nmf/pca images corresponding to that dataset.
            Usage example:
                            ### Align h&e stained images with all nmf components separately
                            ## Use the internal he_store_3d for the alignments
                            # save_data = 1
                            # filename_prefix = 'modified_'
                            # aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(save_data=save_data, filename_prefix=filename_prefix)

                            ## OR
                            ## Use a custom he store for the alignment
                            # thresholded_he_store, false_colored_BGR_thresholded_he_store = he_stains_kp_object.threshold_he_images_for_alignment()
                            # save_data = 1
                            # use_gradient = 1
                            # filename_prefix = 'custom_warp_thresholded_'
                            # custom_he_store = false_colored_BGR_thresholded_he_store
                            # aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_he_store=custom_he_store, save_data=save_data, filename_prefix=filename_prefix, use_gradient=use_gradient)

                            ## OR
                            ## Use a custom warp matrix to generate the most optimal alignment
                            # saved_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/without_gradient_align_using_segmented_he_segment_1_aligned_he_stained_dict.npy"
                            # optimally_warped_component_selection_array = [2, 2, 2, 2, 2, 2, 2, 2]  # Example, this would be warp matrix corresponding to nmf2 (the muscular lining)
                            # custom_warp_matrix_store = he_stains_kp_object.create_custom_warp_matrix_store(saved_all_combinations_aligned_he_dict_filename, optimally_warped_component_selection_array=optimally_warped_component_selection_array)
                            # save_data = 1
                            # filename_prefix = 'optimal_alignment_from_segment_1_muscular_lining_'
                            # aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_warp_matrix_store=custom_warp_matrix_store, save_data=save_data, filename_prefix=filename_prefix)

                            ## OR
                            ## If a pre-saved file exists, use that instead
                            # saved_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/thresholded_aligned_he_stained_dict.npy"
                            # aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(saved_aligned_he_dict_filename = saved_aligned_he_dict_filename)

        @param save_data: Save data if enabled
        @param gradient_alignment: If set to 1, align the gradients (edge detected version) of the images. If set to 0, align the raw images.
        @param custom_warp_matrix_store: If given, the homography will NOT be calculated. Instead, alignment will be done based on this warp matrix.
        @param custom_he_store: Use this array instead of the he_store_3d to do the alignment.
        @param filename_prefix: Necessary if save_data is set to 1.
        @param saved_aligned_he_dict_filename: Load a presaved dictionary containing all combinations of nmf/pca components aligned with the H&E stained images aligned set of images
        @return all_combinations_aligned_he_dict: A dictionary containing the aligned h&e stained images, the normalized dim reduced data used for the alignment, the original h&e stained images used for the alignment,
            warp matrices used for the alignments, enhanced correlation coefficients of the alignments.
        """

        if (saved_aligned_he_dict_filename is not None) and (saved_aligned_he_dict_filename != ''):
            temp_aligned_he_dict = np.load(saved_aligned_he_dict_filename, allow_pickle=True)[()]
            # if ('is_this_the_optimal_alignment' in temp_aligned_he_dict.keys()) and (temp_aligned_he_dict['is_this_the_optimal_alignment'] == 1):
            #     self.optimally_aligned_he_dict = temp_aligned_he_dict
            #     return self.optimally_aligned_he_dict
            # else:
            self.all_combinations_aligned_he_dict = temp_aligned_he_dict
            return self.all_combinations_aligned_he_dict

        if custom_warp_matrix_store is None:
            did_i_use_custom_warp_matrix_store = 0
            warp_matrix_store = []
        else:
            did_i_use_custom_warp_matrix_store = 1
            warp_matrix_store = custom_warp_matrix_store

        if custom_he_store is None:
            self.used_all_combinations_he_store_3d_for_alignment = self.he_store_3d
            did_i_use_custom_he_store_for_alignment = 0
        else:
            self.used_all_combinations_he_store_3d_for_alignment = custom_he_store
            did_i_use_custom_he_store_for_alignment = 1

        channel_extracted_aligned_he_store = []
        channel_extracted_aligned_he_store_flattened = []
        aligned_all_combinations_BGR_he_image_store = []
        enhanced_correlation_coefficient_store = []
        # custom_enhanced_correlation_coefficient_store = []
        ssim_store = []
        nrmse_store = []

        for i in range(self.num_datasets):
            per_dataset_aligned_BGR_he_image_store = []
            per_dataset_channel_extracted_aligned_he_store = []
            per_dataset_channel_extracted_aligned_he_store_flattened = []
            per_dataset_warp_matrix_store = []
            per_dataset_enhanced_correlation_coefficient_store = []
            per_dataset_nrmse_store = []
            per_dataset_ssim_store = []
            # per_dataset_custom_enhanced_correlation_coefficient_store = []

            for j in range(self.num_dim_reduced_components):

                if self.temp_now_aligning_segment != '':
                    print("Aligning h&e segment " + str(self.temp_now_aligning_segment) + " with dataset " + str(i) + ", component " + str(j))
                else:
                    print("Aligning dataset " + str(i) + ", component " + str(j))

                reference_image_for_alignment = self.normalized_datagrid_store[i][j]
                reference_image_for_alignment_gray = reference_image_for_alignment.copy()  # Since the normalized (between 0 and 255) images mentioned above only have one channel, they are already grayscale images. Therefore, simply assign to the variable 'reference_image_for_alignment_gray'

                image_to_be_aligned = self.used_all_combinations_he_store_3d_for_alignment[i]
                image_to_be_aligned_blue_component, image_to_be_aligned_green_component, image_to_be_aligned_red_component = cv2.split(image_to_be_aligned)
                image_to_be_aligned_gray = cv2.cvtColor(image_to_be_aligned, cv2.COLOR_BGR2GRAY)

                height = reference_image_for_alignment.shape[0]
                width = reference_image_for_alignment.shape[1]

                warp_mode = cv2.MOTION_HOMOGRAPHY
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-12)
                # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1e-1)  # For testing purposes

                if gradient_alignment == 1:
                    plot_gradient_of_image = 0
                    reference_image_gradient = self.get_image_gradient(reference_image_for_alignment_gray, plot_gradient_of_image=plot_gradient_of_image)
                    processed_gray_reference_image_for_alignment = reference_image_gradient
                    image_to_be_aligned_gradient = self.get_image_gradient(image_to_be_aligned_gray, plot_gradient_of_image=plot_gradient_of_image)
                    processed_gray_image_to_be_aligned = image_to_be_aligned_gradient
                else:
                    processed_gray_reference_image_for_alignment = reference_image_for_alignment_gray
                    processed_gray_image_to_be_aligned = image_to_be_aligned_gray

                if did_i_use_custom_warp_matrix_store == 0:
                    warp_matrix = np.eye(3, 3, dtype=np.float32)
                    try:
                        (enhanced_correlation_coefficient, warp_matrix) = cv2.findTransformECC(np.float32(processed_gray_reference_image_for_alignment), np.float32(processed_gray_image_to_be_aligned), warp_matrix, warp_mode, criteria, None, 1)  # 'enhanced_correlation_coefficient' variable is the 'enhanced correlation coefficient' that was maximized during the transform
                    except:
                        enhanced_correlation_coefficient = 0
                        warp_matrix = np.eye(3, 3, dtype=np.float32)
                        print("error at dataset " + str(i) + ", component " + str(j))
                else:
                    warp_matrix = custom_warp_matrix_store[i][j]
                    processed_gray_image_aligned = cv2.warpPerspective(processed_gray_image_to_be_aligned, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    enhanced_correlation_coefficient = cv2.computeECC(np.float32(processed_gray_reference_image_for_alignment), np.float32(processed_gray_image_aligned))

                ############################
                # processed_gray_image_aligned = cv2.warpPerspective(processed_gray_image_to_be_aligned, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                # custom_enhanced_correlation_coefficient = cv2.computeECC(np.float32(processed_gray_reference_image_for_alignment), np.float32(processed_gray_image_aligned))
                # per_dataset_custom_enhanced_correlation_coefficient_store.append(custom_enhanced_correlation_coefficient)
                ############################

                aligned_image_blue_component = cv2.warpPerspective(image_to_be_aligned_blue_component, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_image_green_component = cv2.warpPerspective(image_to_be_aligned_green_component, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_image_red_component = cv2.warpPerspective(image_to_be_aligned_red_component, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_image_BGR = cv2.merge((aligned_image_blue_component, aligned_image_green_component, aligned_image_red_component))

                aligned_image_grayscale = cv2.cvtColor(aligned_image_BGR, cv2.COLOR_BGR2GRAY)
                this_pair_rmse = sk_metrics.normalized_root_mse(np.float32(reference_image_for_alignment_gray), np.float32(aligned_image_grayscale))
                this_pair_ssim = sk_metrics.structural_similarity(np.float32(reference_image_for_alignment_gray), np.float32(aligned_image_grayscale))
                per_dataset_nrmse_store.append(this_pair_rmse)
                per_dataset_ssim_store.append(this_pair_ssim)

                per_dataset_channel_extracted_aligned_he_store.append(np.array(cv2.split(aligned_image_BGR)))
                per_dataset_channel_extracted_aligned_he_store_flattened.append(per_dataset_channel_extracted_aligned_he_store[j].reshape(3, -1))
                per_dataset_aligned_BGR_he_image_store.append(aligned_image_BGR)
                per_dataset_warp_matrix_store.append(warp_matrix)
                per_dataset_enhanced_correlation_coefficient_store.append(enhanced_correlation_coefficient)

            ssim_store.append(per_dataset_ssim_store)
            nrmse_store.append(per_dataset_nrmse_store)
            channel_extracted_aligned_he_store.append(per_dataset_channel_extracted_aligned_he_store)
            channel_extracted_aligned_he_store_flattened.append(per_dataset_channel_extracted_aligned_he_store_flattened)
            aligned_all_combinations_BGR_he_image_store.append(per_dataset_aligned_BGR_he_image_store)
            warp_matrix_store.append(per_dataset_warp_matrix_store)
            enhanced_correlation_coefficient_store.append(per_dataset_enhanced_correlation_coefficient_store)
            # custom_enhanced_correlation_coefficient_store.append(per_dataset_custom_enhanced_correlation_coefficient_store)

            # for kp in range(20):
            #     print("custom: " + str(np.round(custom_enhanced_correlation_coefficient_store[0][kp], 3)) + " original: " + str(np.round(enhanced_correlation_coefficient_store[0][kp], 3)) + " difference: " + str(np.round((custom_enhanced_correlation_coefficient_store[0][kp] - enhanced_correlation_coefficient_store[0][kp]), 3)))


        all_combinations_aligned_he_dict = {'aligned_all_combinations_BGR_he_image_store': aligned_all_combinations_BGR_he_image_store,
                                            'warp_matrix_store': warp_matrix_store,
                                            'enhanced_correlation_coefficient_store': enhanced_correlation_coefficient_store,
                                            'normalized_datagrid_store': self.normalized_datagrid_store,
                                            'channel_extracted_he_store': self.channel_extracted_he_store,
                                            'channel_extracted_he_store_flattened': self.channel_extracted_he_store_flattened,
                                            'num_dim_reduced_components': self.num_dim_reduced_components,
                                            'num_datasets': self.num_datasets,
                                            'dataset_order': self.dataset_order,
                                            'used_dim_reduction_technique': self.data_preformatter_kp_object.used_dim_reduction_technique,
                                            'global_path_name': self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.global_path_name,
                                            'used_all_combinations_he_store_3d_for_alignment': self.used_all_combinations_he_store_3d_for_alignment,
                                            'did_i_use_custom_he_store_for_alignment': did_i_use_custom_he_store_for_alignment,
                                            'did_i_use_custom_warp_matrix_store': did_i_use_custom_warp_matrix_store,
                                            'did_i_use_gradient_alignment': gradient_alignment,
                                            'nrmse_store': nrmse_store,
                                            'ssim_store': ssim_store}

        similarity_metrics_summary_dict_form, similarity_metrics_summary_dataframe_form = self.generate_alignment_similarity_metrics(all_combinations_aligned_he_dict=all_combinations_aligned_he_dict)

        all_combinations_aligned_he_dict['similarity_metrics_summary_dict_form'] = similarity_metrics_summary_dict_form
        all_combinations_aligned_he_dict['similarity_metrics_summary_dataframe_form'] = similarity_metrics_summary_dataframe_form

        self.all_combinations_aligned_he_dict = all_combinations_aligned_he_dict

        folder_name = self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/he_stained_images/'
        file_name = filename_prefix + 'aligned_he_stained_dict.npy'

        if save_data == 1:
            np.save(folder_name + file_name, self.all_combinations_aligned_he_dict, 'dtype=object')

        return self.all_combinations_aligned_he_dict

    def print_all_combinations_aligned_similarity_metrics_summary_dataframe_form(self):
        """
        @brief Print a summary of similarity metrix. This function is called by self.align_all_combinations_of_dim_reduced_images_with_all_he_segments()
        @return: None
        """
        pd.options.display.colheader_justify = 'center'
        pd.options.display.width = None
        pd.options.display.max_colwidth = None
        pd.options.display.max_columns = None
        num_segments = self.aligned_all_combinations_with_all_he_segments_dict['segmented_he_store_used']['num_segments']
        for segment_count in range(num_segments):
            print("Similarity metrics for segment: " + str(segment_count))
            print(self.aligned_all_combinations_with_all_he_segments_dict['all_combinations_aligned_with_segment' + str(segment_count)]['similarity_metrics_summary_dataframe_form'])
            print("\n")

    def align_all_combinations_of_dim_reduced_images_with_all_he_segments(self, gradient_alignment=1, saved_aligned_he_segment_dict_filename=None, custom_segmented_he_store_dict=None, save_data=0, filename_prefix='', print_similarity_metrics=1, display_each_aligned_segment=0):

        """
        @brief Takes in a segmented_he_store_dict containing the details of each segment and carries out alignment of all dim_reduced_components with all those he_stain_segments.
            If a segmented_he_store_dict is not given, the current self.segmented_he_store_dict is used.
            If instead, a saved fiename for an already aligned version is given, simply load that into memory.

            Usage example:
                            ## Use segmented he stained images internally in the he_stains_kp_object.
                            # save_data = 1
                            # gradient_alignment = 1
                            # filename_prefix = ''
                            # print_similarity_metrics = 1
                            # display_each_aligned_segment = 1
                            # aligned_all_combinations_with_all_he_segments_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_all_he_segments(gradient_alignment=gradient_alignment,  save_data=save_data, filename_prefix=filename_prefix, print_similarity_metrics=print_similarity_metrics, display_each_aligned_segment=display_each_aligned_segment)

                            ### OR
                            ## Use an externally provided custom segmented he stained image dictionary and carry on the alignment.
                            # custom_segmented_he_store_dict = np.load("D:/msi_project_data/saved_outputs/he_stained_images/modified_naive_4_he_segmented_he_stained_dict.npy", allow_pickle=True)[()]
                            # save_data = 1
                            # gradient_alignment = 1
                            # filename_prefix = ''
                            # print_similarity_metrics = 1
                            # display_each_aligned_segment = 1
                            # aligned_all_combinations_with_all_he_segments_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_all_he_segments(gradient_alignment=gradient_alignment, custom_segmented_he_store_dict=custom_segmented_he_store_dict, save_data=save_data, filename_prefix=filename_prefix, print_similarity_metrics=print_similarity_metrics, display_each_aligned_segment=display_each_aligned_segment)

                            ### OR
                            ## Do not carry out any new alignment. Simply load an already aligned version.
                            # saved_aligned_he_segment_dict_filename = "D:\"
                            # print_similarity_metrics = 1
                            # display_each_aligned_segment = display_each_aligned_segment
                            # aligned_all_combinations_with_all_he_segments_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_all_he_segments(saved_aligned_he_segment_dict_filename=saved_aligned_he_segment_dict_filename, print_similarity_metrics=print_similarity_metrics, display_each_aligned_segment=display_each_aligned_segment)

        @param gradient_alignment: If set to 1, use gradient image (edge detection) based alignment.
        @param display_each_aligned_segment: If set to 1, display each aligned result for each segment.
        @param saved_aligned_he_segment_dict_filename: This can be used to provide a saved fiename for an already aligned version. If this is given, simply load that into memory, and do not do any new alignment.
        @param custom_segmented_he_store_dict: If this is given, use this dictionary for the alignment instead of the internal self.segmented_he_store_dict.
        @param save_data: Save data if set to 1.
        @param filename_prefix: Useful if we want to save the generated dictionary.
        @param print_similarity_metrics: If yes, print the summary of which dim_reduced_component aligned best with which h&e segment etc.
        @return aligned_all_combinations_with_all_he_segments_dict: A dictionary that saves all the combinations of alignments with each and every segment of h&e stained images.
        """

        if (saved_aligned_he_segment_dict_filename is not None) and (saved_aligned_he_segment_dict_filename != ''):
            self.aligned_all_combinations_with_all_he_segments_dict = np.load(saved_aligned_he_segment_dict_filename, allow_pickle=True)[()]

            if print_similarity_metrics == 1:
                self.print_all_combinations_aligned_similarity_metrics_summary_dataframe_form()
            if display_each_aligned_segment == 1:
                num_segments = self.aligned_all_combinations_with_all_he_segments_dict['segmented_he_store_used']['num_segments']
                for segment_count in range(num_segments):
                    custom_aligned_he_store_dict_for_display = self.aligned_all_combinations_with_all_he_segments_dict['all_combinations_aligned_with_segment'+str(segment_count)]
                    self.display_combined_aligned_images(custom_aligned_he_store_dict_for_display=custom_aligned_he_store_dict_for_display, slider_inside=1, save_figures=0, use_opencv_plotting=1, enable_slider=1)

            return self.aligned_all_combinations_with_all_he_segments_dict

        if (custom_segmented_he_store_dict is not None) and (custom_segmented_he_store_dict != ''):
            segmented_he_store_used = custom_segmented_he_store_dict
        else:
            segmented_he_store_used = self.segmented_he_store_dict

        segmented_he_store = segmented_he_store_used['segmented_he_store']
        segmented_he_store_np = np.array(segmented_he_store, dtype=object)
        num_segments = segmented_he_store_used['num_segments']

        aligned_all_combinations_with_all_he_segments_dict = {'segmented_he_store_used': segmented_he_store_used}

        for i in range(num_segments):
            self.temp_now_aligning_segment = i
            custom_segmented_he_store_single_segment = segmented_he_store_np[:, i]
            this_segment_all_combinations_aligned_he_segments_dict = self.align_all_combinations_of_dim_reduced_images_with_he_stained_images(gradient_alignment=gradient_alignment, custom_he_store=custom_segmented_he_store_single_segment, save_data=0)
            aligned_all_combinations_with_all_he_segments_dict['all_combinations_aligned_with_segment' + str(i)] = this_segment_all_combinations_aligned_he_segments_dict

            if display_each_aligned_segment == 1:
                self.display_combined_aligned_images(custom_aligned_he_store_dict_for_display=this_segment_all_combinations_aligned_he_segments_dict, slider_inside=1, save_figures=0, use_opencv_plotting=1, enable_slider=1)

        self.temp_now_aligning_segment = ''
        self.aligned_all_combinations_with_all_he_segments_dict = aligned_all_combinations_with_all_he_segments_dict

        if print_similarity_metrics == 1:
            self.print_all_combinations_aligned_similarity_metrics_summary_dataframe_form()

        folder_name = self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/he_stained_images/'
        file_name = filename_prefix + 'aligned_all_combinations_with_all_he_segments_dict.npy'
        if save_data == 1:
            np.save(folder_name + file_name, self.aligned_all_combinations_with_all_he_segments_dict, 'dtype=object')

        return self.aligned_all_combinations_with_all_he_segments_dict

    def create_custom_warp_matrix_store(self, optimally_warped_dim_reduced_component_and_he_segment_pair_array):

        """
        @brief Take in an array containing which nmf/pca component of each dataset best aligned with which segment of the h&e stained images, and broadcast the warp matrix corresponding to this combination along that entire dataset.
                Usage example:
                                optimally_warped_dim_reduced_component_and_he_segment_pair_array = [[1, 2],  # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [2, 1]]   # segment_2 with nmf/pca 1
                                                                              # Example, cph1-4 and naive1-3 will use the warp matrix corresponding to the alignment of segment 1 with nmf/pca component 2.
                                                                              # For the naive4 dataset which is the last dataset, the warp matrix corresponding to the alignment of segment 2 with nmf/pca 1 will be used.
                                custom_warp_matrix_store = he_stains_kp_object.create_custom_warp_matrix_store(optimally_warped_dim_reduced_component_and_he_segment_pair_array)

        @param optimally_warped_dim_reduced_component_and_he_segment_pair_array: This is an array that gives the index of the warp matrix to chose for each dataset. There should be two element entries for each dataset.
            The first element contains the h&e segment, and the second element, the nmf/pca component with which the alignment happened best.
        @return custom_warp_matrix_store: A custom set of warp matrices (The optimal warp matrix selected from each dataset is tiled to be the same for all nmf/pca components in that dataset)
        """

        if len(optimally_warped_dim_reduced_component_and_he_segment_pair_array) != self.num_datasets:
            print("Error. optimally_warped_dim_reduced_component_and_he_segment_pair_array length must be equal to the number of datasets")
            return

        num_segments = self.aligned_all_combinations_with_all_he_segments_dict['segmented_he_store_used']['num_segments']

        all_segment_all_dim_reduced_component_warp_matrix_store = []
        for segment_count in range(num_segments):
            this_segment_warp_matrix_store = self.aligned_all_combinations_with_all_he_segments_dict['all_combinations_aligned_with_segment' + str(segment_count)]['warp_matrix_store']
            all_segment_all_dim_reduced_component_warp_matrix_store.append(this_segment_warp_matrix_store)

        custom_warp_matrix_store = []
        for i in range(self.num_datasets):
            this_dataset_custom_warp_matrix_store = []
            for j in range(self.num_dim_reduced_components):
                optimally_aligning_segment_with_this_dataset = optimally_warped_dim_reduced_component_and_he_segment_pair_array[i][0]
                optimally_aligning_dim_reduced_component_with_the_optimal_segment = optimally_warped_dim_reduced_component_and_he_segment_pair_array[i][1]
                this_dataset_custom_warp_matrix_store.append(all_segment_all_dim_reduced_component_warp_matrix_store[optimally_aligning_segment_with_this_dataset][i][optimally_aligning_dim_reduced_component_with_the_optimal_segment])

            custom_warp_matrix_store.append(this_dataset_custom_warp_matrix_store)

        return custom_warp_matrix_store

    def blending_slider_callback(self, val):
        """
        @brief This is the callback function that dynamically adjusts the transparency of the two images that form a blended image.
        @param val: The current value of the slider is automatically added to this variable by cv2.
        """
        # print(val)
        new_image_1_weight = val / 100
        new_image_2_weight = (1.0 - new_image_1_weight)
        # cv2.namedWindow('blended_image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('blended_image', (1000, 1000))
        blended_image = cv2.addWeighted(self.temp_image_to_blend_1, new_image_1_weight, self.temp_image_to_blend_2, new_image_2_weight, 0.0)
        cv2.imshow('blended_image', blended_image)

    def overlay_and_blend_two_images(self, image_1, image_2, image_1_weight, image_2_weight, plot_result=0, enable_interactive_slider=0):
        """
        @brief Take two images and make a blended image.
            Usage example:
                            image_1 = he_stains_kp_object.normalized_datagrid_store[0][0]
                            image_2 = he_stains_kp_object.channel_extracted_he_store[0][0] #Blue channel of the 0th dataset
                            image_1_weight = 0.5
                            image_2_weight = (1 - image_1_weight)
                            plot_result = 1
                            enable_interactive_slider = 0

                            blended_image = he_stains_kp_object.overlay_and_blend_two_images(image_1, image_2, image_1_weight, image_2_weight, plot_result=plot_result, enable_interactive_slider=enable_interactive_slider)

        @param image_1: Image 1
        @param image_2: Image 2
        @param image_1_weight: alpha
        @param image_2_weight: beta
        @param plot_result: If set to 1, plot the blended output.
        @param enable_interactive_slider: If this is set to 1, this will start an interactive window where you can change the weights for each image realtime by moving a slider
        @return: A blended image.
        """

        blended_image = cv2.addWeighted(image_1, image_1_weight, image_2, image_2_weight, 0.0)

        if plot_result == 1:
            if enable_interactive_slider == 0:
                plt.imshow(blended_image)
                plt.pause(5)
            elif enable_interactive_slider == 1:
                # cv2.imshow('blended_image', blended_image)
                cv2.namedWindow('blended_image', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('blended_image', (1000, 1000))
                self.temp_image_to_blend_1 = image_1
                self.temp_image_to_blend_2 = image_2
                cv2.createTrackbar('slider_kp', 'blended_image', 0, 100, self.blending_slider_callback)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        return blended_image

    def create_false_colored_normalized_dim_reduced_datagrid_store(self):
        """
        @brief Convert the images in the internal self.normalized_datagrid_store attribute into false color by assigning the grayscale arrays to the green component of an empty BGR image
        @return false_colored_normalized_datagrid_store: False colored version of the normalized_datagrid_store
        """

        false_colored_normalized_datagrid_store = []
        for i in range(self.num_datasets):
            false_colored_normalized_datagrid_store_per_dataset = []
            for j in range(self.num_dim_reduced_components):
                dim_reduced_image_gray = self.normalized_datagrid_store[i][j]
                zeros = np.zeros(dim_reduced_image_gray.shape, np.uint8)
                dim_reduced_image_false_BGR = cv2.merge((zeros, dim_reduced_image_gray, zeros))  # False BGR image by assiging a grayscale image to the green channel of an otherwise all-black BGR image

                false_colored_normalized_datagrid_store_per_dataset.append(dim_reduced_image_false_BGR)

            false_colored_normalized_datagrid_store.append(false_colored_normalized_datagrid_store_per_dataset)

        return false_colored_normalized_datagrid_store

    def cv_subplot(self, imgs, pad=10, titles=None, win_name='all_combo_aligned'):
        """
        @brief Can be used to easily plot subplots in opencv.
        @param imgs: 2d np array of imgs (each img an np arrays of depth 1 or 3).
        @param pad: number of pixels to use for padding between images. must be even
        @param titles: (optional) np array of subplot titles
        @param win_name: name of cv2 window
        @return combined_frame
        """
        rows, cols = imgs.shape

        subplot_shapes = np.array([list(map(np.shape, x)) for x in imgs])
        sp_height, sp_width, depth = np.max(np.max(subplot_shapes, axis=0), axis=0)

        title_pad = 30
        if titles is not None:
            pad_top = pad + title_pad
        else:
            pad_top = pad

        frame = np.zeros((rows * (sp_height + pad_top), cols * (sp_width + pad), depth))

        for r in range(rows):
            for c in range(cols):
                img = imgs[r, c]
                h, w, _ = img.shape
                y0 = r * (sp_height + pad_top) + pad_top // 2
                x0 = c * (sp_width + pad) + pad // 2
                frame[y0:y0 + h, x0:x0 + w, :] = img

                if titles is not None:
                    frame = cv2.putText(frame, titles[r, c], (x0, y0 - title_pad // 4), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # cv2.imshow(win_name, np.uint8(frame))
        # cv2.waitKey(0)

        return np.uint8(frame)

    def display_combined_aligned_images(self, custom_aligned_he_store_dict_for_display=None, save_figures=0, plots_fileformat='svg', plots_dpi=600, use_opencv_plotting=0, enable_slider=0, slider_inside=1):
        """
        @brief Plots a grid of all aligned h&e stained images, aligned against each dim reduced component separately.
            Usage example:
                            ## Display the above generated all combinations of aligned images
                            # save_figures = 0
                            # plots_dpi = 600
                            # plots_fileformat = 'svg'
                            # use_opencv_plotting = 1
                            # enable_slider = 1
                            # slider_inside = 1
                            # he_stains_kp_object.display_combined_aligned_images(slider_inside=slider_inside, save_figures=save_figures, plots_fileformat=plots_fileformat, plots_dpi=plots_dpi, use_opencv_plotting=use_opencv_plotting, enable_slider=enable_slider)

                            ## OR:

                            ## Display the an externally provided aligned images store
                            # save_figures = 0
                            # plots_dpi = 600
                            # plots_fileformat = 'svg'
                            # use_opencv_plotting = 1
                            # enable_slider = 1
                            # saved_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/without_gradient_align_using_segmented_he_segment_2_aligned_he_stained_dict.npy"
                            # custom_aligned_he_store_dict_for_display = np.load(saved_aligned_he_dict_filename, allow_pickle=True)[()]
                            # he_stains_kp_object.display_combined_aligned_images(custom_aligned_he_store_dict_for_display=custom_aligned_he_store_dict_for_display, slider_inside=slider_inside, save_figures=save_figures, plots_fileformat=plots_fileformat, plots_dpi=plots_dpi, use_opencv_plotting=use_opencv_plotting, enable_slider=enable_slider)

        @param custom_aligned_he_store_dict_for_display: If this is given, use this for display as opposed to internally saved aligned dict.
        @param save_figures: save figures if set to 1
        @param slider_inside: If this is set to 1, both the slider and the plots appear in the same window. While this is easy for live demonstration, it does not do well when saving the image. Therefore, when saving the image, set this to 0.
        @param plots_dpi
        @param plots_fileformat
        @param use_opencv_plotting
        @param enable_slider
        """

        # if hasattr(self, 'all_combinations_aligned_he_dict'):
        if (custom_aligned_he_store_dict_for_display is not None) and (custom_aligned_he_store_dict_for_display != ''):
            aligned_he_store_dict_for_display = custom_aligned_he_store_dict_for_display
        else:
            aligned_he_store_dict_for_display = self.all_combinations_aligned_he_dict

        enhanced_correlation_coefficient_store = aligned_he_store_dict_for_display['enhanced_correlation_coefficient_store']
        nrmse_store = aligned_he_store_dict_for_display['nrmse_store']
        ssim_store = aligned_he_store_dict_for_display['ssim_store']
        aligned_he_image_BGR_image_store = aligned_he_store_dict_for_display['aligned_all_combinations_BGR_he_image_store']

        # elif hasattr(self, 'optimally_aligned_he_dict'):
        #     enhanced_correlation_coefficient_store = self.optimally_aligned_he_dict['optimal_enhanced_correlation_coefficient_store']
        #     aligned_he_image_BGR_image_store = self.optimally_aligned_he_dict['aligned_optimal_BGR_he_image_store']

        if use_opencv_plotting == 0:
            fig, ax = plt.subplots(self.num_datasets, self.num_dim_reduced_components)

            for i in range(self.num_datasets):
                for j in range(self.num_dim_reduced_components):
                    dim_reduced_image_gray = self.normalized_datagrid_store[i][j]
                    zeros = np.zeros(dim_reduced_image_gray.shape, np.uint8)
                    dim_reduced_image_false_BGR = cv2.merge((zeros, dim_reduced_image_gray, zeros))  # False BGR image by assiging a grayscale image to the green channel of an otherwise all-black BGR image
                    dim_reduced_image_false_BGR_weight = 0.5

                    aligned_he_image_BGR = aligned_he_image_BGR_image_store[i][j]
                    aligned_he_image_BGR_weight = (1 - dim_reduced_image_false_BGR_weight)

                    blended_image = cv2.addWeighted(dim_reduced_image_false_BGR, dim_reduced_image_false_BGR_weight, aligned_he_image_BGR, aligned_he_image_BGR_weight, 0.0)

                    ax[i, j].imshow(blended_image)
                    ax[i, j].xaxis.set_tick_params(length=0, width=0, labelsize=0)
                    ax[i, j].yaxis.set_tick_params(length=0, width=0, labelsize=0)
                    ax[i, j].spines[['bottom', 'right', 'left', 'top']].set_visible(0)
                    ax[i, j].tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
                    ax[i, j].set_ylabel(self.dataset_order[i], fontsize=40)
                    ax[i, j].set_xlabel(self.used_dim_reduction_technique + str(j) + '\n ECC: ' + str(np.round(enhanced_correlation_coefficient_store[i][j], 2)) + '\n NRMSE: ' + str(np.round(nrmse_store[i][j], 2)) + '\n SSIM: ' + str(np.round(ssim_store[i][j], 2)), fontsize=40)

            plot_title = 'Aligning H&E stained images separately with each  ' + self.used_dim_reduction_technique + '  component'
            plt.suptitle(plot_title, x=4, y=8, fontsize=80)
            fig.supylabel('Datasets', x=-0.5, y=5, fontsize=40)
            fig.supxlabel(self.used_dim_reduction_technique + ' Components', x=7.5, y=-1, fontsize=40)
            plt.subplots_adjust(left=0, bottom=0, right=15, top=10)
            plt.show()

            folder_name = self.global_path_name + '/saved_outputs/figures/'
            if save_figures == 1:
                fig.savefig(folder_name + plot_title.replace(' ', '_') + '.', plots_fileformat, dpi=plots_dpi, bbox_inches='tight')

        elif use_opencv_plotting == 1:
            blended_image_store = []
            blended_image_label_store = []
            for i in range(self.num_datasets):
                blended_image_store_per_dataset = []
                blended_image_label_store_per_dataset = []
                for j in range(self.num_dim_reduced_components):
                    dim_reduced_image_gray = self.normalized_datagrid_store[i][j]
                    zeros = np.zeros(dim_reduced_image_gray.shape, np.uint8)
                    dim_reduced_image_false_BGR = cv2.merge((zeros, dim_reduced_image_gray, zeros))  # False BGR image by assiging a grayscale image to the green channel of an otherwise all-black BGR image
                    dim_reduced_image_false_BGR_weight = 0.5

                    aligned_he_image_BGR = aligned_he_image_BGR_image_store[i][j]

                    aligned_he_image_BGR_weight = (1 - dim_reduced_image_false_BGR_weight)

                    blended_image = cv2.addWeighted(dim_reduced_image_false_BGR, dim_reduced_image_false_BGR_weight, aligned_he_image_BGR, aligned_he_image_BGR_weight, 0.0)

                    blended_image_store_per_dataset.append(blended_image)
                    label_string_this_dataset = self.dataset_order[i] + ',' + self.used_dim_reduction_technique + str(j) + '\n ECC: ' + str(np.round(enhanced_correlation_coefficient_store[i][j], 2)) + '\n NRMSE: ' + str(np.round(nrmse_store[i][j], 2)) + '\n SSIM: ' + str(np.round(ssim_store[i][j], 2))
                    blended_image_label_store_per_dataset.append(label_string_this_dataset)

                blended_image_store.append(blended_image_store_per_dataset)
                blended_image_label_store.append(blended_image_label_store_per_dataset)

            if enable_slider == 0:
                images_for_display = np.array(blended_image_store, dtype=object)
                labels_for_display = np.array(blended_image_label_store, dtype=object)
                combined_image_frame = self.cv_subplot(images_for_display, titles=labels_for_display)

                plot_title = 'Aligning H&E stained images separately with each  ' + self.used_dim_reduction_technique + '  component'
                folder_name = self.global_path_name + '/saved_outputs/figures/'
                if save_figures == 1:
                    cv2.imwrite(folder_name + plot_title.replace(' ', '_') + '.png', combined_image_frame)

            else:
                labels_for_display = np.array(blended_image_label_store, dtype=object)

                aligned_he_image_BGR_for_display = np.array(aligned_he_image_BGR_image_store, dtype=object)
                false_colored_dim_reduced_datagrid_store_for_display = np.array(self.false_colored_normalized_datagrid_store, dtype=object)

                combined_aligned_he_image_BGR_for_display = self.cv_subplot(aligned_he_image_BGR_for_display, titles=labels_for_display)
                combined_false_colored_dim_reduced_datagrid_store_for_display = self.cv_subplot(false_colored_dim_reduced_datagrid_store_for_display, titles=labels_for_display)

                blended_image = cv2.addWeighted(combined_aligned_he_image_BGR_for_display, 0.25, combined_false_colored_dim_reduced_datagrid_store_for_display, 0.75, 0.0)
                # cv2.namedWindow('blended_image', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('blended_image', (1000, 1000))
                self.temp_image_to_blend_1 = combined_aligned_he_image_BGR_for_display
                self.temp_image_to_blend_2 = combined_false_colored_dim_reduced_datagrid_store_for_display
                if slider_inside == 1:
                    cv2.namedWindow('blended_image', cv2.WINDOW_NORMAL)
                    cv2.createTrackbar('slider_kp', 'blended_image', 0, 100, self.blending_slider_callback)
                else:
                    cv2.imshow('blended_image', blended_image)
                    cv2.namedWindow('blended_image_slider', cv2.WINDOW_NORMAL)
                    cv2.createTrackbar('slider_kp', 'blended_image_slider', 0, 100, self.blending_slider_callback)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def threshold_he_images_for_alignment(self):
        """
        @brief This function uses adaptive gaussian thresholding to threshold the he_stained images.
            Usage example:

                          thresholded_he_store, false_colored_BGR_thresholded_he_store = he_stains_kp_object.threshold_he_images_for_alignment()
        @return thresholded_he_store, false_colored_BGR_thresholded_he_store: Returns a grayscale thresholded image store and a false colored thresholded image store.
        """

        thresholded_he_store = []
        false_colored_BGR_thresholded_he_store = []
        for i in range(self.num_datasets):
            this_he_image_BGR = self.he_store_3d[i]
            this_he_image_gray = cv2.cvtColor(this_he_image_BGR, cv2.COLOR_RGB2GRAY)  # convert to grayscale
            thresholded_he_image_gray = cv2.adaptiveThreshold(this_he_image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 10)
            # _, thresholded_he_image_gray_initial = cv2.threshold(this_he_image_gray, 80, 255, cv2.THRESH_TOZERO)
            # _, thresholded_he_image_gray_original = cv2.threshold(this_he_image_gray, 150, 255, cv2.THRESH_BINARY_INV)
            # dilated = cv2.dilate(thresholded_he_image_gray_initial, np.ones((5, 5), np.uint8))
            # thresholded_he_image_gray = (dilated - thresholded_he_image_gray_original)
            # cv2.imshow('kp1', thresholded_he_image_gray)
            # cv2.imshow('kp2', dilated)
            # cv2.imshow('kp3', thresholded_he_image_gray_initial)
            # cv2.imshow('kp4', thresholded_he_image_gray_original)

            zeros = np.zeros(thresholded_he_image_gray.shape, np.uint8)
            false_colored_BGR_thresholded_he_image = cv2.merge((zeros, zeros, thresholded_he_image_gray))

            thresholded_he_store.append(thresholded_he_image_gray)
            false_colored_BGR_thresholded_he_store.append(false_colored_BGR_thresholded_he_image)

        return thresholded_he_store, false_colored_BGR_thresholded_he_store

    def slider_callback_for_color_based_segmentation_roi_determination(self, val):
        """
        @brief This is the callback function that is used to determine a region of interest to sample the max and min hsv color values in that ROI to be later
            used in the color based segmentation of h&e stained images.
        @param val: Not used. (Because I use the same function for the call back of all my sliders. Instead, I get the trackbar position of each slider from within this function itself)
        @return temp_roi_and_preprocessing_dict: A dictionary containing two sub arrays: The lower color limit and upper color limit. Each sub array will have three values in the order: [h_val, s_val, v_val].
            The dictionary also contains the region of interest that was selected, the original image it was selected from, the image with the ROI labeled, the image after dilation and erosion.
        """
        temp_color_based_parameter_dict = self.temp_color_based_parameter_dict
        num_segments = temp_color_based_parameter_dict['num_segments']

        x_init = cv2.getTrackbarPos('x_init', 'roi_selection_for_color_based_segmentation_color_range_calculation')
        y_init = cv2.getTrackbarPos('y_init', 'roi_selection_for_color_based_segmentation_color_range_calculation')
        width = cv2.getTrackbarPos('width', 'roi_selection_for_color_based_segmentation_color_range_calculation')
        height = cv2.getTrackbarPos('height', 'roi_selection_for_color_based_segmentation_color_range_calculation')
        he_erosion_factor = cv2.getTrackbarPos('he_erosion_factor', 'roi_selection_for_color_based_segmentation_color_range_calculation')
        he_dilation_factor = cv2.getTrackbarPos('he_dilation_factor', 'roi_selection_for_color_based_segmentation_color_range_calculation')

        current_segment_number = cv2.getTrackbarPos('select_segment', 'roi_selection_for_color_based_segmentation_color_range_calculation')
        cv2.setTrackbarPos('select_segment', 'segmented_image_post_processing', current_segment_number)

        segmentation_dilation_factor = cv2.getTrackbarPos('segmentation_dilation_factor', 'segmented_image_post_processing')
        segmentation_blur_factor = cv2.getTrackbarPos('segmentation_blur_factor', 'segmented_image_post_processing')

        input_image = self.temp_color_based_segmentation_image

        if he_erosion_factor == 0:
            input_image_eroded = input_image.copy()
        else:
            input_image_eroded = cv2.erode(input_image.copy(), np.ones((1 * he_erosion_factor, 1 * he_erosion_factor), np.uint8))

        if he_dilation_factor == 0:
            input_image_dilated = input_image_eroded
        else:
            input_image_dilated = cv2.dilate(input_image_eroded, np.ones((1 * he_dilation_factor, 1 * he_dilation_factor), np.uint8))

        input_image_mask = input_image_dilated.copy()
        input_image_mask[y_init:y_init+height, x_init:x_init+width, :] = 0

        roi = input_image_dilated[y_init:y_init+height, x_init:x_init+width, :]

        mean_color = np.mean(roi.reshape((-1, 3)), axis=0)  # Do NOT convert to uint8. The cv2.inRange() function requires these values to be floats
        std_color = np.std(roi.reshape((-1, 3)), axis=0)  # Do NOT convert to uint8. The cv2.inRange() function requires these values to be floats

        lower_color_limit_array = mean_color-std_color
        upper_color_limit_array = mean_color+std_color

        mask = cv2.inRange(input_image_dilated, lower_color_limit_array, upper_color_limit_array)  # create mask containing pixels within the given color range
        result = cv2.bitwise_and(input_image_dilated, input_image_dilated, mask=mask)

        if segmentation_blur_factor == 0:
            blurred_segmented_image = result
        elif (segmentation_blur_factor % 2) == 0:
            print("Blur factor must be an odd number")
            blurred_segmented_image = result
        else:
            blurred_segmented_image = cv2.GaussianBlur(result, (1 * segmentation_blur_factor, 1 * segmentation_blur_factor), 0)

        if segmentation_dilation_factor == 0:
            dilated_segmented_image = blurred_segmented_image
        else:
            dilated_segmented_image = cv2.dilate(blurred_segmented_image, np.ones((1 * segmentation_dilation_factor, 1 * segmentation_dilation_factor), np.uint8))

        if self.temp_segmented_in_what_color_space == 'hsv':
            input_image_mask = cv2.cvtColor(input_image_mask, cv2.COLOR_HSV2BGR)  # Convert back into BGR
            final_segmented_image = cv2.cvtColor(dilated_segmented_image, cv2.COLOR_HSV2BGR)  # Convert back into BGR
        elif self.temp_segmented_in_what_color_space == 'rgb':
            input_image_mask = cv2.cvtColor(input_image_mask, cv2.COLOR_RGB2BGR)  # Convert back into BGR
            final_segmented_image = cv2.cvtColor(dilated_segmented_image, cv2.COLOR_RGB2BGR)  # Convert back into BGR
        elif self.temp_segmented_in_what_color_space == 'bgr':
            input_image_mask = input_image_mask  # Do not convert
            final_segmented_image = dilated_segmented_image  # Do not convert

        cv2.imshow('roi_selection_for_color_based_segmentation_color_range_calculation', input_image_mask)
        cv2.imshow('segmented_image_post_processing', final_segmented_image)


        this_segment_parameters = {'lower_color_limit': lower_color_limit_array,
                                    'upper_color_limit': upper_color_limit_array,
                                    'roi': roi,
                                    'roi_x_init': x_init,
                                    'roi_y_init': y_init,
                                    'roi_width': width,
                                    'roi_height': height,
                                    'original_image': input_image,
                                    'erosion_factor_on_original_he_stained_image': he_erosion_factor,
                                    'dilation_factor_on_original_he_stained_image': he_dilation_factor,
                                    'eroded_and_dilated_original_he_image': input_image_dilated,
                                    'roi_selected_eroded_and_dilated_he_image': input_image_mask,
                                    'segmentation_blur_factor': segmentation_blur_factor,
                                    'segmentation_dilation_factor': segmentation_dilation_factor}

        self.temp_roi_and_preprocessing_dict['segment_'+str(current_segment_number)+'_parameters'] = this_segment_parameters

    def color_based_segmentation_semi_automatic_parameter_determination(self, color_deterimining_he_image_BGR, parameter_dict=None):

        """
        @brief Segment a given image such that the segmented image only contains the pixels that fall between a specific color range
                Usage example:
                                color_deterimining_he_image_BGR = he_stains_kp_object.he_store_3d[0]
                                parameter_dict = {'self_determine_color_range': 1,
                                                  'segmented_in_what_color_space': 'rgb',
                                                  'num_segments': 4,
                                                  'roi_and_preprocessing_dict': None}


        @param color_deterimining_he_image_BGR: The image to be segmented
        @param parameter_dict:  The dictionary containing num_segments, erosion_factor_on_original_he_stained_image, dilation_factor_on_original_he_stained_image
            self_determine_color_range, roi_and_preprocessing_dict, segmentation_blur_factor, segmentation_dilation_factor...
        # @param segmented_in_what_color_space: This can be one of 'hsv', 'RGB', 'BGR'.
        # @param erosion_factor_on_original_he_stained_image: The erosion factor used to erode the original H&E stained images prior  to segmenting them
        # @param dilation_factor_on_original_he_stained_image: The dilation factor used to dilate the original H&E stained images prior  to segmenting them
        # @param self_determine_color_range: If this is set to 1, the color range determination interactive code pops up, and returns the necessary color range.
        # @param roi_and_preprocessing_dict: A dictionary containing two sub arrays: The lower color limit and upper color limit. Each sub array will have three values in the order: [h_val, s_val, v_val]
        # @param segmentation_blur_factor: Used to scale the gaussian blur window size.
        # @param segmentation_dilation_factor: Used to scale the dilation kernel size.
        @return final_segmented_image or self_determined_color_range: A segmented image that has also been a little blurred and dilated.
        """

        self.temp_color_based_parameter_dict = parameter_dict
        self.temp_segmented_in_what_color_space = self.temp_color_based_parameter_dict['segmented_in_what_color_space']
        num_segments = self.temp_color_based_parameter_dict['num_segments']

        if self.temp_segmented_in_what_color_space == 'hsv':
            color_corrected_image = cv2.cvtColor(color_deterimining_he_image_BGR.copy(), cv2.COLOR_BGR2HSV)  # Convert the image into HSV
        elif self.temp_segmented_in_what_color_space == 'rgb':
            color_corrected_image = cv2.cvtColor(color_deterimining_he_image_BGR.copy(), cv2.COLOR_BGR2RGB)  # Convert the image into RGB
        elif self.temp_segmented_in_what_color_space == 'bgr':
            color_corrected_image = color_deterimining_he_image_BGR.copy()  # Do not convert

        self.temp_color_based_segmentation_image = color_corrected_image
        self.temp_roi_and_preprocessing_dict = {'num_segments': num_segments,
                                                'segmented_in_what_color_space': self.temp_segmented_in_what_color_space}

        #  Determine ROI semi-automatically
        cv2.namedWindow('segmented_image_post_processing')
        cv2.imshow('roi_selection_for_color_based_segmentation_color_range_calculation', color_corrected_image)
        cv2.createTrackbar('x_init', 'roi_selection_for_color_based_segmentation_color_range_calculation', 0, np.max(color_corrected_image.shape), self.slider_callback_for_color_based_segmentation_roi_determination)
        cv2.createTrackbar('y_init', 'roi_selection_for_color_based_segmentation_color_range_calculation', 0, np.max(color_corrected_image.shape), self.slider_callback_for_color_based_segmentation_roi_determination)
        cv2.createTrackbar('width', 'roi_selection_for_color_based_segmentation_color_range_calculation', 0, np.max(color_corrected_image.shape), self.slider_callback_for_color_based_segmentation_roi_determination)
        cv2.createTrackbar('height', 'roi_selection_for_color_based_segmentation_color_range_calculation', 0, np.max(color_corrected_image.shape), self.slider_callback_for_color_based_segmentation_roi_determination)
        cv2.createTrackbar('he_erosion_factor', 'roi_selection_for_color_based_segmentation_color_range_calculation', 0, 15, self.slider_callback_for_color_based_segmentation_roi_determination)
        cv2.createTrackbar('he_dilation_factor', 'roi_selection_for_color_based_segmentation_color_range_calculation', 0, 15, self.slider_callback_for_color_based_segmentation_roi_determination)
        cv2.createTrackbar('select_segment', 'roi_selection_for_color_based_segmentation_color_range_calculation', 1, num_segments, self.slider_callback_for_color_based_segmentation_roi_determination)

        cv2.createTrackbar('segmentation_dilation_factor', 'segmented_image_post_processing', 0, 15, self.slider_callback_for_color_based_segmentation_roi_determination)
        cv2.createTrackbar('segmentation_blur_factor', 'segmented_image_post_processing', 3, 15, self.slider_callback_for_color_based_segmentation_roi_determination)
        cv2.createTrackbar('select_segment', 'segmented_image_post_processing', 1, num_segments, self.slider_callback_for_color_based_segmentation_roi_determination)

        cv2.setTrackbarPos('x_init', 'roi_selection_for_color_based_segmentation_color_range_calculation', 153)
        cv2.setTrackbarPos('y_init', 'roi_selection_for_color_based_segmentation_color_range_calculation', 96)
        cv2.setTrackbarPos('width', 'roi_selection_for_color_based_segmentation_color_range_calculation', 8)
        cv2.setTrackbarPos('height', 'roi_selection_for_color_based_segmentation_color_range_calculation', 8)
        cv2.setTrackbarPos('he_erosion_factor', 'roi_selection_for_color_based_segmentation_color_range_calculation', 0)
        cv2.setTrackbarPos('he_dilation_factor', 'roi_selection_for_color_based_segmentation_color_range_calculation', 0)
        cv2.setTrackbarPos('segmentation_dilation_factor', 'segmented_image_post_processing', 0)
        cv2.setTrackbarPos('segmentation_blur_factor', 'segmented_image_post_processing', 3)
        cv2.waitKey(0)

        color_based_roi_and_preprocessing_dict = self.temp_roi_and_preprocessing_dict

        return color_based_roi_and_preprocessing_dict

    def segment_he_stained_images(self, segmentation_technique='color_based', saved_segmented_he_store_dict_filename='', parameter_dict=None, save_data=1, filename_prefix=''):

        """
        @brief Segment the H&E stained images into a given number of segments
            Usage example:
                           #### Segment h&e stained images
                            ### 1. Kmeans
                            # segmentation_technique = 'k_means'
                            # parameter_dict = {'num_segments': 4}
                            # save_data = 1
                            # filename_prefix = ''
                            # segmented_he_store_dict = he_stains_kp_object.segment_he_stained_images(segmentation_technique=segmentation_technique, parameter_dict=parameter_dict,  save_data=save_data, filename_prefix=filename_prefix)

                            ### 2. Colorbased
                            ##  Externally provide ROIs-color limits and other processing factors
                            # segmentation_technique = 'color_based'
                            # save_data = 1
                            # filename_prefix = 'original_he_included_'
                            # color_based_roi_and_preprocessing_dict = np.load("D:/msi_project_data/saved_outputs/he_stained_images/segmented_he_stained_dict.npy", allow_pickle=True)[()]['color_based_roi_and_preprocessing_dict']  # Only useful for color_based segmentation technique
                            # parameter_dict = {'color_based_roi_and_preprocessing_dict': color_based_roi_and_preprocessing_dict,
                            #                   'self_determine_color_range': 0,
                            #                   'segmented_in_what_color_space': 'rgb',
                            #                   'num_segments': 4}
                            # segmented_he_store_dict = he_stains_kp_object.segment_he_stained_images(segmentation_technique=segmentation_technique, parameter_dict=parameter_dict, save_data=save_data, filename_prefix=filename_prefix)

                            ## OR
                            ## Determine ROIs-color limits and other processing factors semi-automatically
                            # segmentation_technique = 'color_based'
                            # save_data = 1
                            # filename_prefix = ''
                            # parameter_dict = {'self_determine_color_range': 1,
                            #                   'segmented_in_what_color_space': 'rgb',
                            #                   'num_segments': 4,
                            #                   'roi_and_preprocessing_dict': None}
                            #
                            # segmented_he_store_dict = he_stains_kp_object.segment_he_stained_images(segmentation_technique=segmentation_technique, parameter_dict=parameter_dict,  save_data=save_data, filename_prefix=filename_prefix)

                            ## OR
                            # Simply load an existing segmented_he_store_dict
                            # saved_segmented_he_store_filename = ""
                            # segmented_he_store_dict = he_stains_kp_object.segment_he_stained_images(saved_segmented_he_store_dict_filename=saved_segmented_he_store_dict_filename)


        @param segmentation_technique: Could be 'k_means', 'color_based', '', and ''
        @param saved_segmented_he_store_filename:  If this is given, then load the segmentation results from here rather than calculating it again.
        @param save_data: Save results if set to 1
        @param filename_prefix: Prefix for saving the results dictionary
        @param parameter_dict: An optional parameter dictionary that may be required for specific segmentation techniques
        @return segmented_he_store_dict: A dictioonary containing the segmented H&E stained images and metadata.
        """
        if (saved_segmented_he_store_dict_filename is not None) and (saved_segmented_he_store_dict_filename != ''):
            self.segmented_he_store_dict = np.load(saved_segmented_he_store_dict_filename, allow_pickle=True)[()]
            return self.segmented_he_store_dict

        segmented_he_store_dict = {'segmentation_method_used': segmentation_technique,
                                   'parameter_dict': parameter_dict}
        segmented_he_store = []

        if segmentation_technique == 'k_means':
            num_segments = parameter_dict['num_segments']
            for i in range(self.num_datasets):
                this_he_image_BGR = self.he_store_3d[i]
                this_he_image_RBG = cv2.cvtColor(this_he_image_BGR, cv2.COLOR_BGR2RGB)
                this_he_image_2d = np.float32(this_he_image_RBG.reshape((-1, 3)))

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
                k = num_segments
                attempts = 10

                ret, label, center = cv2.kmeans(this_he_image_2d, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
                center = np.uint8(center)  # This contains the colors at the centers
                entire_segmented_image_flattened = center[label.flatten()]  # Assign the relevant color to each pixel

                result_image = entire_segmented_image_flattened.reshape((this_he_image_RBG.shape))

                segments_per_dataset = []
                for segment_number in range(num_segments):
                    empty_image = np.zeros(entire_segmented_image_flattened.shape, np.uint8)
                    mask_of_segment = empty_image
                    mask_of_segment[label.flatten() == segment_number] = 1
                    this_segmented_image_flattened = np.multiply(mask_of_segment, entire_segmented_image_flattened)  # Set everything other than the segmented region to zero
                    this_segment_reconstructed = this_segmented_image_flattened.reshape((this_he_image_RBG.shape))
                    segments_per_dataset.append(cv2.cvtColor(this_segment_reconstructed, cv2.COLOR_RGB2BGR))

                segmented_he_store.append(segments_per_dataset)

            segmented_he_store_dict['num_segments'] = num_segments
            segmented_he_store_dict['num_attempts'] = attempts

        elif segmentation_technique == 'color_based':
            num_segments = parameter_dict['num_segments']
            self_determine_color_range = parameter_dict['self_determine_color_range']
            segmented_in_what_color_space = parameter_dict['segmented_in_what_color_space']

            if self_determine_color_range == 1:
                color_deterimining_he_image_BGR = self.he_store_3d[0]
                color_based_roi_and_preprocessing_dict = self.color_based_segmentation_semi_automatic_parameter_determination(color_deterimining_he_image_BGR, parameter_dict=parameter_dict)
            else:
                color_based_roi_and_preprocessing_dict = parameter_dict['color_based_roi_and_preprocessing_dict']

            for i in range(self.num_datasets):
                #  Use the color_based_roi_and_preprocessing_dict either externally given, or calculated in the previous step and perform the segmentations
                segmented_in_what_color_space = color_based_roi_and_preprocessing_dict['segmented_in_what_color_space']
                this_he_image_BGR = self.he_store_3d[i]

                if segmented_in_what_color_space == 'hsv':
                    color_corrected_image = cv2.cvtColor(this_he_image_BGR.copy(), cv2.COLOR_BGR2HSV)  # Convert the image into HSV
                elif segmented_in_what_color_space == 'rgb':
                    color_corrected_image = cv2.cvtColor(this_he_image_BGR.copy(), cv2.COLOR_BGR2RGB)  # Convert the image into RGB
                elif segmented_in_what_color_space == 'bgr':
                    color_corrected_image = this_he_image_BGR.copy()  # Do not convert

                segments_per_dataset = [this_he_image_BGR]  # Added the original image so that we can have a good comparison between the segments
                for segment_number in range(1, num_segments+1):
                    this_segment_parameter_dict = color_based_roi_and_preprocessing_dict['segment_' + str(segment_number) + '_parameters']

                    erosion_factor_on_original_he_stained_image = this_segment_parameter_dict['erosion_factor_on_original_he_stained_image']
                    dilation_factor_on_original_he_stained_image = this_segment_parameter_dict['dilation_factor_on_original_he_stained_image']
                    lower_color_limit = this_segment_parameter_dict['lower_color_limit']
                    upper_color_limit = this_segment_parameter_dict['upper_color_limit']
                    segmentation_blur_factor = this_segment_parameter_dict['segmentation_blur_factor']
                    segmentation_dilation_factor = this_segment_parameter_dict['segmentation_dilation_factor']

                    if erosion_factor_on_original_he_stained_image == 0:
                        input_image_eroded = color_corrected_image.copy()
                    else:
                        input_image_eroded = cv2.erode(color_corrected_image.copy(), np.ones((1 * erosion_factor_on_original_he_stained_image, 1 * erosion_factor_on_original_he_stained_image), np.uint8))

                    if dilation_factor_on_original_he_stained_image == 0:
                        input_image_dilated = input_image_eroded
                    else:
                        input_image_dilated = cv2.dilate(input_image_eroded, np.ones((1 * dilation_factor_on_original_he_stained_image, 1 * dilation_factor_on_original_he_stained_image), np.uint8))

                    mask = cv2.inRange(input_image_dilated, lower_color_limit, upper_color_limit)  # create mask containing pixels within the given color range
                    result = cv2.bitwise_and(color_corrected_image, color_corrected_image, mask=mask)

                    if segmentation_blur_factor == 0:
                        blurred_segmented_image = result
                    elif (segmentation_blur_factor % 2) == 0:
                        blurred_segmented_image = result
                    else:
                        blurred_segmented_image = cv2.GaussianBlur(result, (1 * segmentation_blur_factor, 1 * segmentation_blur_factor), 0)

                    if segmentation_dilation_factor == 0:
                        dilated_segmented_image = blurred_segmented_image
                    else:
                        dilated_segmented_image = cv2.dilate(blurred_segmented_image, np.ones((1 * segmentation_dilation_factor, 1 * segmentation_dilation_factor), np.uint8))

                    if segmented_in_what_color_space == 'hsv':
                        final_segmented_image = cv2.cvtColor(dilated_segmented_image, cv2.COLOR_HSV2BGR)  # Convert back into BGR
                    elif segmented_in_what_color_space == 'rgb':
                        final_segmented_image = cv2.cvtColor(dilated_segmented_image, cv2.COLOR_RGB2BGR)  # Convert back into BGR
                    elif segmented_in_what_color_space == 'bgr':
                        final_segmented_image = dilated_segmented_image  # Do not convert

                    segments_per_dataset.append(final_segmented_image)

                segmented_he_store.append(segments_per_dataset)

            segmented_he_store_dict['color_based_roi_and_preprocessing_dict'] = color_based_roi_and_preprocessing_dict
            segmented_he_store_dict['num_segments'] = num_segments
            segmented_he_store_dict['segmented_in_what_color_space'] = segmented_in_what_color_space
            segmented_he_store_dict['was_color_range_self_determined'] = self_determine_color_range

        segmented_he_store_for_display = np.array(segmented_he_store,  dtype=object)
        combined_frame = self.cv_subplot(segmented_he_store_for_display)
        cv2.imshow('segmented_he_images', combined_frame)
        cv2.waitKey(0)

        segmented_he_store_dict['segmented_he_store'] = segmented_he_store

        self.segmented_he_store_dict = segmented_he_store_dict

        folder_name = self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/he_stained_images/'
        file_name = filename_prefix + 'segmented_he_stained_dict.npy'
        if save_data == 1:
            np.save(folder_name + file_name, self.segmented_he_store_dict, 'dtype=object')

        return segmented_he_store_dict

    def generate_alignment_similarity_metrics(self, all_combinations_aligned_he_dict=None, saved_all_combinations_aligned_he_dict_filename=None):

        """
        @brief This function prints a summary of the important metrics per dataset in the aligned images for all combinatios of h&e stained and dim_reduced components
                Usage example:
                                all_combinations_aligned_he_dict=all_combinations_aligned_he_dict
                                all_combinations_alignment_similarity_metrics_dict = he_stains_kp_object.print_alignment_similarity_metrics(all_combinations_aligned_he_dict=all_combinations_aligned_he_dict)

                                ## OR:
                                ## Load a pre-existing aligned data containing dictionary
                                saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/modified_align_using_segmented_he_segment_2_aligned_he_stained_dict.npy"
                                all_combinations_alignment_similarity_metrics_dict = he_stains_kp_object.print_alignment_similarity_metrics(saved_all_combinations_aligned_he_dict_filename=saved_all_combinations_aligned_he_dict_filename)
        @param saved_all_combinations_aligned_he_dict_filename: If this is given, a preexisting dataset will be loaded, and the metrics will be calculated from that.
        @param all_combinations_aligned_he_dict: Enter a aligned dictionary from a previous step here.
        @return all_combinations_alignment_similarity_metrics_dict: A dictionary containing all the important metrics of the alignment.
        """

        if (saved_all_combinations_aligned_he_dict_filename is None) | (saved_all_combinations_aligned_he_dict_filename == ''):
            all_combinations_aligned_he_dict = all_combinations_aligned_he_dict
        else:
            all_combinations_aligned_he_dict = np.load(saved_all_combinations_aligned_he_dict_filename, allow_pickle=True)[()]

        max_nrmse_per_dataset = np.round(np.float32(np.max(np.array(all_combinations_aligned_he_dict['nrmse_store'], dtype=object), axis=1)), 3)
        max_nrmse_per_dataset_index = np.argmax(np.array(all_combinations_aligned_he_dict['nrmse_store'], dtype=object), axis=1)
        min_nrmse_per_dataset = np.round(np.float32(np.min(np.array(all_combinations_aligned_he_dict['nrmse_store'], dtype=object), axis=1)), 3)
        min_nrmse_per_dataset_index = np.argmin(np.array(all_combinations_aligned_he_dict['nrmse_store'], dtype=object), axis=1)
        max_ssim_per_dataset = np.round(np.float32(np.max(np.array(all_combinations_aligned_he_dict['ssim_store'], dtype=object), axis=1)), 3)
        max_ssim_per_dataset_index = np.argmax(np.array(all_combinations_aligned_he_dict['ssim_store'], dtype=object), axis=1)
        min_ssim_per_dataset = np.round(np.float32(np.min(np.array(all_combinations_aligned_he_dict['ssim_store'], dtype=object), axis=1)), 3)
        min_ssim_per_dataset_index = np.argmin(np.array(all_combinations_aligned_he_dict['ssim_store'], dtype=object), axis=1)
        max_ecc_per_dataset = np.round(np.float32(np.max(np.array(all_combinations_aligned_he_dict['enhanced_correlation_coefficient_store'], dtype=object), axis=1)), 3)
        max_ecc_per_dataset_index = np.argmax(np.array(all_combinations_aligned_he_dict['enhanced_correlation_coefficient_store'], dtype=object), axis=1)
        min_ecc_per_dataset = np.round(np.float32(np.min(np.array(all_combinations_aligned_he_dict['enhanced_correlation_coefficient_store'], dtype=object), axis=1)), 3)
        min_ecc_per_dataset_index = np.argmin(np.array(all_combinations_aligned_he_dict['enhanced_correlation_coefficient_store'], dtype=object), axis=1)

        all_combinations_alignment_similarity_metrics_dict_form = {'max(nrmse)': max_nrmse_per_dataset, 'argmax(nrmse)': max_nrmse_per_dataset_index, 'min(nrmse)': min_nrmse_per_dataset, 'argmin(nrmse)': min_nrmse_per_dataset_index, 'max(ssim)': max_ssim_per_dataset, 'argmax(ssim)': max_ssim_per_dataset_index, 'min(ssim)': min_ssim_per_dataset, 'argmin(ssim)': min_ssim_per_dataset_index, 'max(ecc)': max_ecc_per_dataset, 'argmax(ecc)': max_ecc_per_dataset_index, 'min(ecc)': min_ecc_per_dataset, 'argmin(ecc)': min_ecc_per_dataset_index}

        all_combinations_alignment_results_dict_dataframe_form = pd.DataFrame(all_combinations_alignment_similarity_metrics_dict_form)
        columns = [('nrmse', 'max'), ('nrmse', 'argmax'), ('nrmse', 'min'), ('nrmse', 'argmin'), ('ssim', 'max'), ('ssim', 'argmax'), ('ssim', 'min'), ('ssim', 'argmin'), ('ecc', 'max'), ('ecc', 'argmax'), ('ecc', 'min'), ('ecc', 'argmin')]
        all_combinations_alignment_results_dict_dataframe_form.columns = pd.MultiIndex.from_tuples(columns)
        pd.options.display.colheader_justify = 'center'
        pd.options.display.width = None
        pd.options.display.max_colwidth = None
        pd.options.display.max_columns = None


        return all_combinations_alignment_similarity_metrics_dict_form, all_combinations_alignment_results_dict_dataframe_form

    def modify_existing_all_combinations_aligned_he_image_dict(self, old_all_combinations_aligned_he_image_dict_filename, save_data=1, modified_all_combinations_aligned_he_image_dict_filename_prefix=''):

        """
        @brief This function is designed to load an existing saved all_combinations_aligned_he_image dictionary, and modify it to suit newer versions of the he_stains_kp class
            Example use:
                        old_all_combinations_aligned_he_image_dict_filename  = "D:/msi_project_data/saved_outputs/he_stained_images/all_combinations_aligned_he_stained_dict.npy"
                        save_data =  1
                        modified_all_combinations_aligned_he_image_dict_filename_prefix = 'modified_'
                        modified_all_combinations_aligned_he_image_dict = he_stains_kp_object.modify_existing_all_combinations_aligned_he_image_dict(old_all_combinations_aligned_he_image_dict_filename, save_data=save_data, modified_all_combinations_aligned_he_image_dict_filename_prefix=modified_all_combinations_aligned_he_image_dict_filename_prefix)

        @param old_all_combinations_aligned_he_image_dict_filename: This is an essential argument, This points to the existing nmf file that needs to be modified
        @param save_data:  Save the modified nmf file.
        @param modified_all_combinations_aligned_he_image_dict_filename_prefix: Enter a prefix that I want the modified dictionary to have if it gets saved (i.e, if save_data ==1)
            This prefix will be prepended to the name of the old_all_combinations_aligned_he_image_dict_filename's filename part.
        @return  modified_all_combinations_aligned_he_image_dict:  Returns the modified dictionary
        """
        print("This code must be adjusted everytime you use it to include the modifications you intend to do")

        old_saved_all_combinations_aligned_he_image_dict = np.load(old_all_combinations_aligned_he_image_dict_filename, allow_pickle=True)[()]
        modified_all_combinations_aligned_he_image_dict = old_saved_all_combinations_aligned_he_image_dict.copy()

        #############################################
        ###  Do the modifications:
        ### Ex:

        ## del modified_all_combinations_aligned_he_image_dict['nmf_outputs']
        ## modified_all_combinations_aligned_he_image_dict['dim_reduced_outputs'] = old_saved_all_combinations_aligned_he_image_dict['nmf_outputs']

        # modified_all_combinations_aligned_he_image_dict['used_dim_reduction_technique'] = 'nmf'
        # modified_all_combinations_aligned_he_image_dict['global_path_name'] = self.global_path_name
        #############################################

        self.all_combinations_aligned_he_image_dict = modified_all_combinations_aligned_he_image_dict
        print("Internal representation of the all_combinations_aligned_he_image_dict variable has been updated")

        if save_data == 1:
            old_filename = old_all_combinations_aligned_he_image_dict_filename.split('/')[-1]
            folder_name = self.global_path_name + '/saved_outputs/he_stained_images/'
            np.save(folder_name + modified_all_combinations_aligned_he_image_dict_filename_prefix + old_filename, modified_all_combinations_aligned_he_image_dict, 'dtype=object')

        return modified_all_combinations_aligned_he_image_dict

    def determine_optimal_he_alignment(self, optimally_warped_dim_reduced_component_and_he_segment_pair_array, save_data=1, filename_prefix=''):
        """
        @brief This function will enter a custom warp matrix store to the alignment algorithm to optimally align the h&e stained images with the dim_reduced_components
                Usage example:
                                 optimally_warped_dim_reduced_component_and_he_segment_pair_array = [[1, 2],  # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [1, 2],   # segment_1 with nmf/pca 2
                                                                                                  [2, 1]]   # segment_2 with nmf/pca 1
                                                                              # Example, cph1-4 and naive1-3 will use the warp matrix corresponding to the alignment of segment 1 with nmf/pca component 2.
                                                                              # For the naive4 dataset which is the last dataset, the warp matrix corresponding to the alignment of segment 2 with nmf/pca 1 will be used.
                               `save_data = 1
                               fillename_prefix = 'test_optimal_'
                               optimal_alignment = he_stains_kp_object.determine_optimal_he_alignment(optimally_warped_dim_reduced_component_and_he_segment_pair_array, save_data=save_data, filename_prefix=filename_prefix)

        @param optimally_warped_dim_reduced_component_and_he_segment_pair_array: Contains pairs of [segment_number, component_number]. This is an array that gives the index of the warp matrix to chose for each dataset. There should be two element entries for each dataset.
            The first element contains the h&e segment, and the second element, the nmf/pca component with which the alignment happened best.
        @param save_data: Save data if set to 1
        @param filename_prefix: Used while saving
        @return optimally_aligned_he_dict: Returns the optimally aligned h&e stained image store
        """

        print("Please confirm that the optimally_warped_dim_reduced_component_and_he_segment_pair_array is correct")
        val = input("If correct, please type: yes")
        if val != "yes":
            print("Error")
            return
        custom_warp_matrix_store = self.create_custom_warp_matrix_store(optimally_warped_dim_reduced_component_and_he_segment_pair_array)
        self.optimally_aligned_he_store_dict = self.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_warp_matrix_store=custom_warp_matrix_store, save_data=0)

        folder_name = self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/he_stained_images/'
        file_name = filename_prefix + 'optimally_aligned_he_store_dict.npy'
        if save_data == 1:
            np.save(folder_name + file_name, self.optimally_aligned_he_store_dict, 'dtype=object')

        return self.optimally_aligned_he_store_dict

    def basis_reconstruction(self):
        ### Add stuff here
        print("None")

    def he_stained_image_processing_pipeline(self, pipeline_parameters_dict=None, save_data=0, filename_prefixes='_pipeline_'):

        """
        @brief Starting with raw h&e stained images and raw dimensionality reduced components, perform the functions ranging from
         semi automatic segmentation, aligning all combinations of dim reduced components and h&e segments, selecting the most suitable alignment warp matrices and
         hence finding the optimal alignment dictionary, and finally doing the basis reconstruction of h&e images using dim_reduced components as a basis.
        @return basis_reconstruction_result:
        """

        all_pipeline_parameters_dict = {'segmentation_parameter_dict': dict(),
                                        'segmentation_technique': 'color_based',
                                        'enable_printing': 1,
                                        'gradient_alignment': 1,
                                        'optimally_warped_dim_reduced_component_and_he_segment_pair_array': [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [2, 1]],
                                        'slider_inside': 1,
                                        'enable_slider': 1,
                                        'use_opencv_plotting': 1,
                                        'save_figures': 1,
                                        'plots_fileformat': 'png',
                                        'plots_dpi': 600}

        if (pipeline_parameters_dict is not None) and (pipeline_parameters_dict != ''):
            for key in pipeline_parameters_dict.keys():
                all_pipeline_parameters_dict[key] = pipeline_parameters_dict[key]

        segmented_he_store_dict = self.segment_he_stained_images(segmentation_technique=all_pipeline_parameters_dict['segmentation_technique'], parameter_dict=all_pipeline_parameters_dict['parameter_dict'], save_data=save_data, filename_prefix=filename_prefixes)
        aligned_all_combinations_with_all_he_segments_dict = self.align_all_combinations_of_dim_reduced_images_with_all_he_segments(gradient_alignment=all_pipeline_parameters_dict['gradient_alignment'], save_data=save_data, filename_prefix=filename_prefixes, print_similarity_metrics=all_pipeline_parameters_dict['enable_printing'])
        optimal_he_alignment_store_dict = self.determine_optimal_he_alignment(all_pipeline_parameters_dict['optimally_warped_dim_reduced_component_and_he_segment_pair_array'], save_data=save_data, filename_prefix=filename_prefixes)
        # basis_reconstructed_he_image_store_dict = self.basis_reconstruction()
        self.display_combined_aligned_images(save_figures=all_pipeline_parameters_dict['save_figures'], plots_fileformat=all_pipeline_parameters_dict['plots_fileformat'], plots_dpi=all_pipeline_parameters_dict['plots_dpi'], use_opencv_plotting=all_pipeline_parameters_dict['use_opencv_plotting'], enable_slider=all_pipeline_parameters_dict['enable_slider'], slider_inside=all_pipeline_parameters_dict['slider_inside'])
