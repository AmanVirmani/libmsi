import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
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
            this_dataset[for_nmf | for_pca] = 0
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

    def align_all_combinations_of_dim_reduced_images_with_he_stained_images(self, gradient_alignment=1, saved_all_combinations_aligned_he_dict_filename=None, custom_he_store=None, custom_warp_matrix_store=None, custom_enhanced_correlation_coefficient_store=None, save_data=0, filename_prefix=''):

        """
        @brief This function does a homography transformation on the h&e stained images to spatially match them with the nmf/pca images.
            It calculates the correlation coefficient after aligning an H&E stained image of a dataset  with each and every nmf/pca component. It does NOT  determine which component gives the best homography
            This function calculates warp transformations to transform the h&e image  into the nmf image based on maximizing enhanced corr. coeff
            Maximum enhanced correlation coefficient is calculated for a given h&e stained image against all of the nmf/pca images corresponding to that dataset.
            Usage example:
                            ## Use the internal he_store_3d for the alignments
                            save_data = 1
                            filename_prefix = 'modified_'
                            gradient_alignment = 1
                            aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(gradient_alignment=gradient_alignment, save_data=save_data, filename_prefix=filename_prefix)

                            ## OR
                            ## Use a custom he store for the alignment
                            thresholded_he_store, false_colored_BGR_thresholded_he_store = he_stains_kp_object.threshold_he_images_for_alignment()
                            save_data = 1
                            filename_prefix = 'thresholded_'
                            gradient_alignment = 1
                            custom_he_store = false_colored_BGR_thresholded_he_store
                            aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(gradient_alignment=gradient_alignment, custom_he_store=custom_he_store, custom_warp_matrix_store = custom_warp_matrix_store,

                            ## OR
                            ## If a pre-saved file exists, use that instead
                            saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/modified_all_combinations_aligned_he_stained_dict.npy"
                            aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(saved_all_combinations_aligned_he_dict_filename = saved_all_combinations_aligned_he_dict_filename)

                            ## OR
                            thresholded_he_store, false_colored_BGR_thresholded_he_store = he_stains_kp_object.threshold_he_images_for_alignment()
                            saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/thresholded_aligned_he_stained_dict.npy"
                            saved_thresholded_aligned_he_store_dict = np.load(saved_all_combinations_aligned_he_dict_filename, allow_pickle=True)[()]
                            custom_warp_matrix_store = saved_thresholded_aligned_he_store_dict['warp_matrix_store']
                            custom_enhanced_correlation_coefficient_store = saved_thresholded_aligned_he_store_dict['enhanced_correlation_coefficient_store']
                            save_data = 1
                            filename_prefix = 'custom_warp_thresholded_'
                            gradient_alignment = 1
                            aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(gradient_alignment=gradient_alignment, custom_warp_matrix_store=custom_warp_matrix_store, custom_enhanced_correlation_coefficient_store=custom_enhanced_correlation_coefficient_store, save_data=save_data, filename_prefix=filename_prefix)

        @param save_data: Save data if enabled
        @param gradient_alignment: If set to 1, align the gradients (edge detected version) of the images. If set to 0, align the raw images.
        @param custom_warp_matrix_store: If given, the homography will NOT be calculated. Instead, alignment will be done based on this warp matrix.
        @param custom_he_store: Use this array instead of the he_store_3d to do the alignment.
        @param filename_prefix: Necessary if save_data is set to 1.
        @param saved_all_combinations_aligned_he_dict_filename: Load a presaved dictionary containing all combinations of nmf/pca components aligned with the H&E stained images aligned set of images
        @return all_combinations_aligned_he_dict: A dictionary containing the aligned h&e stained images, the normalized dim reduced data used for the alignment, the original h&e stained images used for the alignment,
            warp matrices used for the alignments, enhanced correlation coefficients of the alignments.
        """

        if (saved_all_combinations_aligned_he_dict_filename is not None) and (saved_all_combinations_aligned_he_dict_filename != ''):
            self.all_combinations_aligned_he_dict = np.load(saved_all_combinations_aligned_he_dict_filename, allow_pickle=True)[()]
            self.normalized_datagrid_store = self.all_combinations_aligned_he_dict['normalized_datagrid_store']
            # self.channel_extracted_he_store = self.all_combinations_aligned_he_dict['channel_extracted_he_store']
            # self.channel_extracted_he_store_flattened = self.all_combinations_aligned_he_dict['channel_extracted_he_store_flattened']
            # self.he_store_3d = self.all_combinations_aligned_he_dict['he_store_3d']
            self.num_dim_reduced_components = self.all_combinations_aligned_he_dict['num_dim_reduced_components']
            self.num_datasets = self.all_combinations_aligned_he_dict['num_datasets']
            self.dataset_order = self.all_combinations_aligned_he_dict['dataset_order']
            self.used_dim_reduction_technique = self.all_combinations_aligned_he_dict['used_dim_reduction_technique']
            self.global_path_name = self.all_combinations_aligned_he_dict['global_path_name']

            ######################
            self.used_all_combinations_he_store_3d_for_alignment = self.all_combinations_aligned_he_dict['used_all_combinations_he_store_3d_for_alignment']
            # self.channel_extracted_he_store = self.all_combinations_aligned_he_dict['channel_extracted_he_store']
            # self.channel_extracted_he_store_flattened = self.all_combinations_aligned_he_dict['channel_extracted_he_store_flattened']

            self.did_i_use_custom_he_store_for_alignment = self.all_combinations_aligned_he_dict['did_i_use_custom_he_store_for_alignment']
            self.did_i_use_custom_warp_matrix_store = self.all_combinations_aligned_he_dict['did_i_use_custom_warp_matrix_store']

            ######################

            return self.all_combinations_aligned_he_dict

        if custom_warp_matrix_store == None:
            did_i_use_custom_warp_matrix_store = 0
            warp_matrix_store = []
        else:
            did_i_use_custom_warp_matrix_store = 1
            warp_matrix_store = custom_warp_matrix_store

        if custom_he_store == None:
            self.used_all_combinations_he_store_3d_for_alignment = self.he_store_3d
            did_i_use_custom_he_store_for_alignment = 0
        else:
            self.used_all_combinations_he_store_3d_for_alignment = custom_he_store
            did_i_use_custom_he_store_for_alignment = 1

        channel_extracted_aligned_he_store = []
        channel_extracted_aligned_he_store_flattened = []
        aligned_all_combinations_BGR_he_image_store = []
        enhanced_correlation_coefficient_store = []

        for i in range(self.num_datasets):
            per_dataset_aligned_BGR_he_image_store = []
            per_dataset_channel_extracted_aligned_he_store = []
            per_dataset_channel_extracted_aligned_he_store_flattened = []
            per_dataset_warp_matrix_store = []
            per_dataset_enhanced_correlation_coefficient_store = []

            for j in range(self.num_dim_reduced_components):

                print("Aligning dataset " + str(i) + ", component " + str(j))

                reference_image_for_alignment = self.normalized_datagrid_store[i][j]
                reference_image_for_alignment_gray = reference_image_for_alignment.copy()  # Since the normalized (between 0 and 255) images mentioned above only have one channel, they are already grayscale images. Therefore, simply assign to the variable 'reference_image_for_alignment_gray'

                image_to_be_aligned = self.used_all_combinations_he_store_3d_for_alignment[i]
                image_to_be_aligned_blue_component, image_to_be_aligned_green_component, image_to_be_aligned_red_component = cv2.split(image_to_be_aligned)
                image_to_be_aligned_gray = cv2.cvtColor(image_to_be_aligned, cv2.COLOR_BGR2GRAY)

                height = reference_image_for_alignment.shape[0]
                width = reference_image_for_alignment.shape[1]

                if did_i_use_custom_warp_matrix_store == 0:
                    warp_mode = cv2.MOTION_HOMOGRAPHY
                    warp_matrix = np.eye(3, 3, dtype=np.float32)

                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-12)

                    if gradient_alignment == 1:
                        plot_gradient_of_image = 0
                        reference_image_gradient = self.get_image_gradient(reference_image_for_alignment_gray, plot_gradient_of_image=plot_gradient_of_image)
                        image_to_be_aligned_gradient = self.get_image_gradient(image_to_be_aligned_gray, plot_gradient_of_image=plot_gradient_of_image)

                        try:
                            (enhanced_correlation_coefficient, warp_matrix) = cv2.findTransformECC(reference_image_gradient, image_to_be_aligned_gradient, warp_matrix, warp_mode, criteria, None, 1)  # 'enhanced_correlation_coefficient' variable is the 'enhanced correlation coefficient' that was maximized during the transform
                        except:
                            enhanced_correlation_coefficient = 0
                            warp_matrix = np.eye(3, 3, dtype=np.float32)
                            print("error at dataset " + str(i) + "component" + str(j))

                    else:  # Align the raw images, and NOT the gradients of the raw images.
                        try:
                            (enhanced_correlation_coefficient, warp_matrix) = cv2.findTransformECC(reference_image_for_alignment_gray, image_to_be_aligned_gray, warp_matrix, warp_mode, criteria, None, 1)  # 'enhanced_correlation_coefficient' variable is the 'enhanced correlation coefficient' that was maximized during the transform
                        except:
                            enhanced_correlation_coefficient = 0
                            warp_matrix = np.eye(3, 3, dtype=np.float32)
                            print("error at dataset " + str(i) + "component" + str(j))

                else:  # Used custom warp matrix, Hence do not have to calculate warp matrices, but simply do the alignment based on this given warp matrix
                    warp_matrix = custom_warp_matrix_store[i][j]
                    enhanced_correlation_coefficient = custom_enhanced_correlation_coefficient_store[i][j]



                aligned_image_blue_component = cv2.warpPerspective(image_to_be_aligned_blue_component, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_image_green_component = cv2.warpPerspective(image_to_be_aligned_green_component, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_image_red_component = cv2.warpPerspective(image_to_be_aligned_red_component, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_image_BGR = cv2.merge((aligned_image_blue_component, aligned_image_green_component, aligned_image_red_component))

                per_dataset_channel_extracted_aligned_he_store.append(np.array(cv2.split(aligned_image_BGR)))
                per_dataset_channel_extracted_aligned_he_store_flattened.append(per_dataset_channel_extracted_aligned_he_store[j].reshape(3, -1))
                per_dataset_aligned_BGR_he_image_store.append(aligned_image_BGR)
                per_dataset_warp_matrix_store.append(warp_matrix)
                per_dataset_enhanced_correlation_coefficient_store.append(enhanced_correlation_coefficient)


            channel_extracted_aligned_he_store.append(per_dataset_channel_extracted_aligned_he_store)
            channel_extracted_aligned_he_store_flattened.append(per_dataset_channel_extracted_aligned_he_store_flattened)
            aligned_all_combinations_BGR_he_image_store.append(per_dataset_aligned_BGR_he_image_store)
            warp_matrix_store.append(per_dataset_warp_matrix_store)
            enhanced_correlation_coefficient_store.append(per_dataset_enhanced_correlation_coefficient_store)

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
                                            'did_i_use_gradient_alignment': gradient_alignment}


        self.all_combinations_aligned_he_dict = all_combinations_aligned_he_dict

        folder_name = self.data_preformatter_kp_object.dim_reduced_object.dataloader_kp_object.global_path_name + '/saved_outputs/he_stained_images/'
        file_name = filename_prefix + 'aligned_he_stained_dict.npy'

        if save_data == 1:
            np.save(folder_name + file_name, self.all_combinations_aligned_he_dict, 'dtype=object')

        return self.all_combinations_aligned_he_dict

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

    def display_combined_aligned_images(self, save_figures=0, plots_fileformat='svg', plots_dpi=600, use_opencv_plotting=0, enable_slider=0):
        """
        @brief Plots a grid of all aligned h&e stained images, aligned against each dim reduced component separately.
            Usage example:
                            save_figures = 1
                            plots_dpi = 600
                            plots_fileformat = 'svg'
                            use_opencv_plotting = 0
                            enable_slider = 0
                            he_stains_kp_object.display_combined_aligned_images(save_figures=save_figures, plots_fileformat=plots_fileformat, plots_dpi=plot_dpi, use_opencv_plotting=use_opencv_plotting, enable_slider=enable_slider)

        @param save_figures: save figures if set to 1
        @param plots_dpi
        @param plots_fileformat
        @param use_opencv_plotting
        @param enable_slider
        """

        if use_opencv_plotting == 0:
            fig, ax = plt.subplots(self.num_datasets, self.num_dim_reduced_components)

            for i in range(self.num_datasets):
                for j in range(self.num_dim_reduced_components):
                    dim_reduced_image_gray = self.normalized_datagrid_store[i][j]
                    zeros = np.zeros(dim_reduced_image_gray.shape, np.uint8)
                    dim_reduced_image_false_BGR = cv2.merge((zeros, dim_reduced_image_gray, zeros))  # False BGR image by assiging a grayscale image to the green channel of an otherwise all-black BGR image
                    dim_reduced_image_false_BGR_weight = 0.5

                    aligned_he_image_BGR = self.all_combinations_aligned_he_dict['aligned_all_combinations_BGR_he_image_store'][i][j]
                    aligned_he_image_BGR_weight = (1 - dim_reduced_image_false_BGR_weight)

                    blended_image = cv2.addWeighted(dim_reduced_image_false_BGR, dim_reduced_image_false_BGR_weight, aligned_he_image_BGR, aligned_he_image_BGR_weight, 0.0)

                    ax[i, j].imshow(blended_image)
                    ax[i, j].xaxis.set_tick_params(length=0, width=0, labelsize=0)
                    ax[i, j].yaxis.set_tick_params(length=0, width=0, labelsize=0)
                    ax[i, j].spines[['bottom', 'right', 'left', 'top']].set_visible(0)
                    ax[i, j].tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
                    ax[i, j].set_ylabel(self.dataset_order[i], fontsize=40)
                    ax[i, j].set_xlabel(self.used_dim_reduction_technique + str(j) + '\n ECC: ' + str(self.all_combinations_aligned_he_dict['enhanced_correlation_coefficient_store'][i][j]), fontsize=40)

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

                    aligned_he_image_BGR = self.all_combinations_aligned_he_dict['aligned_all_combinations_BGR_he_image_store'][i][j]
                    aligned_he_image_BGR_weight = (1 - dim_reduced_image_false_BGR_weight)

                    blended_image = cv2.addWeighted(dim_reduced_image_false_BGR, dim_reduced_image_false_BGR_weight, aligned_he_image_BGR, aligned_he_image_BGR_weight, 0.0)

                    blended_image_store_per_dataset.append(blended_image)
                    label_string_this_dataset = self.dataset_order[i] + ',' + self.used_dim_reduction_technique + str(j) + '\n ECC: ' + str(np.round(self.all_combinations_aligned_he_dict['enhanced_correlation_coefficient_store'][i][j], 2))
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

                aligned_he_image_BGR_for_display = np.array(self.all_combinations_aligned_he_dict['aligned_all_combinations_BGR_he_image_store'], dtype=object)
                false_colored_dim_reduced_datagrid_store_for_display = np.array(self.false_colored_normalized_datagrid_store, dtype=object)

                combined_aligned_he_image_BGR_for_display = self.cv_subplot(aligned_he_image_BGR_for_display, titles=labels_for_display)
                combined_false_colored_dim_reduced_datagrid_store_for_display = self.cv_subplot(false_colored_dim_reduced_datagrid_store_for_display, titles=labels_for_display)

                blended_image = cv2.addWeighted(combined_aligned_he_image_BGR_for_display, 0.25, combined_false_colored_dim_reduced_datagrid_store_for_display, 0.75, 0.0)
                cv2.namedWindow('all_combos_blended_image', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('blended_image', (1000, 1000))
                self.temp_image_to_blend_1 = combined_aligned_he_image_BGR_for_display
                self.temp_image_to_blend_2 = combined_false_colored_dim_reduced_datagrid_store_for_display
                cv2.createTrackbar('slider_kp', 'all_combos_blended_image', 0, 100, self.blending_slider_callback)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

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
                                image = he_stains_kp_object.he_store_3d[0]
                                segmentation_blur_factor = 3
                                segmentation_dilation_factor = 6
                                save_data = 1
                                filename_prefix = 'color_based_segmented_he_store_dict'

                                # Determine the color range of interest in the semi-automatic way and use it to segment the images
                                self_determine_color_range = 1
                                segmented_image = he_stains_kp_object.color_based_segmentation_implementation(image, self_determine_color_range=self_determine_color_range, segmentation_blur_factor=segmentation_blur_factor, segmentation_dilation_factor=segmentation_dilation_factor, save_data=save_data, filename_prefix=filename_prefix)

                                # OR, Load a pre-saved color_range dictionary
                                saved_he_muscle_color_range_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/saved_he_muscle_color_range_dict.npy"
                                roi_and_preprocessing_dict = np.load(saved_he_muscle_color_range_dict_filename, allow_pickle=True)[()]
                                segmented_image = he_stains_kp_object.color_based_segmentation_implementation(image, roi_and_preprocessing_dict=roi_and_preprocessing_dict, self_determine_color_range=self_determine_color_range, segmentation_blur_factor=segmentation_blur_factor, segmentation_dilation_factor=segmentation_dilation_factor, save_data=save_data, filename_prefix=filename_prefix)

        @param image: The image to be segmented
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

    def segment_he_stained_images(self, segmentation_technique, parameter_dict=None, save_data=1, filename_prefix=''):

        """
        @brief Segment the H&E stained images into a given number of segments
            Usage example:
                           segmentation_technique = 'k_means'
                           num_segments = 4
                           parameter_dict = None
                           save_data = 1
                           he_stains_kp_object.segment_he_stained_images(segmentation_technique=segmentation_technique, save_data=save_data, filename_prefix=filename_prefix, num_segments=num_segments, parameter_dict=parameter_dict)

        @param segmentation_technique: Could be 'k_means', 'color_based', '', and ''
        @param save_data: Save results if set to 1
        @param filename_prefix: Prefix for saving the results dictionary
        @param parameter_dict: An optional parameter dictionary that may be required for specific segmentation techniques
        @return segmented_he_store_dict: A dictioonary containing the segmented H&E stained images and metadata.
        """
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
                color_based_roi_and_preprocessing_dict = self.color_based_segmentation_semi_automatic_parameter_determination(color_deterimining_he_image_BGR, parameter_dict = parameter_dict)
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

    def use_warp_matrix_for_alignment(self):

        if self.optimal_warp_matrix_store != None:
            optimal_warp_matrix_store = self.optimal_warp_matrix_store
            optimal_enhanced_correlation_coefficient_store = self.optimal_enhanced_correlation_coefficient_store
        else:
            warp_matrix_store = self.all_combinations_aligned_he_dict['warp_matrix_store']
            enhanced_correlation_coefficient_store = self.all_combinations_aligned_he_dict['enhanced_correlation_coefficient_store']






    def find_the_optimal_alignment(self):
        print("To do")

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

