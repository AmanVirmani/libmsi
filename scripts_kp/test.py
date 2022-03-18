############################################
import numpy as np
from nmf_kp import nmf_kp
from pca_kp import pca_kp
import numpy as np
from data_preformatter_kp import data_preformatter_kp
from he_stains_kp import he_stains_kp
import cv2
import matplotlib.pyplot as plt

path_to_compact_msi_dataset = "D:/msi_project_data/binned_binsize_1/compact_data_object/compact_msi_data.npy"
compact_msi_data = np.load(path_to_compact_msi_dataset, allow_pickle=True)[()]

saved_nmf_filename = "D:/msi_project_data/saved_outputs/nmf_outputs/modified_2_truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
nmf_kp_object = nmf_kp(compact_msi_data, saved_nmf_filename=saved_nmf_filename)
dim_reduced_object = nmf_kp_object

## OR:
# saved_pca_filename = "D:/msi_project_data/saved_outputs/pca_outputs/modified_2_no_whitening_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_20_components_pca_dict_tic_normalized.npy"
# pca_kp_object = pca_kp(compact_msi_data, saved_pca_filename=saved_pca_filename)
# dim_reduced_object = pca_kp_object

data_preformatter_kp_object = data_preformatter_kp(dim_reduced_object)
save_data = 0
image_patch_reject_percentage = 25
image_patch_window_size = 30
image_patch_overlap = 0
segregate_data_training_percentage = 80
segregate_data_random_select = 1
preformatting_pipeline_filename_prefix = 'test_data_preformatted_pipelined_'
data_preformatter_kp_object.data_preformatting_pipeline(image_patch_reject_percentage=image_patch_reject_percentage, image_patch_window_size=image_patch_window_size, image_patch_overlap=image_patch_overlap, segregate_data_training_percentage=segregate_data_training_percentage, segregate_data_random_select=segregate_data_random_select, save_data=save_data, preformatting_pipeline_filename_prefix=preformatting_pipeline_filename_prefix)

#########################################################
## Initialize the he_stains_kp class
he_stains_kp_object = he_stains_kp(data_preformatter_kp_object)

#########################################################
## Align h&e stained images with al nmf components separately
# save_data = 1
# filename_prefix = 'test_'
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(save_data=save_data, filename_prefix=filename_prefix)

## OR

## If a presaed file exists, use that instead
saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/all_combinations_aligned_he_stained_dict.npy"
aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(saved_all_combinations_aligned_he_dict_filename = saved_all_combinations_aligned_he_dict_filename)

#########################################################
## Display the above generated all combinations of aligned images
save_figures = 0
plots_dpi = 600
plots_fileformat = 'svg'
use_opencv_plotting = 1
enable_slider = 1
he_stains_kp_object.display_combined_aligned_images(save_figures=save_figures, plots_fileformat=plots_fileformat, plots_dpi=plots_dpi, use_opencv_plotting=use_opencv_plotting, enable_slider=enable_slider)

#########################################################

# array_to_be_normalized = data_preformatter_kp_object.datagrid_store_dict['datagrid_store']
# range_min_value = 0
# range_max_value = 255
# normalized_datagrid_store = he_stains_kp_object.normalize_3d_datagrid_store_to_given_range(range_min_value, range_max_value)
# channel_extracted_he_store, channel_extracted_he_store_flattened, he_store_3d = he_stains_kp_object.load_he_stained_images()
#
# save_data = 0
# filename_prefix = ''
# saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/all_combinations_aligned_he_stained_dict.npy"
# he_stains_kp_object = he_stains_kp()
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(saved_all_combinations_aligned_he_dict_filename=saved_all_combinations_aligned_he_dict_filename, save_data=save_data, filename_prefix=filename_prefix)

# image_1 = he_stains_kp_object.normalized_datagrid_store[7][0]
# zeros = np.zeros(image_1.shape, np.uint8)
# image_1_false_bgr = cv2.merge((zeros, image_1, zeros))  # False BGR image by assiging a grayscale image to the green channel of an otherwise all-black BGR image
# image_2 = he_stains_kp_object.he_store_3d[7]
# image_1_weight = 0.5
# image_2_weight = (1 - image_1_weight)
# plot_result = 1
# enable_interactive_slider = 1
# blended_image = he_stains_kp_object.overlay_and_blend_two_images(image_1_false_bgr, image_2, image_1_weight, image_2_weight, plot_result=plot_result, enable_interactive_slider=enable_interactive_slider)

################################################
# save_figures = 1
# he_stains_kp_object.display_combined_aligned_images(save_figures=save_figures)


################################################
## Modify an existing saved all_combinations_aligned_he_image_dict dictionary

# old_all_combinations_aligned_he_image_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/all_combinations_aligned_he_stained_dict.npy"
# save_data = 1
# modified_all_combinations_aligned_he_image_dict_filename_prefix = 'modified_'
# modified_all_combinations_aligned_he_image_dict = he_stains_kp_object.modify_existing_all_combinations_aligned_he_image_dict(old_all_combinations_aligned_he_image_dict_filename, save_data=save_data, modified_all_combinations_aligned_he_image_dict_filename_prefix=modified_all_combinations_aligned_he_image_dict_filename_prefix)
