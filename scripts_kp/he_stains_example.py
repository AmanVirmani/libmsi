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
### Align h&e stained images with al nmf components separately
## Use the internal he_store_3d for the alignments
# save_data = 1
# filename_prefix = 'modified_'
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(save_data=save_data, filename_prefix=filename_prefix)

## OR
## Use a custom he store for the alignment
# thresholded_he_store, false_colored_BGR_thresholded_he_store = he_stains_kp_object.threshold_he_images_for_alignment()
# save_data = 1
# filename_prefix = 'custom_warp_thresholded_'
# custom_he_store = false_colored_BGR_thresholded_he_store
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_he_store=custom_he_store, save_data=save_data, filename_prefix=filename_prefix)

## OR
## Use a custom warp matrix and  enhanced correlation score stores for alignment
# saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/custom_color_segmented_warp_thresholded_aligned_he_stained_dict.npy"
# saved_thresholded_aligned_he_store_dict = np.load(saved_all_combinations_aligned_he_dict_filename, allow_pickle=True)[()]
# custom_warp_matrix_store = saved_thresholded_aligned_he_store_dict['warp_matrix_store']
# custom_enhanced_correlation_coefficient_store = saved_thresholded_aligned_he_store_dict['enhanced_correlation_coefficient_store']
# save_data = 1
# filename_prefix = 'custom_warp_thresholded_'
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_warp_matrix_store=custom_warp_matrix_store, custom_enhanced_correlation_coefficient_store=custom_enhanced_correlation_coefficient_store, save_data=save_data, filename_prefix=filename_prefix)

## OR
## If a pre-saved file exists, use that instead
# saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/thresholded_aligned_he_stained_dict.npy"
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(saved_all_combinations_aligned_he_dict_filename = saved_all_combinations_aligned_he_dict_filename)

#########################################################
# Display the above generated all combinations of aligned images
# save_figures = 0
# plots_dpi = 600
# plots_fileformat = 'svg'
# use_opencv_plotting = 1
# enable_slider = 1
# he_stains_kp_object.display_combined_aligned_images(save_figures=save_figures, plots_fileformat=plots_fileformat, plots_dpi=plots_dpi, use_opencv_plotting=use_opencv_plotting, enable_slider=enable_slider)

################################################
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

################################################
## Modify an existing saved all_combinations_aligned_he_image_dict dictionary
# old_all_combinations_aligned_he_image_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/all_combinations_aligned_he_stained_dict.npy"
# save_data = 1
# modified_all_combinations_aligned_he_image_dict_filename_prefix = 'modified_'
# modified_all_combinations_aligned_he_image_dict = he_stains_kp_object.modify_existing_all_combinations_aligned_he_image_dict(old_all_combinations_aligned_he_image_dict_filename, save_data=save_data, modified_all_combinations_aligned_he_image_dict_filename_prefix=modified_all_combinations_aligned_he_image_dict_filename_prefix)

###############################################
###############################################
# Example codes
###############################################
###############################################
### Example 1: Use segmented he stained images loaded from a saved file, and align those to the dim_reduced components
color_based_segmented_he_dict = np.load("D:/msi_project_data/saved_outputs/he_stained_images/original_he_included_segmented_he_stained_dict.npy", allow_pickle=True)[()]
segmented_he_store = color_based_segmented_he_dict['segmented_he_store']
segmented_he_store_np = np.array(segmented_he_store, dtype=object)
save_data = 1
for i in range(len(segmented_he_store[0])):
    filename_prefix = 'align_using_segmented_he_segment_' + str(i) + '_'
    custom_he_store = segmented_he_store_np[:, i]
    aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_he_store=custom_he_store, save_data=save_data, filename_prefix=filename_prefix)

###############################################


## Example 2:  Color segment, and use those segments to generate the warp matrices necessary to align the H&E stained images.
# segmentation_technique = 'color_based' #'k_means' #'color_based'
# num_segments = 4
# param_dict = None
# segmented_he_store = he_stains_kp_object.segment_he_stained_images(segmentation_technique=segmentation_technique, num_segments=num_segments, param_dict=param_dict)

# thresholded_he_store, false_colored_BGR_thresholded_he_store = he_stains_kp_object.threshold_he_images_for_alignment()
# save_data = 1
# filename_prefix = 'custom_color_segmented_warp_thresholded_'
# custom_he_store = segmented_he_store
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_he_store=custom_he_store, save_data=save_data, filename_prefix=filename_prefix)


###############################################
###############################################
###############################################
### Testing
