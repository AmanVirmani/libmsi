############################################
import numpy as np
from nmf_kp import nmf_kp
from pca_kp import pca_kp
import numpy as np
from data_preformatter_kp import data_preformatter_kp
from he_stains_kp import he_stains_kp
import pandas as pd
import cv2
import matplotlib.pyplot as plt

path_to_compact_msi_dataset = "D:/msi_project_data/binned_binsize_1/compact_data_object/compact_msi_data.npy"
compact_msi_data = np.load(path_to_compact_msi_dataset, allow_pickle=True)[()]

saved_pca_filename = "D:/msi_project_data/saved_outputs/pca_outputs/modified_3_no_whitening_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_20_components_pca_dict_tic_normalized.npy"
pca_kp_object = pca_kp(compact_msi_data, saved_pca_filename=saved_pca_filename)
dim_reduced_object = pca_kp_object

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
### Align h&e stained images with all PCA components separately
## Use the internal he_store_3d for the alignments
# save_data = 1
# gradient_alignment = 1
# filename_prefix = 'pca_'
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(save_data=save_data, filename_prefix=filename_prefix, gradient_alignment = 1)

## OR
## Use a custom he store for the alignment
# thresholded_he_store, false_colored_BGR_thresholded_he_store = he_stains_kp_object.threshold_he_images_for_alignment()
# save_data = 1
# gradient_alignment = 0
# filename_prefix = 'pca_custom_warp_thresholded_'
# custom_he_store = false_colored_BGR_thresholded_he_store
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_he_store=custom_he_store, save_data=save_data, filename_prefix=filename_prefix, gradient_alignment=gradient_alignment)

## OR
## Use a custom warp matrix to generate the most optimal alignment
# saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/pca_with_gradient_align_using_segmented_he_segment_0_aligned_he_stained_dict.npy"
# # optimally_warped_component_selection_array = [1, 1, 1, 1, 1, 1, 1, 1]  # Example, this would be warp matrix corresponding to PCA1 aligned with segment_2
# optimally_warped_component_selection_array = [0, 0, 0, 0, 0, 0, 0, 0]  # Example, this would be warp matrix corresponding to PCA1 aligned with segment_2
#
# custom_warp_matrix_store = he_stains_kp_object.create_custom_warp_matrix_store(saved_all_combinations_aligned_he_dict_filename, optimally_warped_component_selection_array=optimally_warped_component_selection_array)
# save_data = 1
# filename_prefix = 'with_gradient_pca_optimal_alignment_from_segment_0_and_pca_0_based_warp_matrix'
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_warp_matrix_store=custom_warp_matrix_store, save_data=save_data, filename_prefix=filename_prefix)

## OR
## If a pre-saved file exists, use that instead
# saved_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/pca_thresholded_aligned_he_stained_dict.npy"
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(saved_aligned_he_dict_filename = saved_aligned_he_dict_filename)

#########################################################
## Display the above generated all combinations of aligned images
# save_figures = 0
# plots_dpi = 600
# plots_fileformat = 'svg'
# use_opencv_plotting = 1
# enable_slider = 1
# he_stains_kp_object.display_combined_aligned_images(save_figures=save_figures, plots_fileformat=plots_fileformat, plots_dpi=plots_dpi, use_opencv_plotting=use_opencv_plotting, enable_slider=enable_slider)

## OR:
## Load some pre-existing aligned image containing dictionary, and use it for display.
# saved_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/with_gradient_pca_optimal_alignment_from_segment_0_and_pca_0_based_warp_matrix_optimally_aligned_he_stained_dict.npy"
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(saved_aligned_he_dict_filename=saved_aligned_he_dict_filename)
# save_figures = 0
# plots_dpi = 600
# plots_fileformat = 'png'
# use_opencv_plotting = 1
# enable_slider = 1
# slider_inside = 1
# he_stains_kp_object.display_combined_aligned_images(save_figures=save_figures, plots_fileformat=plots_fileformat, plots_dpi=plots_dpi, use_opencv_plotting=use_opencv_plotting, enable_slider=enable_slider, slider_inside=slider_inside)


################################################
#### Segment h&e stained images
### 1. Kmeans
# segmentation_technique = 'k_means'
# parameter_dict = {'num_segments': 4}
# save_data = 1
# filename_prefix = 'pca_'
# segmented_he_store_dict = he_stains_kp_object.segment_he_stained_images(segmentation_technique=segmentation_technique, parameter_dict=parameter_dict,  save_data=save_data, filename_prefix=filename_prefix)

### 2. Colorbased
##  Externally provide ROIs-color limits and other processing factors
# segmentation_technique = 'color_based'
# save_data = 1
# filename_prefix = 'pca_original_he_included_'
# color_based_roi_and_preprocessing_dict = np.load("D:/msi_project_data/saved_outputs/he_stained_images/pca_segmented_he_stained_dict.npy", allow_pickle=True)[()]['color_based_roi_and_preprocessing_dict']  # Only useful for color_based segmentation technique
# parameter_dict = {'color_based_roi_and_preprocessing_dict': color_based_roi_and_preprocessing_dict,
#                   'self_determine_color_range': 0,
#                   'segmented_in_what_color_space': 'rgb',
#                   'num_segments': 4}
# segmented_he_store_dict = he_stains_kp_object.segment_he_stained_images(segmentation_technique=segmentation_technique, parameter_dict=parameter_dict, save_data=save_data, filename_prefix=filename_prefix)

## OR
## Determine ROIs-color limits and other processing factors semi-automatically
# segmentation_technique = 'color_based'
# save_data = 1
# filename_prefix = 'pca_'
# parameter_dict = {'self_determine_color_range': 1,
#                   'segmented_in_what_color_space': 'rgb',
#                   'num_segments': 4,
#                   'roi_and_preprocessing_dict': None}
#
# segmented_he_store_dict = he_stains_kp_object.segment_he_stained_images(segmentation_technique=segmentation_technique, parameter_dict=parameter_dict,  save_data=save_data, filename_prefix=filename_prefix)


################################################
### Get a summary of the important metrics of aligned h&e stained images with dim_reduced data (all combinations)

## Display the metrics from an already loaded/calculated alignment.
# all_combinations_alignment_similarity_metrics_dict = he_stains_kp_object.print_alignment_similarity_metrics()

## OR
## Load an existing dataset, and display the metrics from there.
# for i in range(5):
#     saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/pca_with_gradient_align_using_segmented_he_segment_" + str(i) + "_aligned_he_stained_dict.npy"
#     print("Similarity metrics for segment: " + str(i))
#     all_combinations_alignment_similarity_metrics_dict = he_stains_kp_object.print_alignment_similarity_metrics(saved_all_combinations_aligned_he_dict_filename=saved_all_combinations_aligned_he_dict_filename)
#     print("\n")
################################################
## Modify an existing saved all_combinations_aligned_he_image_dict dictionary
# old_all_combinations_aligned_he_image_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/pca_all_combinations_aligned_he_stained_dict.npy"
# save_data = 1
# modified_all_combinations_aligned_he_image_dict_filename_prefix = 'pca_modified_'
# modified_all_combinations_aligned_he_image_dict = he_stains_kp_object.modify_existing_all_combinations_aligned_he_image_dict(old_all_combinations_aligned_he_image_dict_filename, save_data=save_data, modified_all_combinations_aligned_he_image_dict_filename_prefix=modified_all_combinations_aligned_he_image_dict_filename_prefix)

###############################################
###############################################
# Example codes
###############################################
###############################################
### Example 1: Use segmented he stained images loaded from a saved file, and align those to the dim_reduced components
# color_based_segmented_he_dict = np.load("D:/msi_project_data/saved_outputs/he_stained_images/original_he_included_segmented_he_stained_dict.npy", allow_pickle=True)[()]
# segmented_he_store = color_based_segmented_he_dict['segmented_he_store']
# segmented_he_store_np = np.array(segmented_he_store, dtype=object)
# save_data = 1
# gradient_alignment = 1
# for i in range(len(segmented_he_store[0])):
#     filename_prefix = 'pca_with_gradient_align_using_segmented_he_segment_' + str(i) + '_'
#     custom_he_store = segmented_he_store_np[:, i]
#     aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_he_store=custom_he_store, save_data=save_data, filename_prefix=filename_prefix, gradient_alignment=gradient_alignment)

###############################################
### Example 2:


###############################################
###############################################
###############################################
### Testing
