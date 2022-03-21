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

saved_nmf_filename = "D:/msi_project_data/saved_outputs/nmf_outputs/modified_2_truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
nmf_kp_object = nmf_kp(compact_msi_data, saved_nmf_filename=saved_nmf_filename)
dim_reduced_object = nmf_kp_object


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
### Align h&e stained images with all nmf components separately
## Use the internal he_store_3d for the alignments
# save_data = 1
# gradient_alignment = 1
# filename_prefix = 'modified_'
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(save_data=save_data, filename_prefix=filename_prefix, gradient_alignment = 1)

## OR
## Use a custom he store for the alignment
# thresholded_he_store, false_colored_BGR_thresholded_he_store = he_stains_kp_object.threshold_he_images_for_alignment()
# save_data = 1
# gradient_alignment = 0
# filename_prefix = 'custom_warp_thresholded_'
# custom_he_store = false_colored_BGR_thresholded_he_store
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_he_store=custom_he_store, save_data=save_data, filename_prefix=filename_prefix, gradient_alignment=gradient_alignment)

## OR
## Use a custom warp matrix to generate the most optimal alignment
# saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/without_gradient_align_using_segmented_he_segment_1_aligned_he_stained_dict.npy"
# optimally_warped_component_selection_array = [2, 2, 2, 2, 2, 2, 2, 2]  # Example, this would be warp matrix corresponding to nmf2 (the muscular lining)
# custom_warp_matrix_store = he_stains_kp_object.create_custom_warp_matrix_store(saved_all_combinations_aligned_he_dict_filename, optimally_warped_component_selection_array=optimally_warped_component_selection_array)
# save_data = 1
# filename_prefix = 'optimal_alignment_from_segment_1_muscular_lining_'
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(custom_warp_matrix_store=custom_warp_matrix_store, save_data=save_data, filename_prefix=filename_prefix)

## OR
## If a pre-saved file exists, use that instead
# saved_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/thresholded_aligned_he_stained_dict.npy"
# aligned_he_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_he_stained_images(saved_aligned_he_dict_filename = saved_aligned_he_dict_filename)

#########################################################
### Display aligned images

## Display the above generated all combinations of aligned images
# save_figures = 0
# plots_dpi = 600
# plots_fileformat = 'svg'
# use_opencv_plotting = 1
# enable_slider = 1
# slider_inside = 1
# he_stains_kp_object.display_combined_aligned_images(save_figures=save_figures, plots_fileformat=plots_fileformat, plots_dpi=plots_dpi, use_opencv_plotting=use_opencv_plotting, enable_slider=enable_slider, slider_inside=slider_inside)

## OR:

## Display the an externally provided aligned images store
save_figures = 0
plots_dpi = 600
plots_fileformat = 'svg'
use_opencv_plotting = 1
enable_slider = 1
slider_inside = 0
saved_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/nmf_optimally_aligned_he_store_dict.npy"
custom_aligned_he_store_dict_for_display = np.load(saved_aligned_he_dict_filename, allow_pickle=True)[()]
he_stains_kp_object.display_combined_aligned_images(custom_aligned_he_store_dict_for_display=custom_aligned_he_store_dict_for_display, save_figures=save_figures, plots_fileformat=plots_fileformat, plots_dpi=plots_dpi, use_opencv_plotting=use_opencv_plotting, enable_slider=enable_slider, slider_inside=slider_inside)

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
# filename_prefix = 'modified_naive_4_he_'
# color_based_roi_and_preprocessing_dict = np.load("D:/msi_project_data/saved_outputs/he_stained_images/original_he_included_segmented_he_stained_dict.npy", allow_pickle=True)[()]['color_based_roi_and_preprocessing_dict']  # Only useful for color_based segmentation technique
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
### Get a summary of the important metrics of aligned h&e stained images with dim_reduced data (all combinations)

## Display the metrics from an already loaded/calculated alignment.
# all_combinations_alignment_similarity_metrics_dict = he_stains_kp_object.print_alignment_similarity_metrics()

## OR
# saved_all_combinations_aligned_he_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/modified_align_using_segmented_he_segment_2_aligned_he_stained_dict.npy"
# all_combinations_alignment_similarity_metrics_dict = he_stains_kp_object.print_alignment_similarity_metrics(saved_all_combinations_aligned_he_dict_filename=saved_all_combinations_aligned_he_dict_filename)
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
### Align all combinations of dim_reduced components with all segments of a segmented h&e stained dataset.
## Use segmented he stained images internally in the he_stains_kp_object.
# save_data = 1
# gradient_alignment = 1
# filename_prefix = ''
# print_similarity_metrics = 1
# aligned_all_combinations_with_all_he_segments_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_all_he_segments(gradient_alignment=gradient_alignment,  save_data=save_data, filename_prefix=filename_prefix, print_similarity_metrics=print_similarity_metrics)

### OR
## Use an externally provided custom segmented he stained image dictionary and carry on the alignment.
# custom_segmented_he_store_dict = np.load("D:/msi_project_data/saved_outputs/he_stained_images/modified_naive_4_he_segmented_he_stained_dict.npy", allow_pickle=True)[()]
# save_data = 1
# gradient_alignment = 1
# filename_prefix = 'test_all_segment_alignment_'
# print_similarity_metrics = 1
# aligned_all_combinations_with_all_he_segments_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_all_he_segments(gradient_alignment=gradient_alignment, custom_segmented_he_store_dict=custom_segmented_he_store_dict, save_data=save_data, filename_prefix=filename_prefix, print_similarity_metrics=print_similarity_metrics)

### OR
## Do not carry out any new alignment. Simply load an already aligned version.
# saved_aligned_he_segment_dict_filename = "D:/msi_project_data/saved_outputs/he_stained_images/nmf_all_segment_alignment_aligned_all_combinations_with_all_he_segments_dict.npy"
# print_similarity_metrics = 1
# aligned_all_combinations_with_all_he_segments_dict = he_stains_kp_object.align_all_combinations_of_dim_reduced_images_with_all_he_segments(saved_aligned_he_segment_dict_filename=saved_aligned_he_segment_dict_filename, print_similarity_metrics=print_similarity_metrics)

###############################################
### Example 2:
# optimally_warped_dim_reduced_component_and_he_segment_pair_array = [[1, 2],  # segment_1 with nmf/pca 2
#                                                                     [1, 2],  # segment_1 with nmf/pca 2
#                                                                     [1, 2],  # segment_1 with nmf/pca 2
#                                                                     [1, 2],  # segment_1 with nmf/pca 2
#                                                                     [1, 2],  # segment_1 with nmf/pca 2
#                                                                     [1, 2],  # segment_1 with nmf/pca 2
#                                                                     [1, 2],  # segment_1 with nmf/pca 2
#                                                                     [2, 1]]  # segment_2 with nmf/pca 1
# ## Example, cph1-4 and naive1-3 will use the warp matrix corresponding to the alignment of segment 1 with nmf/pca component 2.
# ## For the naive4 dataset which is the last dataset, the warp matrix corresponding to the alignment of segment 2 with nmf/pca 1 will be used.
# save_data = 0
# filename_prefix = 'test_optimal_'
# optimal_alignment = he_stains_kp_object.determine_optimal_he_alignment(optimally_warped_dim_reduced_component_and_he_segment_pair_array, save_data=save_data, filename_prefix=filename_prefix)

###############################################
###############################################
###############################################
### Testing
