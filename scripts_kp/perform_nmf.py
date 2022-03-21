from nmf_kp import nmf_kp
import numpy as np

###################################################################
## Loading the msi data object that contains all the metadata and stuff.
path_to_compact_msi_dataset = "D:/msi_project_data/binned_binsize_1/compact_data_object/compact_msi_data.npy" #Compact version or full version can be used. If it is the full version, there is no need to specify the path like this. Instead, more stuff is needed.
compact_msi_data = np.load(path_to_compact_msi_dataset, allow_pickle=True)[()]  # Load the compact version of msi data
###################################################################

## Initialize the nmf_kp object which can then be used to perform NMF using the calc_nmf() function.
# nmf_kp_object = nmf_kp(compact_msi_data)  # Initialize an nmf_kp object using the compact msi data. (Could have used a non-compact full version as well)

## OR:  Initialize the nmf_kp object with the additional argument of the path to a saved nmf dataset
saved_nmf_filename = "D:/msi_project_data/saved_outputs/nmf_outputs/modified_2_truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
nmf_kp_object = nmf_kp(compact_msi_data, saved_nmf_filename=saved_nmf_filename)

###################################################################

## Perform nmf and if necessary, save the result (This is only necessary if you did NOT specify a pathname to a saved nmf dataset during nmf_kp class initialization above).
# save_data = 1
# subset_pattern_array = [1, 1, 1, 1, 1, 1, 1, 1]  # Define a specific subset of the dataset on which I need to perform nmf on.
# data_subset = nmf_kp_object.data_subset_creator(subset_pattern_array=subset_pattern_array)  # Create a data subset according to the above defined pattern
# nmf_num_components = 20
# nmf_num_iter = 50
# nmf_individual = 0  # Determines if NMF should be done on inidividual datasets (1) or on all datasets taken together (1)
# nmf_filename_prefix = 'test_script'
# nmf_dict = nmf_kp_object.calc_nmf(num_components=nmf_num_components, num_iter=nmf_num_iter, individual=nmf_individual, save_data=save_data,filename_prefix=nmf_filename_prefix)

###################################################################

## Downsample the calculated NMF data if necessary
# save_data = 1
# downsampler_step = 12
# downsampler_prefix = 'downsampled_'
# downsampled_nmf_dict = nmf_kp_object.nmf_downsampler(step=downsampler_step, save_data=save_data, filename_prefix=downsampler_prefix)

###################################################################

## Calculate the reconstruction accuracy for different numbers of nmf components used
# max_num_components = 5
# nmf_num_iter = 50
# nmf_individual = 0
# subset_pattern_array = [1, 1, 1, 1, 1, 1, 1, 1]
# plot_result = 1
# save_plot = 1
# plot_file_format = 'svg'
# plot_dpi = 600
# save_data = 1
# data_filename_prefix = 'test_num_comp_vs_accuracy_'
# nmf_kp_object.reconstruction_error_vs_num_nmf_components_calculator(max_num_components=max_num_components, nmf_num_iter=nmf_num_iter, nmf_individual=nmf_individual, subset_pattern_array=subset_pattern_array, plot_result=plot_result, save_plot=save_plot, plot_file_format=plot_file_format, plot_dpi=plot_dpi, save_data=save_data, data_filename_prefix=data_filename_prefix)


###################################################################

## Convert aman's nmf object to my nmf_dict type variable
# aman_nmf_filename = "D:/msi_project_data/other/from_aman/nmf_20_all_0_05.pickle"
# save_data = 1
# filename_prefix = 'aman_0.05_'
# nmf_dict = nmf_kp_object.convert_aman_nmf_output_to_match_kp_nmf_class(aman_nmf_filename, save_data=save_data, filename_prefix=filename_prefix)

###################################################################

## This code need not be used often. Use only in the rare case where I want to change the structure of an nmf dictionary
# old_saved_nmf_filename = "D:/msi_project_data/saved_outputs/nmf_outputs/modified_truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
# save_data = 1
# modified_nmf_filename_prefix = 'modified_2_'
# modified_nmf_dict = nmf_kp_object.modify_existing_nmf_dict(old_saved_nmf_filename, save_data=save_data, modified_nmf_filename_prefix=modified_nmf_filename_prefix)
