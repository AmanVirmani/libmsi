from pca_kp import pca_kp
import numpy as np

###################################################################
## Loading the msi data object that contains all the metadata and stuff.
path_to_compact_msi_dataset = "D:/msi_project_data/binned_binsize_1/compact_data_object/compact_msi_data.npy" #Compact version or full version can be used. If it is the full version, there is no need to specify the path like this. Instead, more stuff is needed.
compact_msi_data = np.load(path_to_compact_msi_dataset, allow_pickle=True)[()]  # Load the compact version of msi data
###################################################################

## Initialize the pca_kp object which can then be used to perform pca using the calc_pca() function.
# pca_kp_object = pca_kp(compact_msi_data)  # Initialize an pca_kp object using the compact msi data. (Could have used a non-compact full version as well)

## OR:  Initialize the pca_kp object with the additional argument of the path to a saved pca dataset
saved_pca_filename = "D:/msi_project_data/saved_outputs/pca_outputs/modified_3_no_whitening_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_20_components_pca_dict_tic_normalized.npy"
pca_kp_object = pca_kp(compact_msi_data, saved_pca_filename=saved_pca_filename)

###################################################################

## Perform pca and if necessary, save the result (This is only necessary if you did NOT specify a pathname to a saved pca dataset during pca_kp class initialization above).
# save_data = 1
# subset_pattern_array = [1, 1, 1, 1, 1, 1, 1, 1]  # Define a specific subset of the dataset on which I need to perform pca on.
# data_subset = pca_kp_object.data_subset_creator(subset_pattern_array=subset_pattern_array)  # Create a data subset according to the above defined pattern
# pca_num_components = 20
# pca_whiten = False
# pca_individual = 0  # Determines if pca should be done on inidividual datasets (1) or on all datasets taken together (1)
# pca_filename_prefix = 'test_script'
# pca_dict = pca_kp_object.calc_pca(num_components=pca_num_components, pca_whiten=pca_whiten, individual=pca_individual, save_data=save_data,filename_prefix=pca_filename_prefix)

###################################################################

## Downsample the calculated pca data if necessary
# save_data = 1
# downsampler_step = 12
# downsampler_prefix = 'downsampled_'
# downsampled_pca_dict = pca_kp_object.pca_downsampler(step=downsampler_step, save_data=save_data, filename_prefix=downsampler_prefix)


###################################################################

## Calculate the reconstruction accuracy for different numbers of pca components used
# max_num_components = 20
# pca_whiten = False
# pca_individual = 0
# subset_pattern_array = [1, 1, 1, 1, 1, 1, 1, 1]
# plot_result = 1
# save_plot = 1
# plot_file_format = 'svg'
# plot_dpi = 600
# save_data = 1
# data_filename_prefix = 'pca_test_num_comp_vs_accuracy_'
# pca_kp_object.explained_variance_vs_num_pca_components_calculator(max_num_components=max_num_components, pca_individual=pca_individual, subset_pattern_array=subset_pattern_array, plot_result=plot_result, save_plot=save_plot, plot_file_format=plot_file_format, plot_dpi=plot_dpi, save_data=save_data, data_filename_prefix=data_filename_prefix)


###################################################################

## Convert aman's pca object to my pca_dict type variable
# aman_pca_filename = "D:/msi_project_data/other/from_aman/pca_20_all_0_05.pickle"
# pca_dict = pca_kp_object.convert_aman_pca_output_to_match_kp_pca_class(aman_pca_filename)

###################################################################

## This code need not be used often. Use only in the rare case where I want to change the structure of a pca dictionary
# old_saved_pca_filename = "D:/msi_project_data/saved_outputs/pca_outputs/modified_3_no_whitening_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_20_components_pca_dict_tic_normalized.npy"
# save_data = 1
# modified_pca_filename_prefix = 'modified_4_'
# modified_pca_dict = pca_kp_object.modify_existing_pca_dict(old_saved_pca_filename, save_data=save_data, modified_pca_filename_prefix=modified_pca_filename_prefix)
