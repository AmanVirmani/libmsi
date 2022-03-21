from nmf_kp import nmf_kp
from pca_kp import pca_kp
import numpy as np
from data_preformatter_kp import data_preformatter_kp


save_data = 1

########################################################
## The data preformatting class requires the metadata inside the msi_data (or compact_msi_data) objects. Load this into memory
path_to_compact_msi_dataset = "D:/msi_project_data/binned_binsize_1/compact_data_object/compact_msi_data.npy"
compact_msi_data = np.load(path_to_compact_msi_dataset, allow_pickle=True)[()]

########################################################
## The data preformatting class also requires the NMF/PCA data. The output of this (nmf_kp_object) will internally incoorporate the msi_data (or compact_msi_data) object inside it.
saved_nmf_filename = "D:/msi_project_data/saved_outputs/nmf_outputs/modified_truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
nmf_kp_object = nmf_kp(compact_msi_data, saved_nmf_filename=saved_nmf_filename)
dim_reduced_object = nmf_kp_object

## OR:
# saved_pca_filename = "D:/msi_project_data/saved_outputs/pca_outputs/modified_2_no_whitening_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_20_components_pca_dict_tic_normalized.npy"
# pca_kp_object = pca_kp(compact_msi_data, saved_pca_filename=saved_pca_filename)
# dim_reduced_object = pca_kp_object

########################################################
## Initialize the data_preformatter object with the nmf_kp_object
data_preformatter_kp_object = data_preformatter_kp(dim_reduced_object)

########################################################
## Perform the data preformatting
image_patch_reject_percentage = 25
image_patch_window_size = 30
image_patch_overlap = 0
segregate_data_training_percentage = 80
segregate_data_random_select = 1
preformatting_pipeline_filename_prefix = 'test_data_preformatted_pipelined_'

data_preformatter_kp_object.data_preformatting_pipeline(image_patch_reject_percentage=image_patch_reject_percentage,
                                                        image_patch_window_size=image_patch_window_size,
                                                        image_patch_overlap=image_patch_overlap,
                                                        segregate_data_training_percentage=segregate_data_training_percentage,
                                                        segregate_data_random_select=segregate_data_random_select,
                                                        save_data=save_data,
                                                        preformatting_pipeline_filename_prefix=preformatting_pipeline_filename_prefix)
