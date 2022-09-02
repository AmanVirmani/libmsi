# %%
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

#%%
fn = "/mnt/sda/avirmani/libmsi/npy_files/pca_optimally_aligned_he_store_dict.npy"
data = np.load(fn, allow_pickle=True)[()]

#%%
data.keys()
# %%
ecc = np.array(data["enhanced_correlation_coefficient_store"])
# %%
pprint(ecc)
# %%
order_nmf_fn = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/nmf_outputs/double_redo_batch_mode_test_negative_mode_0_05_binsize_/ordered_according_to_residual_reconstruction_error_double_redo_batch_mode_test_negative_mode_0_05_binsize_individual_dataset_[1, 1, 1, 1, 1, 1, 1, 1]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
# %%
data = np.load(order_nmf_fn, allow_pickle=True)[()]
# %%
data.keys()
# %%
data["dim_reduced_outputs"][0][0].shape
# %%
data["pixel_count_array"]
# %%
data["dim_reduced_object"]
# %%
