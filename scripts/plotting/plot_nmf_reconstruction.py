# %%
from libmsi import Imzml
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm as norm
import os
import pickle

# %%

if __name__ == "__main__":
    data_path = "/home/avirmani/projects/aim-hi/data/"
    # data_path = "/lab/projects/aim-hi/data/"
    all_paths = [
        "DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004_0_05.pickle",
        "DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005_0_05.pickle",
        "DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008_0_05.pickle",
        "DMSI0011_F896CPH_LipidNegIMZML/dmsi0011_0_05.pickle",
        "DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002_0_05.pickle",
        "DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006_0_05.pickle",
        "DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009_0_05.pickle",
        "DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012_0_05.pickle",
    ]
    # nmf_fn = "/mnt/sda/avirmani/libmsi/scripts/pos_nmf_20_all_0_05_v2_sklearnNMF.pickle"
    nmf_fn = (
        "/mnt/sda/avirmani/libmsi/scripts/neg_nmf_20_all_0_05_sklearnNMF_tic.pickle"
    )
    # %%
    with open(nmf_fn, "rb") as fh:
        data = pickle.load(fh)

    # %%
    norm_arr = np.zeros(
        (len(data["nmf_data_list"]), data["nmf_component_spectra"].shape[0])
    )
    for data_idx in range(norm_arr.shape[0]):
        for nmf_comp_idx in range(norm_arr.shape[1]):
            norm_arr[data_idx][nmf_comp_idx] = norm(
                data["nmf_data_list"][data_idx][:, :nmf_comp_idx]
                @ data["nmf_component_spectra"][:nmf_comp_idx]
            )
# %%
for i in range(8):
    plt.plot(np.arange(20), norm_arr[i])
plt.show()
# %%
norms = norm_arr.mean(axis=0)
# %%
# TODO: sort based on residual error
sorted_indices = np.argsort(norms)[::-1]
# %%
data["nmf_component_spectra"] = data["nmf_component_spectra"][sorted_indices]
for data_idx in range(8):
    data["nmf_data_list"][data_idx] = data["nmf_data_list"][data_idx][:, sorted_indices]

# %%
orig_data = Imzml(os.path.join(data_path, all_paths[0]))
# %%
sorted_nmf_fn = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/nmf_outputs/double_redo_batch_mode_test_negative_mode_0_05_binsize_/ordered_according_to_residual_reconstruction_error_double_redo_batch_mode_test_negative_mode_0_05_binsize_individual_dataset_[1, 1, 1, 1, 1, 1, 1, 1]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
sorted_nmf = np.load(sorted_nmf_fn, allow_pickle=True)
spatial_comp = sorted_nmf['dim_reduced_outputs'][0][0][86613:187244, :]
spectral_comp = sorted_nmf['dim_reduced_outputs'][1][0]
# %%
store = 10000
recon_err = np.zeros(20)
for data_idx in range(1):
    for nmf_comp_idx in range(20):
        recon_err[nmf_comp_idx] = norm(
            orig_data.imzml_2d_array[:, :store]
            - spatial_comp[:,:nmf_comp_idx]
            @ spectral_comp[:nmf_comp_idx]
            # - data["nmf_data_list"][data_idx][:, :nmf_comp_idx]
            # @ data["nmf_component_spectra"][:nmf_comp_idx]
        )
# %%
plt.plot(recon_err)
plt.title("Reconstruction Error vs NMF components")
plt.xlabel("Reconstruction Error")
plt.ylabel("NMF component")
plt.show()
# %%
