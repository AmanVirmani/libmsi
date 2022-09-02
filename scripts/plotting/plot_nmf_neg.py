import os

# os.environ["PYTHONPATH"] = "/lab/msi_project/avirmani/libmsi"
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import customCmap


if __name__ == "__main__":
    nmf_fn = (
        "/mnt/sda/avirmani/libmsi/scripts/neg_nmf_20_all_0_05_sklearnNMF_tic.pickle"
    )
    nmf_ordered_fn = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/nmf_outputs/negative_mode_0_05_binsize/for_paper_ordered_according_to_residual_reconstruction_error_for_paper_individual_dataset_[1, 1, 1, 1, 1, 1, 1, 1]_nmf_dict_max_iter_5000_tic_normalized_nmf_outputs_20.npy"
    data_path = "/home/avirmani/projects/aim-hi//data"
    # nmf_fn = "/lab/msi_project/avirmani/libmsi/scripts/pos_nmf_20_all_0_05_v2.pickle"
    # data_path = "/lab/msi_project/avirmani/data"

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

    binSize = 0.05
    split = 1
    fnames = [os.path.join(data_path, path) for path in all_paths]
    dataset = DataLoader(fnames, binSize, split, store=10000)
    # with open(nmf_fn, "rb") as fh:
    # nmf_data = pickle.load(fh)
    nmf_ordered_data = np.load(nmf_ordered_fn, allow_pickle=True)[()]
    ### Take 256 colors from the 'Greens' colormap, and distribute it between 0 and 1.
    greens = cm.get_cmap("Greens", 256)
    greens_map = greens(np.linspace(0, 1, 256))
    ### Modify the 'Greens' colormap.Set the first five colors starting from 0 to pure white
    greens_map[0:5, :] = [1, 1, 1, 1]
    new_greens = ListedColormap(
        greens_map
    )  ### Create a matplotlib colormap object from the newly created list of colors

    fig, ax = plt.subplots(len(fnames), 20, figsize=(15, 15))
    for i, data in enumerate(nmf_data["nmf_data_list"]):
        if i == len(dataset.files):
            break
        msi_data = dataset.files[i].reconstruct_3d_array(data)
        for j in range(data.shape[-1]):
            ax[i, j].imshow(
                msi_data[:, :, j], cmap=customCmap.Black2Green
            ).set_interpolation("none")
            ax[i, j].set_title(
                "Dataset {0}, NMF component {1}".format(i, 1 + j), fontsize=2
            )
    # plt.show()
    plt.savefig("ordered_NMF_data_for_neg_mode_sklearn_tic.png")
    pass
