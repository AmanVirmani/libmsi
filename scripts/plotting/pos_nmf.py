import os

# os.environ["PYTHONPATH"] = "/lab/msi_project/avirmani/libmsi"
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap


if __name__ == "__main__":
    nmf_fn = "/mnt/sda/avirmani/libmsi/scripts/pos_nmf_20_all_0_05_v2.pickle"
    data_path = "/mnt/sda/avirmani/data"
    # nmf_fn = "/lab/msi_project/avirmani/libmsi/scripts/pos_nmf_20_all_0_05_v2.pickle"
    # data_path = "/lab/msi_project/avirmani/data"

    all_paths = [
        "DMSI0047_F893CPH_LipidPos_IMZML/dmsi0047_0_05_v2.pickle",
        "DMSI0048_F894CPH_LipidPos_IMZML/dmsi0048_0_05_v2.pickle",
        "DMSI0053_F895CPH_LipidPos_IMZML/dmsi0053_0_05_v2.pickle",
        "DMSI0054_F896CPH_LipidPos_IMZML/dmsi0054_0_05_v2.pickle",
        "DMSI0045_F885naive_LipidPos_IMZML/DMSI0045_0_05_v2.pickle",
        "DMSI0046_F886naive_LipidPos_IMZML/dmsi0046_0_05_v2.pickle",
        "DMSI0049_F887naive_LipidPos_IMZML/dmsi0049_0_05_v2.pickle",
        "DMSI0068_F888naive_LipidPos_IMZML/dmsi0068_0_05_v2.pickle",
    ]

    binSize = 0.05
    split = 1
    fnames = [os.path.join(data_path, path) for path in all_paths]
    dataset = DataLoader(fnames, binSize, split)
    with open(nmf_fn, "rb") as fh:
        nmf_data = pickle.load(fh)

    greens = cm.get_cmap("Greens", 256)
    greens_map = greens(
        np.linspace(0, 1, 256)
    )  ### Take 256 colors from the 'Greens' colormap, and distribute it between 0 and 1.
    greens_map[0:5, :] = [
        1,
        1,
        1,
        1,
    ]  ### Modify the 'Greens' colormap.Set the first five colors starting from 0 to pure white
    new_greens = ListedColormap(
        greens_map
    )  ### Create a matplotlib colormap object from the newly created list of colors

    fig, ax = plt.subplots(len(fnames), 20, figsize=(15, 15))
    for i, data in enumerate(nmf_data["nmf_data_list"]):
        if i == len(dataset.files):
            break
        msi_data = dataset.files[i].reconstruct_3d_array(data)
        for j in range(data.shape[-1]):
            ax[i, j].imshow(msi_data[:, :, j], cmap=new_greens).set_interpolation(
                "none"
            )

    plt.savefig("NMF_data_for_pos_mode.svg", format="svg")
