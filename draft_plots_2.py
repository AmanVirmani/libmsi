import cv2
import os
import numpy as np
import pickle
import overlay as ov
import libmsi
import segmentation as sgm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

if __name__ == "__main__":
    # load nmf data
    # nmf_data_path = "/work/projects/aim-hi/data/nmf_20_all_0_05.pickle"
    nmf_data_path = "./nmf_20_all_0_05.pickle"
    print("start unpacking nmf data")
    nmf_data = pickle.load(open(nmf_data_path, "rb"))
    print("finished unpacking nmf data")

    # load coordinate list -> convert to 3d image
    data_dir = "/home/avirmani/projects/aim-hi/data/"
    imzml_paths = [
        # "DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005_0_05.pickle",
        # "DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004_0_05.pickle",
        # "DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008_0_05.pickle",
        # "DMSI0011_F896CPH_LipidNegIMZML/dmsi0011_0_05.pickle",
        # "DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002_0_05.pickle",
        # "DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006_0_05.pickle",
        # "DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009_0_05.pickle",
        # "DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012_0_05.pickle",
        "DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.npy",
        "DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004.npy",
        "DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.npy",
        "DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.npy",
        "DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.npy",
        "DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006.npy",
        "DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009.npy",
        "DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.npy",
    ]

    ### Creating my custom colormaps
    reds = cm.get_cmap("Reds", 256)
    reds_map = reds(
        np.linspace(0, 1, 256)
    )  ### Take 256 colors from the 'reds' colormap, and distribute it between 0 and 1.
    reds_map[0:5, :] = [
        1,
        1,
        1,
        1,
    ]  ### Modify the 'reds' colormap.Set the first five colors starting from 0 to pure white
    new_reds = ListedColormap(
        reds_map
    )  ### Create a matplotlib colormap object from the newly created list of colors

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

    blues = cm.get_cmap("Blues", 256)
    blues_map = blues(
        np.linspace(0, 1, 256)
    )  ### Take 256 colors from the 'Blues' colormap, and distribute it between 0 and 1.
    blues_map[0:5, :] = [
        1,
        1,
        1,
        1,
    ]  ### Modify the 'Blues' colormap.Set the first five colors starting from 0 to pure white
    new_blues = ListedColormap(
        blues_map
    )  ### Create a matplotlib colormap object from the newly created list of colors
    ###

    cmap_array_spatial = [new_reds, new_greens, new_blues]
    cmap_array_spectra = ["r", "g", "b"]

    fig, axes = plt.subplots(len(imzml_paths), 20, figsize=(15, 15))
    for i, imzml_path in enumerate(imzml_paths):
        imzml = libmsi.Imzml(os.path.join(data_dir, imzml_path))
        nmf_2d_data = nmf_data["nmf_data_list"][i]
        assert len(imzml.coordinates) == len(nmf_2d_data)
        nmf_3d_data = imzml.reconstruct_3d_array(nmf_2d_data)
        nmf_components = nmf_data["components_spectra"].transpose()
        starting_mz = 640

        for j in range(len(nmf_components)):
            axes[i, j].imshow(
                nmf_3d_data[:, :, j], cmap=cmap_array_spatial[1]
            ).set_interpolation("none")
    plt.show()
    # plt.savefig()

    # fig, axes = plt.subplots(4, 5)
    # for i in range(len(nmf_components)):
    #     axes[int(i/5), i%5].plot(np.arange(starting_mz, starting_mz+nmf_components.shape[-1]), nmf_components[i], cmap_array_spectra[1], linewidth=1)
    # plt.show()
    pass
