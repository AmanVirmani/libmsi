import os
import numpy as np
import matplotlib

# matplotlib.use("TkAgg")
from matplotlib.widgets import Slider
from dataloader import DataLoader
import matplotlib.pyplot as plt
import customCmap


if __name__ == "__main__":
    prefix = "pos_msi_bin_"
    data_path = "/mnt/sda/avirmani/data"
    # data_path = '/lab/msi_project/avirmani/data'
    all_paths = [
        "DMSI0047_F893CPH_LipidPos_IMZML/dmsi0047_0_05_v2.pickle",
        "DMSI0048_F894CPH_LipidPos_IMZML/dmsi0048_0_05_v2.pickle",
        "DMSI0053_F895CPH_LipidPos_IMZML/dmsi0053_0_05_v2.pickle",
        "DMSI0054_F896CPH_LipidPos_IMZML/dmsi0054_0_05_v2.pickle",
        "DMSI0045_F885naive_LipidPos_IMZML/dmsi0045_0_05_v2.pickle",
        "DMSI0046_F886naive_LipidPos_IMZML/dmsi0046_0_05_v2.pickle",
        "DMSI0049_F887naive_LipidPos_IMZML/dmsi0049_0_05_v2.pickle",
        "DMSI0068_F888naive_LipidPos_IMZML/dmsi0068_0_05_v2.pickle",
    ]
    binSize = 0.05
    fileNames = [os.path.join(data_path, path) for path in all_paths]
    dataset = DataLoader(fileNames, binSize)

    def plot_dataset(mz):
        for i, ax_i in enumerate(ax.reshape(-1)):
            idx = round((mz-min_mz)/binSize)
            print(idx)
            im = ax_i.imshow(dataset.files[i].imzml_array[:, :, idx],
                        cmap=customCmap.Black2Green,
                        # vmin=0, vmax=0.4
                             ).set_interpolation("nearest")
        if im is not None:
            fig.colorbar(im)

    fig, ax = plt.subplots(1, len(dataset.files))
    print(ax.reshape(-1))
    plt.subplots_adjust(bottom=0.35)
    mz_ax = plt.axes([0.15, 0.1, 0.65, 0.03])
    min_mz = dataset.files[0].min_mz.item()
    max_mz = dataset.files[0].max_mz.item()
    plot_dataset(min_mz)
    mz = Slider(mz_ax, "mz", min_mz, max_mz, min_mz, valstep=binSize)
    mz.on_changed(plot_dataset)
    # fig.colorbar()
    plt.show()
    pass
