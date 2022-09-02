from dataloader import DataLoader
import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from dataloader import DataLoader
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_path = "/home/avirmani/projects/aim-hi/data/"
    all_paths = [
        "DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004.npy",
        "DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.npy",
        "DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.npy",
        "DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.npy",
        "DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.npy",
        "DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006.npy",
        "DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009.npy",
        "DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.npy",
    ]
    binSize = 0.05
    split = 1
    pca_comp = 20
    pca_fn = "./neg_pca_20_all_1_sklearnpca_tic.pickle"

    fnames = [os.path.join(data_path, path) for path in all_paths]
    dataset = DataLoader(fnames, binSize, split, store=10000)
    dataset.perform_pca(pca_comp)
    with open(pca_fn, "wb") as fh:
        pickle.dump(dataset.pca, fh, pickle.HIGHEST_PROTOCOL)

    # with open(pca_fn, "rb") as fh:
    # pca_data = pickle.load(fh)
    data = np.vstack([file.imzml_2d_array for file in dataset.files])
    pca_data = dataset.pca.transform(data)
    dmsi_4 = dataset.files[0].reconstruct_3d_array(
        pca_data[: len(dataset.files[0].coordinates), :]
    )

    plt.imshow(dmsi_4)
    plt.show()
    pass
