import libmsi
import numpy as np
import os
import random
from dataloader import DataLoader
from pyimzml.ImzMLParser import ImzMLParser as read_msi
from sklearn.linear_model import LogisticRegression as LR
import pickle
from nonnegfac.nmf import NMF as newNMF
from dataloader import DataLoader

if __name__ == "__main__":
    data_path = "/mnt/sda/avirmani/data"

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
    split = 1
    nmf_comp = 20
    nmf_fn = "./pos_nmf_20_all_0_05_v2_sklearnNMF.pickle"

    fnames = [os.path.join(data_path, path) for path in all_paths]
    dataset = DataLoader(fnames, binSize, split, store=14000)
    print("dataset_aligned")
    # dataset.saveImzmlData()
    # for file in dataset.files:
        # file.imzml_2d_array[:,:15000]
        # file.get_peaks(filename=file.filename.split('.')[0] + '.svg')

    nmf_data = dataset.perform_nmf(nmf_comp, n_iter=1500)
    with open(nmf_fn, "wb") as fh:
        pickle.dump(nmf_data, fh, pickle.HIGHEST_PROTOCOL)

    exit()
