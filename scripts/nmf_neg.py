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
    binSize = 0.05
    split = 1
    nmf_comp = 20
    nmf_fn = "./neg_nmf_20_all_0_05_sklearnNMF_tic.pickle"

    fnames = [os.path.join(data_path, path) for path in all_paths]
    # dataset = DataLoader(fnames, binSize, split)
    dataset = DataLoader(fnames, binSize, split, store=10000)
    # for file in dataset.files:
    #     file.get_peaks(filename=file.filename.replace('pickle', 'png'))
    # exit()
    # nmf_data = dataset.perform_nmf(nmf_comp, n_iter=1)
    nmf_data = dataset.perform_nmf(nmf_comp, n_iter=10000)
    with open(nmf_fn, "wb") as fh:
        pickle.dump(nmf_data, fh, pickle.HIGHEST_PROTOCOL)

    exit()

    # data_list = []
    # for i, path in enumerate(all_paths):
    #     with open(os.path.join(data_path, path), 'rb') as fh:
    #         fdata = pickle.load(fh)['imzml_array']
    #         fdata_2d = fdata[~np.all(fdata == 0, axis=-1)]
    #         data_list.append(fdata_2d)
    #     print('loaded data_', i)
    # data = np.vstack(data_list)
    # print('Data Loaded! Yay!!')
    #
    # nmf = newNMF()
    # X_transformed, components, info = nmf.run(data, nmf_comp, max_iter=150,
    # verbose=5)
    # print('NMF done!')
    # with open(nmf_fn, 'wb') as fh:
    #     save_dict = {'nmf_data_list': X_transformed,
    #     'components_spectra':components}
    #     pickle.dump(save_dict, fh, pickle.HIGHEST_PROTOCOL)

    # nmf_data_list = []
    # start_idx = 0
    # for i in range(len(data_list)):
    #     nmf_data_list.append(X_transformed[start_idx:start_idx+len(data_list[i])])
    #     start_idx += len(data_list[i])

    # with open(nmf_fn, 'wb') as fh:
    #     save_dict = {'nmf_data_list': nmf_data_list,
    #     'components_spectra':components}
    #     pickle.dump(save_dict, fh, pickle.HIGHEST_PROTOCOL)
    # pass
