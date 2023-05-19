import libmsi
import numpy as np
import os
import random
from dataloader import DataLoader
from pyimzml.ImzMLParser import ImzMLParser as read_msi
from sklearn.linear_model import LogisticRegression as LR
import pickle

def bin_data(fn, bs):
    fs = libmsi.Imzml(fn, bs)
    fs.saveImzmlData()

if __name__ == "__main__":
    data_path = "/home/avirmani/projects/aim-hi/data/"
    # data_path = "/lab/projects/aim-hi/data/"
    all_paths = ["DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009.imzML",
    "DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.imzML",
    "DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.imzML",
    "DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.imzML",
    "DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004.imzML",
    "DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006.imzML",
    "DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.imzML",
    "DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.imzML"
    ]
    binSize = 0.05
    split = 0.75
    nmf_comp = 20
    nmf_fn = "./nmf_all_0_05_0_05.pickle"
    
    data = DataLoader([os.path.join(data_path, path) for path in all_paths],
    binSize, split)
    print('Data Loaded! Yay!!')
    nmf_data_list = data.perform_nmf(nmf_comp,'all')
    with open(nmf_fn, 'wb') as fh:
        pickle.dump(nmf_data_list, fh, pickle.HIGHEST_PROTOCOL)
    pass
