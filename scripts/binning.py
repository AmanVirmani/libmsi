import libmsi
import numpy as np
import os,sys
import random
from dataloader import DataLoader
from pyimzml.ImzMLParser import ImzMLParser as read_msi
from sklearn.linear_model import LogisticRegression as LR
from multiprocessing import Process

def bin_data(fn, bs):
    print('binning ' ,fn ,' with binSize: ',bs)
    fs = libmsi.Imzml(fn, bs)
    fs.saveImzmlData()

if __name__ == "__main__":
    data_path = "/home/avirmani/projects/aim-hi/data/"
    all_paths = [ "DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009.imzML",
    "DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.imzML",
    "DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.imzML",
    "DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.imzML",
    "DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004.imzML",
    "DMSI0006_FF886naive_LipidNeg_IMZML/dms0006.imzML",
    "DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.imzML",
    "DMSI0002_F885naive_LipidNeg_IMZML/DMSI0002.imzML"
    ]
    binSize = 0.05

    t_list = []
    for path in all_paths:
        t_list.append(Process(target=bin_data,args=(os.path.join(data_path,
        path), binSize,)))

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()

    pass
