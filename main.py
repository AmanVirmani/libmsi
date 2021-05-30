import libmsi
import numpy as np
import random
from pyimzml.ImzMLParser import ImzMLParser as read_msi

if __name__ == "__main__":
    cph = "../real_data/DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.imzML"
    naive = "../real_data/DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.imzML"
    msi_cph = libmsi.Imzml(cph, 1)
    msi_cph.saveImzmldata(cph.split('imzML')[0]+'npy')
    msi_naive = libmsi.Imzml(naive, 1)
    msi_naive.saveImzmldata(naive.split('imzML')[0]+'npy')
    # for index in random.sample(range(len(msi.imzml_2d_array)), 10):
    #     msi.plotSpectra(index)
    # for mz in random.sample(range(np.shape(msi.imzml_array)[-1]), 10):
    #     msi.plotMSI(mz)
    # msi_cph.get_peaks()
    # msi_naive.get_peaks()
    pass