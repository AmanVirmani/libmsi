from pyimzml.ImzMLParser import ImzMLParser as read_msi
import matplotlib.pyplot as plt
import numpy as np

def normalize(intA):
    return np.array(intA)/sum(intA)
if __name__=='__main__':
    f894_fn = '/home/avirmani/projects/aim-hi/data/DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.imzML'
    f896_fn = '/home/avirmani/projects/aim-hi/data/DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.imzML'

    f896 = read_msi(f896_fn)
    for i in range(len(f896.coordinates)):
        mzA, intA = f896.getspectrum(i)
        plt.plot(mzA, normalize(intA))
    plt.savefig('f896.png')