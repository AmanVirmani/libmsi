import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks
import pandas as pd

if __name__=='__main__':
    # nmf_fn = '/mnt/sda/avirmani/libmsi/scripts/pos_nmf_20_all_0_05_v2_sklearnNMF.pickle'
    nmf_fn = '/lab/msi_project/avirmani/libmsi/scripts/pos_nmf_20_all_0_05_v2_sklearnNMF.pickle'

    with open(nmf_fn, 'rb') as fh:
        data = pickle.load(fh)
    peaks = []
    binSize = 0.05
    min_mz = 399.95
    # fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    for i in range(20):
        tic_data = data['nmf_component_spectra'][i]/sum(data['nmf_component_spectra'][i])
        print('max: {}'.format(np.max(tic_data)))
        peak_mz, prop = find_peaks(tic_data, height=0.03)
        intensities = tic_data[peak_mz]
        peak_mz = peak_mz.astype(np.float64)
        peak_mz *= binSize
        peak_mz += min_mz
        peaks.append({'peak_mz': peak_mz, 'intensities': intensities})
        # ax[0].plot(tic_data)
        # ax[1].plot(peak_mz, intensities)
        # ax[0].set_ylim(0, 0.3)
        # plt.show()
        # plt.pause(3)

    pos_nmf_spectral_peaks = 'pos_nmf_spectral_peaks.csv'
    df = pd.DataFrame(peaks).T
    df.to_csv(pos_nmf_spectral_peaks)
    pass
    # plt.savefig('pos_nmf_spectra.svg')

