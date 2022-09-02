import os
import libmsi
import pickle
import customCmap
import matplotlib.pyplot as plt


def plot_nmf_comp(data, cph_fn, naive_fn):
    binSize = 0.05
    cph_nmf = data['nmf_data_list'][0]
    naive_nmf = data['nmf_data_list'][4]
    cph_data = libmsi.Imzml(cph_fn, binSize)
    naive_data = libmsi.Imzml(naive_fn, binSize)
    cph_img = cph_data.reconstruct_3d_array(cph_nmf)
    naive_img = naive_data.reconstruct_3d_array(naive_nmf)
    for i in range(cph_nmf.shape[-1]):
        plt.imsave(cph_fn.split('.')[0]+'_NMF_{}.svg'.format(i), cph_img[:, :, i], cmap=customCmap.Black2Green) #.set_interpolation(
            # "none"
        # )
        # plt.savefig(cph_fn.split('.')[0]+'_NMF_{}.svg'.format(i))

        plt.imsave(naive_fn.split('.')[0]+'_NMF_{}.svg'.format(i), naive_img[:, :, i], cmap=customCmap.Black2Green) #.set_interpolation(
            # "none"
        # )
        # plt.savefig(naive_fn.split('.')[0]+'_NMF_{}.svg'.format(i))

if __name__=='__main__':
    nmf_neg_fn = '/mnt/sda/avirmani/libmsi/scripts/neg_nmf_20_all_0_05_sklearnNMF.pickle'
    pos_neg_fn = '/mnt/sda/avirmani/libmsi/scripts/pos_nmf_20_all_0_05_v2_sklearnNMF.pickle'

    split = 1

    with open(nmf_neg_fn, 'rb') as fh:
        nmf_neg = pickle.load(fh)

    cph_fn = '/home/avirmani/projects/aim-hi/data/DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004_0_05.pickle'
    naive_fn = '/home/avirmani/projects/aim-hi/data/DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002_0_05.pickle'
    plot_nmf_comp(nmf_neg, cph_fn, naive_fn)

    with open(pos_neg_fn, 'rb') as fh:
        nmf_pos = pickle.load(fh)

    cph_fn = '/mnt/sda/avirmani/data/DMSI0047_F893CPH_LipidPos_IMZML/dmsi0047_0_05_v2.pickle'
    naive_fn = '/mnt/sda/avirmani/data/DMSI0045_F885naive_LipidPos_IMZML/dmsi0045_0_05_v2.pickle'
    plot_nmf_comp(nmf_pos, cph_fn, naive_fn)
