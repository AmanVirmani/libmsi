import os
import matplotlib.pyplot as plt
from dataloader import DataLoader
import numpy as np

if __name__ == "__main__":
    data_path = "/home/kasun/aim_hi_project_kasun/data/"
    cph_paths = ["DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004.npy", "DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.npy",
                 "DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.npy", "DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.npy"]
    naive_paths = ["DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.npy", "DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006.npy",
                   "DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009.npy", "DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.npy"]

    msi_cph = DataLoader([os.path.join(data_path, cph) for cph in cph_paths])
    msi_naive = DataLoader([os.path.join(data_path, naive) for naive in naive_paths])

    # # cph_recon_err = np.load('cph_recon_err_5_30_2.npy', allow_pickle=True)
    cph_data = np.vstack([file.imzml_2d_array for file in msi_cph.train_data])[:, :500]
    cph_norm = np.linalg.norm(cph_data)
    cph_recon_err = []
    cmp_list = np.arange(1, 5)
    for i in cmp_list:
        cph_nmf_data = msi_cph.perform_nmf(i, 'all', cph_data)
        cph_recon_err.append(msi_cph.nmf.reconstruction_err_/cph_norm)
    cph_recon_err = np.hstack([cph_recon_err, np.load('cph_recon_err_5_30_2.npy', allow_pickle=True)])
    # np.save('norm_cph_recon_err_5_30_2_clipped.npy', cph_recon_err)
    # plt.plot(cmp_list, cph_recon_err, 'b-')
    # plt.title("Normalized Reconstruction error vs NMF components")
    # plt.xlabel('No. of NMF components')
    # plt.ylabel('Reconstruction error')
    # plt.savefig('norm_nmf_cph_plt_5_30_clipped.png')
    # plt.close()

    # # naive_recon_err = np.load('naive_recon_err_5_30_2.npy', allow_pickle=True)
    naive_data = np.vstack([file.imzml_2d_array for file in msi_naive.train_data])[:, :500]
    naive_norm = np.linalg.norm(naive_data)
    naive_recon_err = []
    # cmp_list = np.arange(5, 30, 2)
    for i in cmp_list:
        naive_nmf_data = msi_naive.perform_nmf(i, 'all', naive_data)
        naive_recon_err.append(msi_naive.nmf.reconstruction_err_/naive_norm)
    naive_recon_err = np.hstack([naive_recon_err, np.load('naive_recon_err_5_30_2.npy', allow_pickle=True)])
    # np.save('norm_naive_recon_err_5_30_2_clipped.npy', naive_recon_err)
    # plt.figure()
    # plt.plot(cmp_list, naive_recon_err, 'b-')
    # plt.title("Normalized Reconstruction error vs NMF components")
    # plt.xlabel('No. of NMF components')
    # plt.ylabel('Reconstruction error')
    # plt.savefig('norm_nmf_naive_plt_5_30_clipped.png')
    # plt.close()

    cmp_list = np.hstack([np.arange(1,5), np.arange(5, 30, 2)])
    # cph_recon_err = np.load('norm_cph_recon_err_5_30_2_clipped.npy', allow_pickle=True)
    # naive_recon_err = np.load('norm_naive_recon_err_5_30_2_clipped.npy', allow_pickle=True)
    plt.figure()
    plt.plot(cmp_list, naive_recon_err, 'b-')
    plt.plot(cmp_list, cph_recon_err, 'r-')
    plt.legend(['naive', 'cph'])
    plt.title("Normalized Reconstruction error vs NMF components")
    plt.xlabel('No. of NMF components')
    plt.ylabel('Reconstruction error')
    plt.savefig('norm_nmf_err_plt_1_30_clipped.png')
    plt.close()
