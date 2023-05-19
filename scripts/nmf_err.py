import os
import matplotlib.pyplot as plt
from dataloader import DataLoader
import numpy as np

if __name__ == "__main__":
    data_path = "/home/avirmani/projects/aim-hi/data/"
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

    msi = DataLoader(
        [os.path.join(data_path, cph) for cph in all_paths], split_ratio=1, store=10000
    )

    # # cph_recon_err = np.load('cph_recon_err_5_30_2.npy', allow_pickle=True)
    cph_data = np.vstack([file.imzml_2d_array for file in msi.files])
    cph_norm = np.linalg.norm(cph_data)
    cph_recon_err = []
    cmp_list = range(1, 5)
    for i in cmp_list:
        print("Computing NMF with {} no. of components".format(i))
        cph_nmf_data = msi.perform_nmf(i, "all", cph_data, n_iter=1000)
        cph_recon_err.append(msi.nmf.reconstruction_err_ / cph_norm)

    cmp_list = range(5, 30, 2)
    for i in cmp_list:
        print("Computing NMF with {} no. of components".format(i))
        cph_nmf_data = msi.perform_nmf(i, "all", cph_data, n_iter=1000)
        cph_recon_err.append(msi.nmf.reconstruction_err_ / cph_norm)

    np.save("norm_recon_err_0p05_1_30.npy", cph_recon_err)
    cmp_list = np.hstack([np.arange(1, 5), np.arange(5, 30, 2)])
    # cph_recon_err = np.load('norm_cph_recon_err_5_30_2_clipped.npy', allow_pickle=True)
    plt.figure()
    plt.plot(cmp_list, cph_recon_err, "r-")
    plt.title("Normalized Reconstruction error vs NMF components")
    plt.xlabel("No. of NMF components")
    plt.ylabel("Reconstruction error")
    plt.savefig("paper_recon_err.png")
    # plt.savefig("./results/norm_nmf_err_plt_1_30_clipped.svg", format="svg")
    # plt.close()
    # plt.show()
    # pass
