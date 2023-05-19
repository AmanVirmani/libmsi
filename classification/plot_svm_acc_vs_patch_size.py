import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.svm import SVC
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

if __name__ == "__main__":
    # change cwd
    os.chdir("/home/avirmani/msi_project/avirmani/libmsi/npy_files/")
    print("current working directory", os.getcwd())

    patch_dims = np.arange(5, 31, 5)

    acc_scores = []
    svms = []
    svms = np.load("svm_results.npy", allow_pickle=True)[()]
    acc_scores = np.load(
        "/mnt/sda/avirmani/libmsi/nmf_acc_patch_size_5_31_5.npy", allow_pickle=True
    )
    pca_acc_scores = np.load(
        "/mnt/sda/avirmani/libmsi/pca_19_acc_patch_size_5_31_5.npy", allow_pickle=True
    )[()]

    # plt.xlim([0, 30])
    # plt.ylim([0.8, 1])
    plt.plot(patch_dims, acc_scores, "b", linewidth=4)
    plt.plot(patch_dims, pca_acc_scores, "g", linewidth=4)
    # plt.plot(n_comp_list[4], acc_scores[4], "ro", markersize=8)
    # plt.plot(n_comp_list[18], acc_scores[18], "ro", markersize=8)
    # plt.hlines(
    #     y=acc_scores[4], xmin=0, xmax=n_comp_list[4], color="k", linestyles="dashed"
    # )
    # plt.vlines(
    #     x=n_comp_list[4], ymin=0, ymax=acc_scores[4], color="k", linestyles="dashed"
    # )
    # plt.hlines(
    #     y=acc_scores[18], xmin=0, xmax=n_comp_list[18], color="k", linestyles="dashed"
    # )
    # plt.vlines(
    #     x=n_comp_list[18], ymin=0, ymax=acc_scores[18], color="k", linestyles="dashed"
    # )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.02))

    plt.xlabel("# of components")
    plt.ylabel("Accuracy")
    plt.legend(["NMF", "PCA"])
    plt.show()
    exit()

    fig, axes = plt.subplots(2, 1)
    fig.tight_layout(pad=4.0)
    axes[0].set_xlim([0, 30])
    axes[0].set_ylim([0.5, 1])
    axes[0].plot(n_comp_list, acc_scores, "b", linewidth=4)
    axes[0].plot(n_comp_list[4], acc_scores[4], "ro", markersize=8)
    axes[0].plot(n_comp_list[18], acc_scores[18], "ro", markersize=8)
    axes[0].hlines(
        y=acc_scores[4], xmin=0, xmax=n_comp_list[4], color="k", linestyles="dashed"
    )
    axes[0].vlines(
        x=n_comp_list[4], ymin=0, ymax=acc_scores[4], color="k", linestyles="dashed"
    )
    axes[0].hlines(
        y=acc_scores[18], xmin=0, xmax=n_comp_list[18], color="k", linestyles="dashed"
    )
    axes[0].vlines(
        x=n_comp_list[18], ymin=0, ymax=acc_scores[18], color="k", linestyles="dashed"
    )
    axes[0].xaxis.set_major_locator(MultipleLocator(5))
    axes[0].xaxis.set_minor_locator(MultipleLocator(1))
    axes[0].yaxis.set_major_locator(MultipleLocator(0.1))
    axes[0].yaxis.set_minor_locator(MultipleLocator(0.02))
    # ax.autoscale_view()
    axes[0].set_xlabel("# NMF components")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("SVM accuracy on test data")

    w = svms[18].dual_coef_.dot(svms[18].support_vectors_)
    w = w / np.linalg.norm(w)
    w_r = w.reshape(-1, 19)
    w_r = np.abs(w_r.transpose())
    axes[1].imshow(w_r, aspect="auto")
    axes[1].set_title("SVM weights")
    axes[1].set_xlabel("pixel index")
    axes[1].set_ylabel("NMF")
    # axes[1].axis('off')
    # axes[1].get_xaxis().set_visible(False)
    axes[1].tick_params("x", bottom=False, labelbottom=False)
    plt.show()
    # plt.savefig('svm_accuracy_and_weights.png')
    pass
