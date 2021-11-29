import cv2
import os
import numpy as np
import overlay as ov
import libmsi
import segmentation as sgm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

if __name__=="__main__":
    # load nmf data
    nmf_data_path = "/work/projects/aim-hi/data/nmf_20_cmps_bin_1.npy"
    nmf_data = np.load(nmf_data_path, allow_pickle=True)[()]

    # load coordinate list -> convert to 3d image
    data_dir = "/lab/projects/aim-hi/data/"
    data_dir = "/work/projects/aim-hi/data/"
    imzml_paths = [
        "msi/dmsi0005.npy",  # 0
        "msi/dmsi0004.npy",  # 1
        "msi/dmsi0008.npy",  # 2
        "msi/dmsi0011.npy",  # 3
        "msi/dmsi0002.npy",  # 4
        "msi/dmsi0006.npy",  # 5
        "msi/dmsi0009.npy",  # 6
        "msi/dmsi0012.npy",  # 7
    ]
    # imzml_paths = [
    #     "DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.npy", #4
    #     "DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004.npy",
    #     "DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.npy",
    #     "DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006.npy",
    #     "DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.npy",
    #     "DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009.npy",
    #     "DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.npy",
    #     "DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.npy",
    # ]
    nmf_3d_data = []
    curr = 0
    for path in imzml_paths:
        imzml = libmsi.Imzml(os.path.join(data_dir,path))
        print(len(imzml.coordinates))
        nmf_2d_data = nmf_data["nmf_outputs"][0][0][curr:curr+len(imzml.coordinates)]
        curr = curr + len(imzml.coordinates)
        nmf_3d_data.append(imzml.reconstruct_3d_array(nmf_2d_data))

    nmf_components = nmf_data["nmf_outputs"][1][0]
    # load segmentation outputs and original he image
    y, x, _ = nmf_3d_data[4].shape
    he_gold = "/work/projects/aim-hi/data/HE AIM-HI Colons Negative/DMSI0002HE.tif"
    he_img = cv2.imread(he_gold)
    he_img = cv2.resize(he_img, (x, y))
    segments = sgm.make_image_segments()
    # color different segments with different colors
    segmented_img_list = []
    segmented_img = np.zeros(segments[0].shape, dtype=np.uint8)
    colors = {
        "black":(0,0,0),
        "red": (0,102,204),
        "caramel": (153, 63, 0),
        # "caramel": (0,63,153),
        "blue": (255,41,0),
        "green": (0,153,0),
        "white": (255, 255, 255),
    }

    color_list = list(colors.values())
    for i, segment in enumerate(segments):
        # segment[segment != (0, 0, 0)] = color_list[i]
        segment[(segment != 0).all(axis=-1)] = color_list[i]
        segment[(segment == 0).all(axis=-1)] = color_list[-1]
        segmented_img += segment
        segment = cv2.resize(segment, (x,y))
        segmented_img_list.append(segment)
        # ov.show(segment)

    # ov.show(segmented_img)
    segmented_img = cv2.resize(segmented_img, (x, y))
    # ov.show(segmented_img)
    cv2.destroyAllWindows()

    # to plot them in subplots
    # best_nmf_indices=[1, 8, 9]
    best_nmf_indices=[2, 17, 9]
    starting_mz = 640

    ### Creating my custom colormaps
    reds = cm.get_cmap('Reds', 256)
    reds_map=reds(np.linspace(0,1,256))  ### Take 256 colors from the 'reds' colormap, and distribute it between 0 and 1.
    reds_map[0:5,:]=[1,1,1,1] ### Modify the 'reds' colormap.Set the first five colors starting from 0 to pure white
    new_reds=ListedColormap(reds_map)  ### Create a matplotlib colormap object from the newly created list of colors

    greens = cm.get_cmap('Greens', 256)
    greens_map=greens(np.linspace(0,1,256))  ### Take 256 colors from the 'Greens' colormap, and distribute it between 0 and 1.
    greens_map[0:5,:]=[1,1,1,1] ### Modify the 'Greens' colormap.Set the first five colors starting from 0 to pure white
    new_greens=ListedColormap(greens_map)  ### Create a matplotlib colormap object from the newly created list of colors

    blues = cm.get_cmap('Blues', 256)
    blues_map=blues(np.linspace(0,1,256))  ### Take 256 colors from the 'Blues' colormap, and distribute it between 0 and 1.
    blues_map[0:5,:]=[1,1,1,1] ### Modify the 'Blues' colormap.Set the first five colors starting from 0 to pure white
    new_blues=ListedColormap(blues_map)  ### Create a matplotlib colormap object from the newly created list of colors
    ###

    cmap_array_spatial=[new_reds,new_greens, new_blues]
    cmap_array_spectra=['r','g','b']

    fig, axes = plt.subplots(3, 3, gridspec_kw={"width_ratios":[3,3,4]})
    # axes[0, 0].imshow(he_img)
    axes[0, 2].imshow(segmented_img_list[3], aspect="auto") # , extent=[0,300,9,267])
    axes[1, 2].imshow(segmented_img_list[4], aspect="auto") # , extent=[0,300,9,267])
    axes[2, 2].imshow(segmented_img_list[1], aspect="auto") # , extent=[0,300,9,267])
    axes[2, 2].set(xlabel='Segmented H&E\nstained tissues')
    axes[2, 1].set(xlabel='NMF component Image')
    axes[2, 0].set(xlabel='MSI Spectra (m/z)')
    for i, idx in enumerate(best_nmf_indices):
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        # axes[i, 0].xaxis.set_tick_params(length=0, width=0, labelsize=0)
        # axes[i, 0].yaxis.set_tick_params(length=0, width=0, labelsize=0)
        axes[i, 2].xaxis.set_tick_params(length=0, width=0, labelsize=0)
        axes[i, 2].yaxis.set_tick_params(length=0, width=0, labelsize=0)
        axes[i, 1].imshow(nmf_3d_data[0][:,:,idx], cmap=cmap_array_spatial[i]).set_interpolation('none')
        # axes[i, 1].imshow(nmf_3d_data[4][:,:,idx], cmap=cmap_array_spatial[i]).set_interpolation('none')
        axes[i, 1].xaxis.set_tick_params(length=0, width=0,labelsize=0)
        axes[i, 1].yaxis.set_tick_params(length=0, width=0,labelsize=0)
        axes[i, 0].plot(np.arange(starting_mz, starting_mz+nmf_components.shape[-1]), nmf_components[idx], cmap_array_spectra[i], linewidth=1)
        axes[i, 0].spines["right"].set_visible(False)
        axes[i, 0].spines["top"].set_visible(False)
        axes[i, 1].spines["right"].set_visible(False)
        axes[i, 1].spines["left"].set_visible(False)
        axes[i, 1].spines["top"].set_visible(False)
        axes[i, 1].spines["bottom"].set_visible(False)
        axes[i, 2].spines["right"].set_visible(False)
        axes[i, 2].spines["left"].set_visible(False)
        axes[i, 2].spines["top"].set_visible(False)
        axes[i, 2].spines["bottom"].set_visible(False)

    # axes[1, 1].imshow(nmf_3d_data[4][:, :, 15]+nmf_3d_data[4][:, :, 17], cmap=cmap_array_spatial[1]).set_interpolation('none')
    # axes[0,0].get_shared_y_axes().join(axes[0,0], axes[1,0], axes[2,0])
    # axes[0,0].get_shared_x_axes().join(axes[0,0], axes[1,0], axes[2,0])
    # axes[0,1].get_shared_x_axes().join(axes[0,1], axes[0,2])
    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.show()
    pass
