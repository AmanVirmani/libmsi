import cv2
import os
import numpy as np
import pickle
import libmsi
import matplotlib.pyplot as plt

def show(img, name = 'img'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    # cv2.destroyWindow(name)

def normalize_img_array(a):
    return (255*a/(a.max()-a.min())).astype(np.float32)

def plotOverlayImage(img1, img2, weights=[0.5, 0.5], name='blended image'):
    dst = cv2.addWeighted(img1.astype(np.float32),weights[0],img2.astype(np.float32),weights[1],0)
    cv2.imshow(name,dst)
    # TODO: make segmentaion of he image
    # TODO: enable colored visualization of the blending
    # cv2.imshow('Blended Image Ratio: {}'.format(1-weights[1]),dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    he_path = "/work/projects/aim-hi/data/HE AIM-HI Colons Negative/DMSI0002HE.tif"
    he_img = cv2.imread(he_path)
    # show(he_img)
    nmf_cmps_path = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/pca_outputs/no_whitening_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_20_components_pca_dict_tic_normalized.npy"
    # nmf_cmps_path = "/work/projects/aim-hi/data/nmf_20_cmps_bin_1.npy"
    nmf_cmps = np.load(nmf_cmps_path, allow_pickle=True)[()]
    nmf_cmps_2d = nmf_cmps['nmf_outputs'][0][0][int(sum(nmf_cmps['pixel_count_array'][:4])):int(sum(nmf_cmps['pixel_count_array'][:4]))+int(nmf_cmps['pixel_count_array'][4])]
    imzml_path = "/lab/projects/aim-hi/data/DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.npy"
    imzml = libmsi.Imzml(imzml_path)
    nmf_cmps_3d = imzml.reconstruct_3d_array(nmf_cmps_2d)

    he_small = cv2.resize(he_img, nmf_cmps_3d.shape[:2][::-1])
    he_small_gray = cv2.cvtColor(he_small, cv2.COLOR_RGB2GRAY)
    he_norm = normalize_img_array(he_small_gray)

    # for weight in 0.1*np.arange(10):
    #     cmp_img = normalize_img_array(nmf_cmps_3d[:, :, 1])
    #     # plotOverlayImage(nmf_cmps_3d[:,:,1], he_small_gray, [weight, 1-weight])
    #     plotOverlayImage(nmf_cmps_3d[:,:,1], he_norm, [10*weight, 1-weight])
    #     # plotOverlayImage(cmp_img, he_norm, [weight, 1-weight])

    # for i in range(nmf_cmps_3d.shape[-1]):
    #     cmp_img = normalize_img_array(nmf_cmps_3d[:,:,i])
    #     show(nmf_cmps_3d[:,:,i], 'nmf')
    #     show(np.hstack([cmp_img, he_norm]),'both')
    #     cv2.destroyAllWindows()
    # exit()
    methods = [
        # cv2.TM_SQDIFF,
        # cv2.TM_SQDIFF_NORMED,
        # cv2.TM_CCORR,
        cv2.TM_CCORR_NORMED,
        # cv2.TM_CCOEFF,
        cv2.TM_CCOEFF_NORMED
    ]
    # method_names = ["cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR", "cv2.TM_CCORR_NORMED", "cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]
    results = np.zeros((len(methods), nmf_cmps_3d.shape[-1]))
    # show(he_small_gray, 'he_gray')
    # show(he_norm, 'he_norm')
    golden_weight = 0.7
    for i in range(nmf_cmps_3d.shape[-1]):
        print("="*10)
        print("component index:", i)
        cmp_img = normalize_img_array(nmf_cmps_3d[:, :, i])
        plotOverlayImage(nmf_cmps_3d[:,:,i], he_norm, [10*golden_weight, 1-golden_weight], "nmf component: {}".format(i))
        # plotOverlayImage(nmf_cmps_3d[:,:,i], he_small_gray)
        # plotOverlayImage(cmp_img, he_norm)
        # show(nmf_cmps_3d[:,:,i], "component")
        # show(cmp_img, "component_norm")
        for j, method in enumerate(methods):
            # results[j, i] = cv2.matchTemplate(he_small_gray.astype(np.float32), nmf_cmps_3d[:,:,i].astype(np.float32), method)
            results[j, i] = cv2.matchTemplate(he_norm,cmp_img,method)[0]
            print("Method {}  : Result{}" .format(method,results[j, i]))
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(results[j, i])
    cv2.destroyAllWindows()
    # results = np.squeeze(results)
    plt.plot(results[0], 'r')
    plt.plot(results[1], 'b')
    plt.xticks(np.arange(20))
    plt.yticks(np.arange(10)*0.1)
    # plt.title('Raw')
    plt.title('Normalized')
    plt.legend(["cv2.TM_CCORR_NORMED", "cv2.TM_CCOEFF_NORMED"])
    plt.show()
    pass