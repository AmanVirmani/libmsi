import cv2
import numpy as np
import overlay as ov
from sklearn.cluster import KMeans
import copy as cp


def make_image_segments():
    he_gold = "/work/projects/aim-hi/data/HE AIM-HI Colons Negative/DMSI0002HE.tif"
    he_path = "/work/projects/aim-hi/data/HE AIM-HI Colons Negative/DMSI0002HE_color_scaled.tif"
    he_gold_img = cv2.imread(he_gold)
    he_gold_img = np.uint8(he_gold_img)
    he_img = cv2.imread(he_path)
    he_img_blur = cv2.GaussianBlur(he_img, (5, 5), 0)
    he_img_hsv = cv2.cvtColor(he_img, cv2.COLOR_BGR2HSV)
    h, w, d = he_img.shape

    hsv_array = np.array(he_img_hsv, dtype=np.float64) / 255
    hsv_flat = np.reshape(hsv_array, (-1, 3))

    n_colors = 5
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(hsv_flat)

    labels = kmeans.predict(hsv_flat)
    segmented_array = kmeans.cluster_centers_[labels]
    segmented_img = segmented_array.reshape((h, w, d))
    segmented_img_hsv = np.uint8(segmented_img*255)
    segmented_rgb = cv2.cvtColor(segmented_img_hsv, cv2.COLOR_HSV2BGR)
    # ov.show(segmented_rgb)
    # cv2.destroyAllWindows()

    # for i in [1,4]:
    segments = []
    for i in range(n_colors):
        curr_labels = cp.deepcopy(labels)
        curr_labels[labels == i] = -1
        curr_labels[labels != i] = 0
        curr_labels[labels == i] = 1
        img_mask = np.reshape(curr_labels, (h, w)).astype(np.uint8)
        # masked = cv2.bitwise_and(segmented_rgb, segmented_rgb, mask=img_mask)
        masked = cv2.bitwise_and(he_gold_img, he_gold_img, mask=img_mask)
        # ov.show(masked, "segement_{}".format(i))
        segments.append(masked)

    # cv2.destroyAllWindows()
    return segments

if __name__ == "__main__":
    make_image_segments()
    pass