import copy as cp

import cv2
import numpy as np
import overlay as ov
from sklearn.cluster import KMeans


def make_image_segments(he_img, n_segments=5):
    processed_img = preprocess_he_img(he_img)
    # he_img_blur = cv2.GaussianBlur(processed_img, (5, 5), 0)
    # he_img_hsv = cv2.cvtColor(he_img_blur, cv2.COLOR_BGR2HSV)
    he_img_hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    h, w, d = he_img.shape

    hsv_array = np.array(he_img_hsv, dtype=np.float64) / 255
    hsv_flat = np.reshape(hsv_array, (-1, 3))

    kmeans = KMeans(n_clusters=n_segments, random_state=0).fit(hsv_flat)

    labels = kmeans.predict(hsv_flat)
    segmented_array = kmeans.cluster_centers_[labels]
    segmented_img = segmented_array.reshape((h, w, d))
    segmented_img_hsv = np.uint8(segmented_img * 255)
    segmented_rgb = cv2.cvtColor(segmented_img_hsv, cv2.COLOR_HSV2BGR)

    segments = []
    for i in range(n_segments):
        curr_labels = cp.deepcopy(labels)
        curr_labels[labels == i] = -1
        curr_labels[labels != i] = 0
        curr_labels[labels == i] = 1
        img_mask = np.reshape(curr_labels, (h, w)).astype(np.uint8)
        # masked = cv2.bitwise_and(segmented_rgb, segmented_rgb, mask=img_mask)
        masked = cv2.bitwise_and(he_img, he_img, mask=img_mask)
        segments.append(masked)

    return segments


def preprocess_he_img(he_rgb):
    preprocessed_img = he_rgb.copy()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    for i in range(he_rgb.shape[2]):
        preprocessed_img[:, :, i] = clahe.apply(he_rgb[:, :, i])

    return preprocessed_img


if __name__ == "__main__":
    he_path = "/mnt/sda/avirmani/data/HE_neg/HE_Colons_Negative/DMSI0002HE.tif"
    he_img = cv2.imread(he_path)
    segments = make_image_segments(he_img)
    for segment in segments:
        cv2.imshow('seg', segment)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
