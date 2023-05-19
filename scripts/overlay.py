import cv2
import libmsi
import matplotlib.pyplot as plt
import numpy as np
# from segmentation import make_image_segments
import segmentation


def show(img, name='img', hold=True):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    if not hold:
        cv2.destroyWindow(name)


def image_registration(src, target, n_iter=5000, termination_eps=1e-10, warp_mode=cv2.MOTION_HOMOGRAPHY, debug=False):
    image = normalize_img(src)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    templateGray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if len(target.shape) == 3 else target

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, termination_eps)
    input_mask = np.ones_like(imageGray)
    try:
        cc, warp_matrix = cv2.findTransformECC(templateGray, imageGray, warp_matrix, warp_mode, criteria, input_mask, 5)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(image, warp_matrix, templateGray.shape[::-1],
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            aligned = cv2.warpAffine(image, warp_matrix, templateGray.shape[::-1],
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except:
        print("Warning: Images may be uncorrelated!!")
        aligned = imageGray
    if debug:
        show(image, 'NMF')
        show(templateGray, 'H&E')
        show(aligned, 'aligned')
    return aligned


def normalize_img(image):
    try:
        norm_img = (image * 255) / (image.max() - image.min())
    except:
        norm_img = image
    return norm_img.astype(np.uint8)


def align_images(image, template, maxFeatures=500, keepPercent=0.2,
                 debug=False):
    # convert both the input image and template to grayscale
    image = normalize_img(image)
    # template = normalize_img(template)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
                                     matches, None)
        # matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)
        cv2.destroyWindow("Matched Keypoints")
    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    # return the aligned image
    return aligned


if __name__ == "__main__":
    he_path = "/home/avirmani/projects/aim-hi/data/HE AIM-HI Colons Negative/DMSI0002HE.tif"
    he_img = cv2.imread(he_path)
    segments = segmentation.make_image_segments(he_img)
    # show(he_img)
    nmf_cmps_path = "/mnt/sda/avirmani/data/nmf_20_all.npy"
    nmf_cmps = np.load(nmf_cmps_path, allow_pickle=True)[()]
    nmf_cmps_2d = nmf_cmps['nmf_outputs'][0][0][
                  int(sum(nmf_cmps['pixel_count_array'][:4])):int(sum(nmf_cmps['pixel_count_array'][:4])) + int(
                      nmf_cmps['pixel_count_array'][4])]
    imzml_path = "/home/avirmani/projects/aim-hi/data/DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.npy"
    imzml = libmsi.Imzml(imzml_path)
    nmf_cmps_3d = imzml.reconstruct_3d_array(nmf_cmps_2d)

    he_small = cv2.resize(he_img, nmf_cmps_3d.shape[:2][::-1])
    he_small_gray = cv2.cvtColor(he_small, cv2.COLOR_RGB2GRAY)
    he_norm = normalize_img(he_small_gray)

    methods = [
        cv2.TM_CCORR_NORMED,
        cv2.TM_CCOEFF_NORMED,
        cv2.TM_SQDIFF_NORMED
    ]
    score_before = np.zeros((len(segments), nmf_cmps_3d.shape[-1]))
    score_after = np.zeros((len(segments), nmf_cmps_3d.shape[-1]))
    for segment_idx, segment in enumerate(segments):
        segment = cv2.resize(segment, nmf_cmps_3d.shape[:2][::-1])
        for nmf_idx in range(nmf_cmps_3d.shape[-1]):
            print("Matching {} nmf component with {} segment".format(nmf_idx, segment_idx))
            nmf_img = normalize_img(nmf_cmps_3d[:, :, nmf_idx])
            segmentGray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
            # score_before[segment_idx, nmf_idx] = cv2.matchTemplate(nmf_img, segmentGray, methods[2])
            score_before[segment_idx, nmf_idx] = cv2.computeECC(segmentGray, nmf_img)
            # align_images(nmf_cmps_3d[:, :, nmf_idx], segment, debug=True)
            aligned = image_registration(nmf_cmps_3d[:, :, nmf_idx], segment)
            # score_after[segment_idx, nmf_idx] = cv2.matchTemplate(aligned, segmentGray, methods[2])
            score_before[segment_idx, nmf_idx] = cv2.computeECC(segmentGray, aligned)
        best_score = max(score_after[segment_idx])
        best_match = np.where(score_after[segment_idx] == best_score)[0][0]
        print("Segment {} best matches with NMF index {}; match score is {}".format(segment_idx, best_match, best_score))
        best_aligned = image_registration(nmf_cmps_3d[:,:,best_match], segment)
        cv2.imwrite("nmf_{}_segment_{}.jpg".format(best_match, segment_idx), np.hstack((best_aligned, segmentGray)))
        # cv2.imshow("nmf {} | segment {}".format(best_match, segment_idx), np.hstack((best_aligned, segmentGray)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    for i in range(len(segments)):
        plt.figure()
        plt.plot(score_before[i],'b')
        plt.plot(score_after[i], 'r')
        plt.legend(["before_matching", "after matching"])
        plt.title("Normalized Correlation Score for H&E segment {}".format(i))
        plt.xlabel("NMF components")
        plt.ylabel("Score")
        plt.savefig("Segment_{}.jpg".format(i))

    exit()
