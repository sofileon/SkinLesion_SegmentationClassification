# Import Libraries and internal functions
import cv2
import numpy as np
from utils import imshow_resize
from utils import get_fov


def kmeans(img):
    """
        This function performs the K-means segmentation of a given image. If the image has FOV, it will be eliminated
        :param img: input image
        -------------
        :return: final_mask : final segmentation
    """
    copy_img = img.copy()

    # K-means Parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.1)  # max_iter and epsilon
    k = 2  # number of classes
    attempts = 10

    # Image contrast
    # l a b  components
    lab_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab_img)
    # Applying adaptive histogram equalization (CLAHE) to L-channel
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(6, 6))
    new_l_channel = clahe.apply(l_channel)
    # Merge the new L-channel with the a and b channels
    lab_img_contrast = cv2.merge((new_l_channel, a, b))
    # Converting image from LAB Color model back to BGR
    img_contrast = cv2.cvtColor(lab_img_contrast, cv2.COLOR_LAB2BGR)

    # Image blurred to remove noise before K-means
    image_blurred = cv2.blur(img_contrast, (5, 5))

    # Reshape the image matrix to a vector of single pixel observations to perform K-means
    data = np.float32(image_blurred)
    data = data.reshape((-1, 3))

    # K-means with kmeans++ label initialization
    retval, best_labels, centers = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # Merge the centers with labels
    data = centers[best_labels.flatten()]
    # Reshape image to original shape
    data = np.uint8(data)
    data = data.reshape(copy_img.shape)

    # Smooth boundaries
    smooth_data = cv2.medianBlur(data, 7)

    # Grayscale of the K-means result to by applied in the OTSU binarization
    gray_img = cv2.cvtColor(smooth_data, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape

    # Selecting a box in the corners of input image in Grayscale to check if there is FOV
    gray_img_copy = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)
    top_left_corner = np.mean(gray_img_copy[0:3, 0:3])
    top_right_corner = np.mean(gray_img_copy[0:3, width - 3:width])
    bottom_left_corner = np.mean(gray_img_copy[height - 3:height, 0:3])
    bottom_right_corner = np.mean(gray_img_copy[height - 3:height, width - 3:width])

    # OTSU Thresholding if there is FOV also remove it
    if int(top_left_corner < 40) + int(top_right_corner < 40) + int(bottom_left_corner < 40) + int(bottom_right_corner < 40) > 2:
        # Internal function get_fov
        image_fov = get_fov(gray_img_copy)
        # OTSU
        ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Remove FOV
        thresh = thresh * image_fov
    else:
        # OTSU
        ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Closing to remove holes from the thresh
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for i in range(1):
        thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel1)

    # Find contours
    all_objects, hierarchy = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Segmented mask
    # select largest area (should be the skin lesion)
    mask = np.zeros(thresh1.shape, dtype='uint8')
    area = [cv2.contourArea(object_) for object_ in all_objects]
    index_contour = area.index(max(area))
    cv2.drawContours(mask, all_objects, index_contour, 255, -1)

    # Final dilation to enlarge and smooth the mask
    kernel_final = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    final_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_final, iterations=10)

    return final_mask

