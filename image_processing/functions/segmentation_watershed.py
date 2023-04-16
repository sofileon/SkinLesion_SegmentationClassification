# Import Libraries and internal functions
import cv2
import numpy as np
from utils import imshow_resize


def segmentation_watershed(img):
    """
        Perform the segmentation of an input image with Watershed algorithm
        :param img: input image
        ---------------
        :return mask: final segmentation
    """
    # Convert BGR image to HSV color space
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    # Segmented images with Otsu
    ret, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply a morphological operation with a 3x3 kernel to eliminate black holes
    kernel = np.ones((3, 3), np.uint8)
    bg1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply a morphological operation with a 9x9 kernel to dilate the segmented area
    kernel = np.ones((9, 9), np.uint8)
    bg = cv2.morphologyEx(bg1, cv2.MORPH_DILATE, kernel, iterations=2)

    # Create a factor to normalize the distance image
    factor_dist_transform = 0.01

    # In this operation, the gray level intensities of the points inside the foreground regions are changed to distance
    # their respective distances from the closest 0 value
    dist_transform = cv2.distanceTransform(bg, cv2.DIST_L2, 3)

    # Apply a threshold with distances
    ret, sure_fg = cv2.threshold(dist_transform, factor_dist_transform * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Get the negative of the foreground
    not_fg = cv2.subtract(bg, sure_fg)

    # Connect components in the foreground
    ret, markers = cv2.connectedComponents(sure_fg)

    # Eliminate 0 value
    markers = markers + 1

    # Mark area of interest
    markers[not_fg == 255] = 0

    # Create a copy of the original image to work with
    segmented_img = img.copy()

    # Apply Watershed function
    markers = cv2.watershed(segmented_img, markers)

    # Mark images contour with color blue
    segmented_img[markers == -1] = [255, 0, 0]

    # Extract the bigger area
    markers = (markers * 255 + 255)
    markers[markers > 0] = 255
    markers = markers.astype('uint8')
    markers = 255 - markers

    # Auxiliary mask with the shape of the contour of interest
    mask = np.zeros(markers.shape, dtype='uint8')
    row, col = markers.shape

    # Find all the contours
    all_objects, hierarchy = cv2.findContours(markers[1:row, 1:col], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Get the area of all the contours
    area = [cv2.contourArea(object_) for object_ in all_objects]

    # Select the biggest area contour
    index_contour = area.index(max(area))

    # Find the contour of the skin lesion
    cv2.drawContours(mask, all_objects, index_contour, 255, -1)

    return mask

