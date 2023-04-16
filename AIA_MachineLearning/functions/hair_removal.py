# Import Libraries and internal functions
import cv2
import numpy as np
from utils import base, se_rotation, stacker, imshow_resize


def hair_removal(img):
    """
        This function removes dark hairs from the input image
        :param img: image that will get hairs (if any) removed
        ---------------
        :return img_no_hair: image with the hairs removed
    """
    # Setting parameters
    angle_steps = 16
    if img.shape[1] < 1000:
        hair_se = base(15, 3)
        thresh_hair = 12
        line_searcher_se = base(45, 3)
    elif 1000 < img.shape[1] < 2000:
        hair_se = base(18, 3)
        thresh_hair = 8
        line_searcher_se = base(55, 3)
    elif img.shape[1] > 2000:
        hair_se = base(19, 3)
        thresh_hair = 17
        line_searcher_se = base(51, 3)

    # Splitting the channels to perform bottom-hat operation on the red channel
    b, g, r = cv2.split(img)

    # Generation of the set of 16 rotated structuring elements and performance of 16 bottom hat operations
    set_se = [se_rotation(hair_se, step) for step in range(0, 180, int(180 / angle_steps))]
    bottom_hats = [cv2.morphologyEx(r, cv2.MORPH_BLACKHAT, SE_) for SE_ in set_se]

    # Addition of all the bottom hat operations into one uint16 array, followed by a normalization
    # and a grayscale opening operation
    bottom_hats_sum = stacker(bottom_hats)
    hairs = cv2.normalize(bottom_hats_sum, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    hairs = cv2.morphologyEx(hairs, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Binarization of the resulting image and dilation of the binary image
    ret, binary = cv2.threshold(hairs, thresh_hair, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                              iterations=2)

    # Creation of the set of 16 rotated long structuring elements to search for hairs in the binary image
    se_line_rotated = [se_rotation(line_searcher_se, angle) for angle in range(0, 180, int(180 / angle_steps))]

    # Opening of the binary image followed by a closing with each of the 16 SE in se_line_rotated
    masked_lines = [cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, SE), cv2.MORPH_CLOSE, SE) for SE in
                    se_line_rotated]

    # Accumulating the lines in one 2-D array and dilating the result to get thick lines
    binary_lines = np.zeros(masked_lines[1].shape, dtype='uint8')
    for mask in masked_lines:
        binary_lines[mask > 0] = 255
    binary_lines = cv2.morphologyEx(binary_lines, cv2.MORPH_DILATE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

    # Make the decision to apply the inpaint based on the mean
    if 2 < np.mean(binary_lines) > 90:
        img_no_hair = img
    else:
        img_no_hair = cv2.inpaint(img, binary_lines, 15, cv2.INPAINT_TELEA)

    return img_no_hair

