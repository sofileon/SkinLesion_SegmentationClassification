# Import Libraries
import cv2
import numpy as np


def base(width, height):
    """
        This function makes a black squared structuring element with a horizontal white line
        of determined thickness in the middle
        :param width: size of the square
        :param height: thickness of the white line
         ---------------
        :return base_result: Structuring element
    """
    # Ensuring that width of the square is larger than the thickness of the line
    if width < height:
        raise Exception("Specified width must be greater than specified height")
    # Creation of a black square
    base_result = np.zeros((width, width), dtype=np.uint8)
    # To center the horizontal line inside the SE we do: start=width/2-height/2; end=width/2+height/2
    start = int(width / 2 - height / 2)
    end = int(width / 2 + height / 2)
    # Creation of the horizontal line, it is repeated "height" times
    for i in np.arange(start, end):
        base_result = cv2.line(base_result, (0, i), (width, i), 255, 1)

    return base_result


def se_rotation(SE, angle):
    """
        This function rotates a given structuring element with a specified angle
        :param SE: structuring element to be rotated
        :param angle: angle of rotation in degrees (positive for anti-clockwise and negative for clockwise)
        ---------------
        :return rotated_SE: rotated SE of the same dimensions
    """
    (h, w) = SE.shape[:2]
    center = (w / 2, h / 2)
    # transformation matrix which will be used for rotating the SE
    rot_m = cv2.getRotationMatrix2D(center, angle, 1)
    # Actual rotation of the SE
    rotated_SE = cv2.warpAffine(SE, rot_m, (w, w), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    return rotated_SE


def stacker(list_of_images):
    """
        This function adds a list of images of the same size s
        :param list_of_images: images to be added pixel-wise.
        ---------------
        :return finalResult: uint16 2-D array of size s
    """
    s = list_of_images[0].shape[:2]
    final_result = np.zeros(s, dtype='uint16')
    for image in list_of_images:
        final_result = np.add(final_result, image.astype('uint16'), dtype=np.uint16)

    return final_result

