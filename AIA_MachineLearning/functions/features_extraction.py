# Import Libraries
import cv2
import numpy as np
import math as m
from skimage import feature


def geometric_moments(mask):
    """
        Function that returns the following features from the geometric moments: area, 7 hu moments
        and the eccentricity, and the perimeter from the contour
        :param mask: Input segmentation, it has to be a binary image
        ------------
        :return: area
        :return: perimeter
        :return: hu_flat
        :return: eccentricity
    """
    # Calculate Moments
    moments = cv2.moments(mask)
    # m00: area
    area = moments['m00']

    # The 2nd-order moments give the ellipsoid of inertia
    # Computing eigenvalues for eccentricity
    eigenvalue_1 = (moments['mu20'] + moments['mu02']) / 2 + (
        m.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2)) / 2
    eigenvalue_2 = (moments['mu20'] + moments['mu02']) / 2 - (
        m.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2)) / 2
    eigenvalues = [eigenvalue_1, eigenvalue_2]

    # Eccentricity
    eccentricity = m.sqrt(1 - min(eigenvalues) / max(eigenvalues))

    # Calculate Hu Moments
    hu_moments = np.asarray(cv2.HuMoments(moments))

    # Log scale hu moments to enlarge the scale
    for i in range(0, 7):
        hu_moments[i] = -1 * m.copysign(1.0, hu_moments[i]) * m.log10(abs(hu_moments[i]))

    hu_flat = np.hstack(hu_moments)

    # Contour
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(contour) for contour in contours]
    max_index = np.argmax(areas)
    perimeter = cv2.arcLength(contours[max_index], True)

    return area, perimeter, hu_flat, eccentricity


def irregularity(perimeter, area):
    """
        Computes the irregularity index from the equation: perimeter^2 / 4 * pi * area
        :param perimeter
        :param area
        --------------
        :return: irregularity_index
    """
    irregularity_index = perimeter ** 2 / (4 * m.pi * area)

    return irregularity_index


class LocalBinaryPatterns:
    """
        Class to compute the Local Binary Patterns (LBP) of an image
        ----------
    """

    def __init__(self, numpoints, radius, method):
        """
            Store the number of points and radius
            :param numpoints: number of sampling points
            :param radius: distance from a given pixel to the desired sampling points
            :param method: 3 possible methods to by applied 'default', 'ror' and 'uniform'
            ------------
        """
        self.numpoints = numpoints
        self.radius = radius
        self.method = method

    def describe(self, image, eps=1e-7):
        """
            Compute the LBP
            :param image: image to compute the LBP
            :param eps: No dividing by 0 constant
            ------------
            :return hist: LBP histogram
            :return lbp: patterns in ROI
        """
        # compute the LBP representation of the image, and build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numpoints, self.radius, self.method)
        n_bins = int(lbp.max() + 1)
        (hist, _) = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        # Auxiliary vector with ROR patterns location
        aux = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 37, 39, 43, 45, 47, 51, 53, 55, 59, 61, 63,
               85, 87, 91, 95, 111, 119, 127, 255]
        hist2 = []
        j = 0

        # Loops for search, fit and save the ROR patterns in a vector
        if self.method == 'ror':
            for i in range(0, n_bins):
                for j in range(0, len(aux)):
                    if i == aux[j]:
                        hist2.append(hist[i])
            hist = np.asarray(hist2)

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist, lbp


def texture(mask_img):
    """
        Compute the texture features Gray Level Co-occurrence Matrix (GLCM) and LBP
        :param mask_img: Merged image with the computed segmentation
        ------------
        :return texture_features: vector with all texture features
    """
    copy = mask_img.copy()
    # Convert the mask in grayscale
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

    # Find the GLCM
    # graycomatrix(image, distances, angles, levels=256, symmetric=False, normed=False)
    graycom = feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

    # Find the GLCM properties
    contrast = feature.graycoprops(graycom, 'contrast')
    dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
    homogeneity = feature.graycoprops(graycom, 'homogeneity')
    energy = feature.graycoprops(graycom, 'energy')
    correlation = feature.graycoprops(graycom, 'correlation')
    ASM = feature.graycoprops(graycom, 'ASM')

    # Gray Level co-occurrence matrix data
    contrast = contrast.flatten()
    dissimilarity = dissimilarity.flatten()
    homogeneity = homogeneity.flatten()
    energy = energy.flatten()
    correlation = correlation.flatten()
    ASM = ASM.flatten()

    # LBP Texture vector
    desc = LocalBinaryPatterns(8, 1, 'uniform')
    hist, lbp = desc.describe(gray)
    desc1 = LocalBinaryPatterns(8, 2, 'uniform')
    hist1, lbp1 = desc1.describe(gray)
    desc2 = LocalBinaryPatterns(16, 2, 'uniform')
    hist2, lbp2 = desc2.describe(gray)

    # Concatenation of all the textures features in one unique list
    texture_features = np.concatenate((contrast, dissimilarity, homogeneity, energy, correlation, ASM, hist, hist1,
                                       hist2), axis=0)

    return texture_features


def color(mask_img):
    """
        Compute the color features: mean, standard deviation and skewness by channel
        :param mask_img: Merged image with the computed segmentation
        ------------
        :return color_features: list of all color features
    """
    copy = mask_img.copy()
    # Creating color_features empty list
    color_features = []

    # Get the pixels of the mask
    pixels = np.asarray(copy)

    # Convert them into float type
    pixels = pixels.astype('float32')

    # Split the pixels by color
    b = pixels[:, :, 0]
    g = pixels[:, :, 1]
    r = pixels[:, :, 2]

    # Get the mean and standard deviation for each channel and for the whole image
    mean, std = pixels.mean(), pixels.std()
    r_mean, r_std = r.mean(), r.std()
    g_mean, g_std = g.mean(), g.std()
    b_mean, b_std = b.mean(), b.std()

    # Append the features
    color_features.append(r_mean)
    color_features.append(g_mean)
    color_features.append(b_mean)
    color_features.append(r_std)
    color_features.append(g_std)
    color_features.append(b_std)

    # Calculate the Skewness feature, by channel and for the whole image

    r_skewness, g_skewness, b_skewness = pow(abs(r - r_mean), 3), pow(abs(g - g_mean), 3), pow(abs(b - b_mean), 3)
    r_skewness, g_skewness, b_skewness = r_skewness / len(r_skewness), g_skewness / len(g_skewness), \
                                         b_skewness / len(b_skewness)
    r_skewness, g_skewness, b_skewness = pow(abs(r_skewness - r_mean), 1 / 3), pow(abs(g_skewness - g_mean), 1 / 3), \
                                         pow(abs(b_skewness - b_mean), 1 / 3)
    # Get the mean skewness
    r_skewness, g_skewness, b_skewness = r_skewness.mean(), g_skewness.mean(), b_skewness.mean()

    # Append the features
    color_features.append(r_skewness)
    color_features.append(g_skewness)
    color_features.append(b_skewness)

    return color_features

