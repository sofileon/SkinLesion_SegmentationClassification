# Import Libraries
import glob
import cv2
import csv
from pathlib import Path

def load_images(path, format_img, flag):
    """
        To load multiple files
        :param path: script with the folder that containes to files to be loaded
        :param format_img: script with the extension of the files
        :param flag: 0: to load in grayscale useful for the masks to have them directly in 1 channel, 1: BGR images
        -------------
        :return: images: list containing all the files
    """
    files = glob.glob(path + format_img)
    files.sort()
    images = [cv2.imread(file, flag) for file in files]

    return images


def imshow_resize(script, img, factor_resize):
    """
        :param script: Title of the image
        :param img: image to plot
        :param factor_resize: factor wanted for the resizing
        -------------
        :return: cv2.imshow with the resizing factor applied
    """
    width = int(img.shape[1] * factor_resize)
    height = int(img.shape[0] * factor_resize)
    dim = (width, height)
    # resize image
    resized = (cv2.resize(img, dim, interpolation=cv2.INTER_AREA))

    return cv2.imshow(script, resized)


def store(image, name, output_path, suffix_name):
    """
        :param image: image to be stored as file
        :param name: string with the desired name for the file
        :param output_path: path to save the image
        :param suffix_name: suffix name to be added to the name and with the extension to be saved
    -------------
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    img_name = output_path + name + suffix_name
    cv2.imwrite(img_name, image)


def get_fov(img):
    """
        This function returns a binary image with values =0 in the pixels with low intensities (as the FOV)
        :param img:  image with FOV
        ---------------
        :return thresh1: FOV with values of 0 the image with values of 1
    """
    copy = img.copy()
    ret, thresh = cv2.threshold(copy, 45, 255, cv2.THRESH_BINARY)
    thresh[thresh == 255] = 1

    return thresh


def csv_writer(filename, action, row):
    """
         :param filename: name of the csv to be written on
         :param action: either 'w' to write a new csv file or 'a' to append a new row
         :param row: data to be appended to new row
         ---------------
    """
    with open(filename, action, encoding='UTF8', newline='') as f:  # 'a' to append row
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()

