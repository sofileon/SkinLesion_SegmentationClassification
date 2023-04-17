# main Python script Skin Lesions Segmentation and Classification Project (IPA-AIA)

# Import Libraries and internal functions
import glob
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from utils import load_images, imshow_resize, store, pad_image, csv_writer
from functions import hair_removal, segmentation_watershed, kmeans, dice_coeff, jaccard_index, geometric_moments, irregularity, texture, color
from pathlib import Path
thispath = Path(__file__).resolve()


# Run the main
if __name__ == '__main__':
    """
        flag_1_vs_all --> Flag to choose among 4:
        0: Perform hair removal, segmentation and feature extraction with just one image 
        1: Perform hair removal and save the result for all the dataset in a chosen folder
        2: Perform segmentation with the no_hair images and save the metric results (Jaccard and dice index) in a .csv file
        3: Perform feature extraction of the ISIC 2017 DATASET for the machine learning
    """
    flag_1_vs_all = 0

    if flag_1_vs_all == 0:
        # 1 Image
        num_img = "ISIC_0000043"

        # Load images
        path_mole = 'data/subset-IPA-AIA/images/'
        path_mask = 'data/subset-IPA-AIA/masks/'

        image = cv2.imread(f'{str(thispath.parent.parent)}/{path_mole}/{num_img}.jpg')
        imshow_resize('Image', image, 0.5)
        cv2.waitKey(0)
        image_mask = cv2.imread(f'{str(thispath.parent.parent)}/{path_mask}/{num_img}_segmentation.png', 0)
        image_mask = image_mask[1:image_mask.shape[0] - 1, 1:image_mask.shape[1] - 1]
        imshow_resize('Ground truth mask', image_mask, 0.5)
        cv2.waitKey(0)

        # Hair removal
        image_nohair = hair_removal(image)
        imshow_resize("Image No Hair", image_nohair, 0.5)
        cv2.waitKey(0)

        hairs = image_nohair-image
        imshow_resize("Hairs", hairs, 0.8)
        cv2.waitKey(0)
        # Remove frame for the segmentation
        image_nohair = image_nohair[1:image_nohair.shape[0] - 1, 1:image_nohair.shape[1] - 1, :]

        # Segmentation watershed
        mask_water = segmentation_watershed(image_nohair)
        imshow_resize("Mask Watershed", mask_water, 0.8)
        cv2.waitKey(100)

        # K-means segmentation
        mask_kmeans = kmeans(image_nohair)
        imshow_resize("Mask kmeans", mask_kmeans, 0.8)
        cv2.waitKey(100)

        # Computing scores
        dice_kmeans = dice_coeff(image_mask, mask_kmeans)
        print("Dice coefficient kmeans: {}".format(dice_kmeans))
        dice_water = dice_coeff(image_mask, mask_water)
        print("Dice coefficient watershed: {}".format(dice_water))
        jaccard_kmeans = jaccard_index(image_mask, mask_kmeans)
        print("Jaccard index kmeans: {}".format(jaccard_kmeans))
        jaccard_water = jaccard_index(image_mask, mask_water)
        print("Jaccard index watershed: {}".format(jaccard_water))

        # Merge Mask and image
        mask_kmeans[mask_kmeans == 255] = 1
        b, g, r = cv2.split(image_nohair)
        b_mask = b * mask_kmeans
        g_mask = g * mask_kmeans
        r_mask = r * mask_kmeans
        mask_img = cv2.merge([b_mask, g_mask, r_mask])
        imshow_resize("Mask Image", mask_img, 0.8)
        cv2.waitKey(100)

        image = image[1:image.shape[0] - 1, 1:image.shape[1] - 1, :]
        img_mask3 = np.zeros_like(image)
        img_mask3[:, :, 0] = image_mask
        img_mask3[:, :, 1] = image_mask
        img_mask3[:, :, 2] = image_mask
        img_water = np.zeros_like(mask_img)
        img_water[:, :, 0] = mask_water
        img_water[:, :, 1] = mask_water
        img_water[:, :, 2] = mask_water
        img_kmean = np.zeros_like(mask_img)
        img_kmean[:, :, 0] = mask_kmeans
        img_kmean[:, :, 1] = mask_kmeans
        img_kmean[:, :, 2] = mask_kmeans
        img_kmean[img_kmean>0] = 255
        img_water[img_water > 0] = 255
        img_mask3[img_mask3>0] = 255
        v1 = np.concatenate((image, img_mask3, image_nohair), axis=0)
        v2 = np.concatenate((img_water,img_kmean, mask_img), axis=0)
        visual = np.concatenate((v1, v2), axis=1)
        imshow_resize("All", visual, 0.2)
        cv2.waitKey(0)

        # Extract features
        features = []

        # Area, perimeter, Hu moments normalized and eccentricity
        area, perimeter, hu_moments, eccentricity = geometric_moments(mask_kmeans)
        features.extend(hu_moments)
        features.append(eccentricity)

        # Irregularity index
        irregularity_index = irregularity(perimeter, area)
        features.append(irregularity_index)

        # Texture features (GLCM and LBP)
        texture_features = texture(mask_img)
        features.extend(texture_features)

        # Color features (mean, std and skewness by channel)
        color_features = color(mask_img)
        features.extend(color_features)

    # Apply hair_removal function and save the result
    elif flag_1_vs_all == 1:
        # All Images

        # Load images
        path_mole = f'{str(thispath.parent.parent)}/data/subset-IPA-AIA/images/'
        path_mask = f'{str(thispath.parent.parent)}/data/subset-IPA-AIA/masks/'

        images = load_images(path_mole, '*.jpg', 1)
        masks = load_images(path_mask, '*png', 0)
        masks_frameless = [image[1:image.shape[0] - 1, 1:image.shape[1] - 1] for image in masks]

        # Save names
        names = [f for f in listdir(path_mole) if isfile(join(path_mole, f))]
        names = [name.removesuffix(".jpg") for name in names]
        names.sort()

        for i, image in enumerate(images):
            print('Image: ' + names[i])

            # Hair removal
            image_nohair = hair_removal(image)

            # Remove frame for the segmentation
            image_nohair = image_nohair[1:image_nohair.shape[0] - 1, 1:image_nohair.shape[1] - 1, :]

            # Store no hair images
            store(image_nohair, names[i], f'{str(thispath.parent.parent)}/data/subset-IPA-AIA/images_nohair/', '_nohair.jpg')

    # Apply kmeans and watershed functions on already saved images with no hair and save the result
    elif flag_1_vs_all == 2:
        # Run for all images_nohair

        # Name .csv
        name_metrics_csv = f"{str(thispath.parent.parent)}/data/metrics_segmentation_nohair.csv"

        # Load images
        path_mole = f'{str(thispath.parent.parent)}/data/subset-IPA-AIA/images_nohair/'
        path_mask = f'{str(thispath.parent.parent)}/data/subset-IPA-AIA/masks/'


        images_nohair = load_images(path_mole, '*.jpg', 1)  # already image_frameless
        masks = load_images(path_mask, '*.png', 0)
        masks_frameless = [image[1:image.shape[0] - 1, 1:image.shape[1] - 1] for image in masks]

        # Save names
        names = [f for f in listdir(path_mole) if isfile(join(path_mole, f))]
        names = [name.removesuffix("_nohair.jpg") for name in names]
        names.sort()

        # Initialize .csv writer with the header
        header_metric = ['Image', 'Dice Kmeans', 'Dice Watershed', 'Jaccard Kmeans', 'Jaccard Watersheds']
        csv_writer(name_metrics_csv, "w", header_metric)

        # Creating empty lists
        dice_kmeans = []
        dice_water = []
        jaccard_kmeans = []
        jaccard_water = []
        metrics = []
        i = 0

        for image in images_nohair:
            print('Image: ' + names[i])

            # Watershed
            mask_water = segmentation_watershed(image)

            # K-means
            mask_kmeans = kmeans(image)

            # Computing scores and append to the corresponding lists
            dice_m = dice_coeff(masks_frameless[i], mask_kmeans)
            dice_kmeans.append(dice_m)
            print("Dice kmeans: {}".format(dice_m))
            dice_w = dice_coeff(masks_frameless[i], mask_water)
            dice_water.append(dice_w)
            print("Dice watershed: {}".format(dice_w))
            jaccard_m = jaccard_index(masks_frameless[i], mask_kmeans)
            jaccard_kmeans.append(jaccard_m)
            print("Jaccard index kmeans: {}".format(jaccard_m))
            jaccard_w = jaccard_index(masks_frameless[i], mask_water)
            jaccard_water.append(jaccard_w)
            print("Jaccard index watershed: {}".format(jaccard_w))
            metrics.append(names[i])
            metrics.append(dice_m)
            metrics.append(dice_w)
            metrics.append(jaccard_m)
            metrics.append(jaccard_w)

            # Append metric to the .csv file
            csv_writer(name_metrics_csv, "a", metrics)

            # Clear metrics
            metrics = []

            # To save the final mask
            mask_kmeans[mask_kmeans == 1] = 255
            mask_water[mask_water == 1] = 255

            # Store kmeans and watershed masks
            store(mask_kmeans, names[i],  f'{str(thispath.parent.parent)}/data/subset-IPA-AIA/images_kmeans/', '_mask_kmeans.png')
            store(mask_water, names[i],  f'{str(thispath.parent.parent)}/data/subset-IPA-AIA/images_watershed/', '_mask_watershed.png')

            i += 1

        # Score final mean metrics
        dice_kmeans_mean = np.mean(np.asarray(dice_kmeans))
        dice_water_mean = np.mean(np.asarray(dice_water))
        jaccard_kmeans_mean = np.mean(np.asarray(jaccard_kmeans))
        jaccard_water_mean = np.mean(np.asarray(jaccard_water))
        print('-'*30)
        print("Dice_kmeans_mean: {}".format(dice_kmeans_mean))
        print("Dice_water_mean: {}".format(dice_water_mean))
        print("Jaccard index_kmeans_mean: {}".format(jaccard_kmeans_mean))
        print("Jaccard index_water_mean: {}".format(jaccard_water_mean))
        mean_metrics = ["MEAN", dice_kmeans_mean, dice_water_mean, jaccard_kmeans_mean, jaccard_water_mean]
        # Append mean metrics to the .csv file
        csv_writer(name_metrics_csv, "a", mean_metrics)

# Run for all the ISIC DATASET to perform feature extraction needed in Machine Learning
    elif flag_1_vs_all == 3:
        # Path of the ISIC 2017 dataset and the corresponding masks obtained from the U-Net
        path_DATASET = f"{str(thispath.parent.parent)}/data/train"
        path_mask_DL = f"{str(thispath.parent.parent)}/data/segmentation/train_UNET"

        # Get the names of the images
        names_DATASET = [f for f in listdir(path_DATASET) if isfile(join(path_DATASET, f))]
        names_DATASET = [name.removesuffix(".jpg") for name in names_DATASET]
        names_DATASET.sort()

        # .csv file's name
        name_features_csv = f"{str(thispath.parent.parent)}/data/feature_extraction_train.csv"

        # Creating feature empty list
        features = []

        # Construct header list with name of the different features
        header = ["Image", "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7", "Eccentricity", "Irregularity index"]
        glcm_contrast = "glcm_contrast_"
        glcm_dissimilarity = "glcm_dissimilarity_"
        glcm_homogeneity = "glcm_homogeneity_"
        glcm_energy = "glcm_energy_"
        glcm_correlation = "glcm_correlation_"
        glcm_asm = "glcm_asm_"
        for i in range(1, 5):
            header.append(glcm_contrast + str(i))
        for i in range(1, 5):
            header.append(glcm_dissimilarity + str(i))
        for i in range(1, 5):
            header.append(glcm_homogeneity + str(i))
        for i in range(1, 5):
            header.append(glcm_energy + str(i))
        for i in range(1, 5):
            header.append(glcm_correlation + str(i))
        for i in range(1, 5):
            header.append(glcm_asm + str(i))
        lbp_81 = "lbp_81_uni_"
        lbp_82 = "lbp_82_uni_"
        lbp_162 = "lbp_162_uni_"
        for i in range(1, 11):
            header.append(lbp_81 + str(i))
        for i in range(1, 11):
            header.append(lbp_82 + str(i))
        for i in range(1, 19):
            header.append(lbp_162 + str(i))
        color_names = ["mean_color_r", "mean_color_g", "mean_color_b", "std_color_r", "std_color_g", "std_color_b",
                    "skewness_r", "skewness_g", "skewness_b"]
        header.extend(color_names)
        # Write header on .csv
        csv_writer(name_features_csv, "w", header)
        i = 0

        files_mask_DL = load_images(path_mask_DL, '*.png',0)
        files_img = load_images(path_DATASET, '*.jpg', 1)

        for image, image_masks_DL in zip(files_img, files_mask_DL):
            print('Image: ' + names_DATASET[i])

            # Convert the mask into a 0/1 binary mask
            image_masks_DL[image_masks_DL == 255] = 1
            
            #check if they are the same shape, if not zero pad the smaller one
            if not image.shape==image_masks_DL.shape:
                image_masks_DL=pad_image(image_masks_DL,image)


            # Merging the image with the mask
            b, g, r = cv2.split(image)
            b_mask = b * image_masks_DL
            g_mask = g * image_masks_DL
            r_mask = r * image_masks_DL
            mask_img = cv2.merge([b_mask, g_mask, r_mask])

            # Append the name of the image
            features.append(names_DATASET[i])

            # Area, perimeter, Hu moments normalized and eccentricity
            area, perimeter, hu_moments, eccentricity = geometric_moments(image_masks_DL)
            features.extend(hu_moments)
            features.append(eccentricity)

            # Irregularity index
            irregularity_index = irregularity(perimeter, area)
            features.append(irregularity_index)

            # Texture features (GLCM and LBP)
            texture_features = texture(mask_img)
            features.extend(texture_features)

            # Color features (mean, std and skewness by channel)
            color_features = color(mask_img)
            features.extend(color_features)

            # Append on the .csv
            csv_writer(name_features_csv, "a", features)

            # Clear the features list
            features = []
            i += 1

