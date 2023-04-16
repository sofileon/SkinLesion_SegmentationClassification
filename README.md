# Skin Lesion Segmentation and Classification
Advanced Image Analysis and  Deep Learning Final Project - MAIA (June 2022)

Authors: Juan Cisneros, Sofia Leon, Lluis Borras


This repository contains the code necessary to perform segmentation and classification of skin lesion images coming from the [ISIC 2017 dataset](https://challenge.isic-archive.com/data/#2017) that we developed for the courses of AIA and ML-DL courses.
The project included two different skin lesion segmentation approaches: one using classic Image Processing and another one using Deep
Learning. Additionally, two different classification methods of the skin lesions diagnosis (nevus, melanoma and seborrheic keratosis) is done with Machine
Learning and Deep Learning. 


## Requirements

Create a new conda environment and install the requirements via `pip install -r requirements.txt`.

Download our pre-trained weights for the U-Net (segmentation) and the ConvNext (classification) via the following commands:

<details>
<summary>
Download commands
</summary>

```
mkdir checkpoints
cd checkpoints
gdown 1IrG3V-Fc9oXTQEo2Wf0OQq_x2bPWZ_FH
gdown 1pM8GtUysfSQJOcCPIHw-kI9BbQD_FdLg
```
</details>

If you would like to reproduce the results presented, download the [ISIC 2017 dataset](https://challenge.isic-archive.com/data/#2017).

## Skin Lesion Segmentation 

In the first segmentation method we propose the utilization of the K-means algorithm combined with constructed function to remove the undesirable hairs found in some images. We test our model on a 200-image subset of the ISIC 2017 challenge reaching a performance of 81.34% in the Jaccard Index. To process the imagge OpenCV was the main library used. 

In the second approach for the segmentation we build a modified U-Net convolutional neuronal network (CNN) and trained/validates it with the whole train/validation data sets of ISIC 2017 challenge, this method reached a final Jaccard Index of 74.91% in the ISIC 2017 test set.

