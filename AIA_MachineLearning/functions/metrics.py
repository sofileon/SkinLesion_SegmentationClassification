# Import Libraries
import numpy as np


def dice_coeff(y_true, y_pred):
    """
        Dice coefficient is the overlapping between the two segmentations
        DSC = (2 * |X & Y|)/ (|X|+ |Y|)
            = 2 * sum(|A*B|)/(sum(|A|)+sum(|B|))
        :param y_true: ground truth
        :param y_pred: prediction
        -----------
        :return dice_score --> Dice Coefficient
    """
    y_true[y_true == 255] = 1
    y_pred[y_pred == 255] = 1
    dice_score = 2 * np.sum(y_true * y_pred) / (np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred)))

    return dice_score


def jaccard_index(y_true, y_pred):
    """
        Jaccard index for semantic segmentation, also known as the intersection-over-union.
            iou = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                    = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        :param y_true: ground truth
        :param y_pred: prediction
        -----------
        :return iou --> Jaccard Index
    """
    y_true[y_true == 255] = 1
    y_pred[y_pred == 255] = 1
    iou = np.sum(np.abs(y_true * y_pred)) / (np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred)) - np.sum(y_true * y_pred))

    return iou

