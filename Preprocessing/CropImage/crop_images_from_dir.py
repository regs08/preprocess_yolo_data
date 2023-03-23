from yolo_formats.default_param_configs import cat_id_map
from yolo_formats.Conversions.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc
from yolo_formats.LoadingData.load_utils import get_yolo_bboxes_from_txt_file
from yolo_formats.Preprocessing.CropImage.crop_utils import get_min_max_bbox_coords_from_bbox_arr
from yolo_formats.Preprocessing.CropImage.crop_single_image_using_A import \
    crop_height_wise_from_left_of_center, \
    crop_height_wise_from_right_of_center

import cv2
import os
import numpy as np


def left_crop_from_center_and_bbox_dat(img_arr, bboxes, class_labels):
    """
    first we get the coords of our max and min bbox is then we use those coords to crop the image down the
    middle
    :param img_arr: img we will crop
    :param bboxes: list of bboxes
    :param class_labels: corresponding class labels
    :return: a dict containing the cropped image, the bboxes, class labels, and a suffix to give a unique filename
    """
    img_h, img_w = img_arr.shape[:2]
    yolo_boxes = bboxes
    pascal_voc_boxes = np.asarray([convert_yolo_to_pascal_voc((img_w, img_h), yolo_b) for yolo_b in yolo_boxes])
    #[xmin, ymin, xmax, ymax] coords of where the bboxes are
    min_max_bbox_arr = get_min_max_bbox_coords_from_bbox_arr(pascal_voc_boxes)


    left_crop = crop_height_wise_from_left_of_center(coords=min_max_bbox_arr,
                                                                 format='yolo',
                                                                 bboxes=yolo_boxes,
                                                                 image_arr=img_arr,
                                                                 class_labels=class_labels)
    return left_crop


def right_crop_from_center_and_bbox_dat(img_arr, bboxes, class_labels):
    """
    first we get the coords of our max and min bbox is then we use those coords to crop the image down the
    middle
    :param img_arr: img we will crop
    :param bboxes: list of bboxes
    :param class_labels: corresponding class labels
    :return: a dict containing the cropped image, the bboxes, class labels, and a suffix to give a unique filename
    """
    img_h, img_w = img_arr.shape[:2]
    yolo_boxes = bboxes
    pascal_voc_boxes = np.asarray([convert_yolo_to_pascal_voc((img_w, img_h), yolo_b) for yolo_b in yolo_boxes])
    min_max_bbox_arr = get_min_max_bbox_coords_from_bbox_arr(pascal_voc_boxes)

    right_crop = crop_height_wise_from_right_of_center(coords=min_max_bbox_arr,
                                                     format='yolo',
                                                     bboxes=yolo_boxes,
                                                     image_arr=img_arr,
                                                     class_labels=class_labels)
    return right_crop


