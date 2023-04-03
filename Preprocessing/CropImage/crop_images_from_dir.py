from preprocess_yolo_data.Conversions.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc
from preprocess_yolo_data.Preprocessing.CropImage.crop_utils import get_min_max_bbox_coords_from_bbox_arr
from preprocess_yolo_data.Preprocessing.CropImage.crop_single_image_using_A import \
    crop_height_wise_from_left_of_center, \
    crop_height_wise_from_right_of_center
from preprocess_yolo_data.WritingRenamingFile.writing_to_file_utils import insert_string_before_extension


import numpy as np
import os
from PIL import Image

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


def crop_split_images_in_folder(folder_path, save_folder, split_function, **args):
    """

    :param folder_path: where our images are stored
    :param save_folder: save folder
    :param split_function: function we split the images with, e.g vertically, center
    :param args: any
    :return:
    """
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith(".JPG")]

    # Loop through each image file and crop it vertically down the center
    for file_name in image_files:
        # Open the image file
        image_path = os.path.join(folder_path, file_name)
        split_images = split_function(image_path, **args)

        # Save the split images with a number designating the split
        for i, split_image in enumerate(split_images):
            image_filename = insert_string_before_extension(os.path.basename(image_path), f'_{i}')
            split_image_path = os.path.join(save_folder, image_filename)
            split_image.save(split_image_path)



