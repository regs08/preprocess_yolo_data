"""
following functions utilize the albumentations library to crop a single image
"""

import albumentations as A
from yolo_formats.Preprocessing.CropImage.crop_utils import get_bbox_center


def create_transformer_and_crop_image(image, coords, format, bboxes, class_labels, label_fields='class_labels'):
    transformer = create_crop_transformer(coords, format, label_fields)
    cropped_image = crop_image_with_bboxes(image=image, bboxes=bboxes, class_labels=class_labels, transformer=transformer)
    return cropped_image


def crop_image_with_bboxes(image, transformer, **args):
    """
        :param image_arr: image_arr we will patch
        :param transformer: transformer from A. may contain bboxes, class_label args to pass

    :param args: args pertaining to the transform object e.g...
        :param bboxes: bboxes of the image
        :param class_labels: class labels: len must be == to len bboxes
        ...
    :return: a dict containing the transformed image and the adjusted bboxes/masks
    """
    return transformer(image=image, bboxes=args['bboxes'], class_labels=args['class_labels'])


def create_crop_transformer(coords, format, label_fields='class_labels', **args):
    """

    :param coords: coords to crop
    :param format: yolo, pascal, voc, etc..
    :param label_fields: variable name for our classes by default will be class labels
    :param args:
    :return:
    """
    return A.Compose([
        A.Crop(coords[0], coords[1], coords[2], coords[3]),
    ], bbox_params=A.BboxParams(format=format, label_fields=[label_fields], min_visibility=0.3))


def crop_height_wise_from_left_of_center(coords, format, image_arr, class_labels, bboxes):
    """
        takes in 4 coords, of a given image array along  with its bounding boxes and class labels
         finds the center of them and crops the image from the left most to the center
    and then from the right most
    :param coords: coords in pascal voc format [xmin, ymin, xmax, ymax]
    :param format: format style of the bounding boxes
    :param image_arr: img_arr as numpy
    :param class_labels: corresponding class labels to the bounding boxes
    :return: splits the orginal image into two images: left_side_crop, right_side_crop
    """
    x_center, y_center = get_bbox_center(coords)

    left_of_center_crop_coords = [coords[0], coords[1], x_center, coords[3]]
    left_of_center_transformer = create_crop_transformer(coords=left_of_center_crop_coords, format=format)

    left_of_center_transform = crop_image_with_bboxes(image_arr, left_of_center_transformer, bboxes=bboxes,
                                                    class_labels=class_labels)
    left_of_center_transform['filename_prefix'] = 'left'

    return left_of_center_transform


def crop_height_wise_from_right_of_center(coords, format, image_arr, class_labels, bboxes):
    x_center, y_center = get_bbox_center(coords)

    right_of_center_crop_coords = [x_center, coords[1], coords[2], coords[3]]
    right_of_center_transformer = create_crop_transformer(coords=right_of_center_crop_coords, format=format)
    right_of_center_transform = crop_image_with_bboxes(image_arr, right_of_center_transformer, bboxes=bboxes,
                                                            class_labels=class_labels)
    right_of_center_transform['filename_prefix'] = 'right'

    return right_of_center_transform

